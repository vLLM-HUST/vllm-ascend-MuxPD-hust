# MuxPD 接入设计与验证边界

本项目面向 HUST 的 vLLM / vllm-ascend 环境，以可选择的插件方式引入
MuxWise 的阶段分离机制。用户显式选择 MuxPD 调度器后，prefill 处理完整输入，
每次模型调用只包含 prefill 或 decode 请求，取消 token chunk 和 P/D 混合批次。

## 实现层次

| 层次 | 行为 | 当前状态 |
| --- | --- | --- |
| 同阶段批次调度 | 完整 prefill；decode 与 prefill 使用不同批次；共享原引擎内的缓存管理 | 第一版实现路径；端到端正确性和性能以测试回执为准 |
| 串行执行 | 每次运行一个同阶段批次；目标配置为 TP=2、EP 开启 | 当前默认接入路径；没有实现空间并发 |
| `layered` 执行 | 双 runner 分别执行完整 P/D 批次，在层边界交错；一次 scheduler transaction 等待两个阶段均完成 | 已实现；单次完整 P 期间只产生一轮 D 结果 |
| `decoupled` 执行 | P 的完整输入和中间状态跨引擎迭代保留；每轮 D 独立采样、输出，P 使用固定或自适应层数配额 | 已实现协议及生命周期回归；固定配额的设备进度、取消与 API 回执见下文链接 |
| `stream_mode=fenced` | 每个层片段通过事件建立跨流依赖，保留阶段执行隔离 | 双 runner 默认路径；不宣称设备 kernel 同时执行 |
| `stream_mode=overlap` | 实验性独立流、workspace 和集合通信事件门控，取消普通层边界的跨流依赖 | 固定 TP2+EP、8 层配额的设备 profile 已观测 P/D compute 区间重叠；其他条件仍需验证 |
| NPU 资源分区 | 分别控制 P/D 的 Cube / Vector 资源 | 尚未接入控核；待逐算子支持与 profiler 验证 |
| SLO 自适应分区 | 用本机延迟数据和争用测量选择分区及每轮 prefill 层数 | 待实现与校准；不能使用 NVIDIA 的回归参数 |

串行同阶段调度消除了 mixed batch 和 token chunk，但长 prefill 仍会阻塞
decode。它是接入与对照基础，不能称为完整 MuxWise，也不能据此声称改善了
TBT 或实现了算力隔离。实验性双 runner 不自动意味着并发安全或性能更好。

serial 对原生同阶段批次耗时更新 EWMA。layered 使用固定 `layers_per_slice`；
decoupled 默认同样固定，可显式启用下述 TTFT 层数配额。P 等待阈值也受已有
pending cohort 和可用 KV 容量约束，不是请求到达至首 token 的硬上界。

## 固定参考版本

- HUST vLLM：`d0f22d2bda562156e4dbf433ce645e1769b4f804`。
- HUST vllm-ascend：`03766ac696fde5ab1980d80ca0b8543d3580c989`。
- MuxWise artifact：`ykcombat/sglang` 的 `slo_config` 分支，
  `eeac5148f456428a05823c330e1d40afb0ef2438`。
- 本机运行环境的 CANN 目录为 `cann-9.1.0`；兼容环境变量为
  `VLLM_VERSION=0.25.1`。完整运行版本应另外保存于每次设备验证回执。
- 默认模型为 `Qwen3.5-35B-A3B`。它包含混合 attention / Mamba 状态，不能将
  Llama 的 KV-only 正确性推论直接用于这个模型。

源码出处、论文正式版本和具体入口见 [references.md](references.md)。

## 请求和状态约束

完整 prefill 需要足够的 token budget。启动配置应满足
`max_num_batched_tokens >= max_model_len`，并显式关闭 chunked prefill。
超过配置上限的请求应由原有请求校验拒绝，不能通过静默截断或偷偷开启 chunk
来满足调度预算。等待队列、抢占恢复和请求输出仍应服从 vLLM 的状态协议。

普通 decode 不能误包含尚未完成 prefill 的请求。阶段判断必须考虑已计算
token 数、缓存命中及抢占后状态，不能只看请求是否已经进入 running 队列。
对于存在 KV/状态不足的情况，应继续使用原引擎的容量检查与回收机制。

按层执行需要为一个 prefill 保存位置、hidden states、residual、layer cursor、
attention metadata 和相关模型状态。仅首个片段执行 embedding，仅最后一个
片段执行最终 norm、logits 和采样。这里的片段指连续模型层，输入 token 不分块。
prefill 的输出在最后一层完成前不能进入 decode。

双 runner 若共用模型权重和 KV，仍必须隔离输入 staging buffers、attention
workspace、采样状态及 graph 的静态缓冲区；不能因为模型权重只读就假设其余
运行时状态也只读。Mamba state、slot mapping、缓存回收与抢占尤其需要检查。
取消请求后，必须先处理仍在运行的设备任务与事件，才能复用其缓存或状态槽位。

## 跨迭代 prefill 协议

`execution=decoupled` 使用
[`DecoupledMuxPDScheduler`](../src/vllm_muxpd/decoupled_scheduler.py) 与
[`DecoupledMuxPDNPUWorker`](../src/vllm_muxpd/decoupled_worker.py)。每次最多保留
一个 prefill cohort。首次使用原生 scheduler 为整个 prompt 分配 KV，并保存
P 子输出的完整 token 成本；worker 为 P 保存可继续执行的调用栈、输入和中间状态。
D 使用另一个 runner。fenced 后端先完成当轮 D forward，随后推进 P 的
`layers_per_slice` 层；overlap 后端交替发射 D/P 的完整模型层，避免 P 的首个
collective 必须等待整个 D 网络的最后一个 collective。每轮只采样 D 和本轮完成的 P。
若当轮没有 D，worker 一次运行剩余 P 层，避免空闲时人为拖长 TTFT。

原生调度在发射完整 P 时已经增加 `num_computed_tokens`，但此时尚未完成
forward 或产生首 token。本插件显式记录 pending cohort，后续迭代暂时从
原生 running 选择集移除这些请求，并保留其 KV 引用和 worker 槽位。P 完成前
不能进入 decode、被重复发射或被原生 KV 抢占；其他 D 仍可按原生容量规则抢占。
取消但尚未 ACK 的 P 也占用槽位。现有 cohort 结束前不接纳下一组 P。

| 跨进程字段 | 语义 |
| --- | --- |
| `muxpd_prefill_request_ids` | 本轮新发射的 P；后续继续执行时为空 |
| `muxpd_pending_prefill_ids` | 尚未完成或取消 ACK 的全部 P，包含 host 已取消的请求 |
| `muxpd_cancel_prefill_request_ids` | 请求 worker 安全停止或过滤的 P |
| `muxpd_prefill_cohort_id` | 单调递增的 cohort 标识，用于拒绝错序回执 |
| `muxpd_finish_prefill` / `muxpd_pause_prefill` | 无 D 或层数配额要求时完成剩余 P / 防御性暂停控制 |
| `muxpd_completed_prefill_ids` / `muxpd_cancelled_prefill_ids` | worker 的完成及取消 ACK；二者不相交 |
| `muxpd_prefill_layers_completed` / `muxpd_prefill_layers_total` | worker 已提交的层边界计数及总层数；不单独表示设备任务已完成 |

worker 回执只包含当轮 D 和本轮完成 P 的样本。scheduler 先校验 cohort、
样本行及 ACK，再分别调用原生 `update_from_output()` 回收 D 子输出和已保存
的 P 子输出，按客户端合并结果。P 未完成时不执行原生 P retirement；因此 D
可在多次引擎迭代中持续输出，P 的首 token 只在其全部层和最终采样完成后出现。

此模式要求同步 `EngineCore.step()`，显式关闭 async scheduling，且仅允许
一个引擎 batch 在执行。即使 `total_num_scheduled_tokens=0`，也必须执行
worker 和 sampler：该步骤可能继续最后一个 P，或传递取消 ACK。批次队列路径
可能跳过零 token worker 调用，因此拒绝使用 `schedule_batch()`。

## 可选 TTFT 层数配额

`adaptive_layers` 默认为 `false`，只能与 `execution=decoupled` 同时使用。
启用后先观测独立完整 P 和独立 D 的耗时；两类有效样本未齐备时保持固定
`layers_per_slice`。采样窗口是 host 从 schedule 到 worker 回执的墙钟时间，
包含调度、通信和采样开销，不是 profiler 的逐层 kernel 时长。混合 P/D、
悬挂 P 的后续步骤、取消及晚到 abort 不用于这两类独立延迟校准。

P 样本按 token 数的二次幂范围分桶，同桶不同长度先归一到共同 token 数再
更新 EWMA，避免把短输入的耗时直接绑定到长输入。预测优先使用覆盖输入长度的
已有样本，仅在超出最大实测长度时按比例外推。D 使用独立步骤耗时的 EWMA。
`ewma_alpha` 控制更新，`estimate_safety_factor` 放大估计；这些是近似预算，
没有逐层区分 full attention / Mamba，也没有建模当前 batch 大小的争用。

调度器用 cohort 中最早请求的到达时间和 `ttft_slo_ms` 计算剩余 TTFT 预算，
扣除估计剩余 P 耗时，再计算还容得下几轮 D。每轮至少执行 `layers_per_slice`
层；预算越紧，P 配额越大，必要时完成全部剩余层。无 D 时仍直接完成 P。
该策略优先缩短 P 被多轮 D 延后的时间，可能增加 fenced 模式的 D 输出间隔，
因此它不是 TPOT 上界，也不是论文的 contention guard 或资源分区预测器。

trace 中 `decode_slo_feasible_without_prefill` 仅检查独立 D 估计是否已经超过
`tpot_slo_ms`，不包含 P 片段成本。若单独 D 已超过目标，改变 P 层配额不能
使该目标自动可达。应保留固定配额对照并验证真实 TTFT、TPOT 和 goodput。

scheduler 要求回执的总层数与模型一致、累计提交层数与本轮配额一致，且完成 P
必须到达最终层。这个计数用于检测错序和执行进度；缓存释放仍只能依据完成或
取消 ACK，不能因看见最后一个层 hook 就提前复用设备内存。

## 取消、暂停与缓存生命周期

host 取消 P 时，原生请求对象可以立即离开等待/运行队列；插件通过
`pop_blocks_for_free()` 分离块所有权但保留引用计数。在 worker ACK 之前，
这些块不回到空闲池，原生 `finished_req_ids` 通知也会被屏蔽，防止 worker
提前删除悬挂 P 的输入行。收到 ACK 后才释放缓存并清理 worker 元数据。

部分取消无法直接从已经执行若干层的 hidden-state 批次中删行。worker 记住
取消 ID，继续执行其余 cohort，并在完成后过滤取消样本、统一 ACK。整组取消
则关闭悬挂调用栈、同步 P stream，再 ACK。若取消发生在 worker 完成和原生
结果处理之间，允许晚到的 completed 回执，释放持有的块并忽略已取消请求的样本。
完整取消控制必须获得整组 ACK，禁止静默留存无法推进的 cohort。

存在未 ACK 的 P cohort 时，`PAUSED_ALL`（keep pause）在修改暂停状态前
直接拒绝，避免原生 keep 流程随后清空缓存。`PAUSED_NEW` 可继续 drain 或
abort，等待取消 ACK。`reset_prefix_cache(reset_running_requests=True)` 同样
拒绝悬挂 cohort；不能通过 reset 抢占持有中间状态的 P。worker 配置另行拒绝
sleep mode。原生 deferred frees、KV/EC connector、结构化输出、LoRA、返回
routed experts 和有状态输入续写不在 decoupled 支持范围内。
此外拒绝 `enable_mfu_metrics`：原生 MFU 核算依赖单轮新请求/缓存请求 metadata，
不能直接解释跨迭代 P 的分阶段 retirement；不能以空 metadata 的零 FLOPs 充作测量。

## Ascend 执行路径

原 artifact 用 CUDA GreenContext 将 stream 绑定到 SM 资源。Ascend 的候选
替代接口为 `torch_npu.set_stream_limit(stream, cube_num, vector_num)`，
配合 `torch_npu.npu.config.allow_internal_format = False`。该接口只保证其
支持的 Ascend C 算子按相应资源限制运行；两个 NPU stream 本身不能证明
实现了独立算力分区。

设备验证应检查各算子的 `BlockDim` 与流资源设置是否一致，并确认设备时间线
存在 P/D 执行重叠。Cube 与 Vector 核数要分别记录；不能将 CUDA SM 数字
直接转换成同名配置。普通流优先级也不能替代资源限制。

事件用于确定已发射层和最终 prefill 的完成状态，避免全设备同步。
重分区需保证旧流上的相关任务完成及新流正确等待依赖。若启用 ACL graph，
必须验证 capture 时的资源约束、graph 复用和静态缓冲区生命周期。
初期 eager 验证通过前不应假设 graph 路径可用。

TP/EP 的不同阶段可能同时发起集合通信。独立计算流不足以保证 collective
顺序正确；需要验证 HCCL communicator、通信流与各 rank 的完成一致性。
参考 artifact 为 P/D 复制 TP communicator。实验实现的 TP/EP 支持范围
必须由实际多卡验证确定。

`stream_mode=overlap` 的实验路径见
[`overlap.py`](../src/vllm_muxpd/overlap.py)。它为 P/D 分别维护 stream、forward
context 和 workspace，以事件发布共享初始化工作；同一 device process group
上的 `all_reduce`、`all_gather` 和 `reduce_scatter` 按主机发射顺序门控。
它依赖原生同步 collective API 已将内部 HCCL 工作连接回调用 stream，且要求
各 rank 发射顺序一致。普通模型层计算可摆脱整个上一层的跨流等待，但这不等于
已经验证计算重叠，更不等于 CUDA GreenContext 式硬件资源隔离。

当前 worker 的实验白名单为 Ascend A2、Qwen3.5 BF16、eager、text-only，
禁用 MC2、FlashComm、动态 EPLB、shared-expert 多流、原生 DBO/ubatching；
overlap 还要求关闭 fine-grained TP。这些限制用于界定需要验证的算子和通信路径。
当前没有调用 `set_stream_limit()`，也没有接入论文的在线争用修正与资源预测器。

当前设备证据见 [2026-10-10 回执](../evidence/2026-10-10/README.md#device-profile-independent-pd-compute-intervals)。
在指定 Qwen3.5-35B-A3B BF16 / Ascend 910B2 / TP2+EP / 固定 8 层配额的
capture 中，两 rank 的 P/D compute 区间交集分别约占活跃并集的 10.78% 和
9.16%。计算使用已映射到 P/D 的 CANN physical stream，排除 CPU、拷贝、
event wait 和通信区间。它证明该 capture 存在设备 compute 任务重叠，不能
证明同时的每核占用、空间资源隔离、完整工作负载利用率或吞吐提升。profile
有记录窗口结束时的停止告警；导出区间已检查，但不能据此保证未记录任务的完整性。

## 验证与测量

`scripts/smoke_serving.py` 只提交固定合成数据，保存响应、usage 和时间回执：

```bash
python scripts/smoke_serving.py \
  --base-url http://127.0.0.1:8002 \
  --output artifacts/smoke-muxpd.json
```

检查内容包括 greedy 算术、并发中文 SSE、2305-token 完整输入，以及主动关闭
长生成连接后服务 health 和后续请求是否正常。SSE 回执要求 `[DONE]`、
finish reason 和 usage；长输入用显式 token IDs 校验实际 prompt token 数，
避免字符长度被误认为 token 长度。取消用例验证客户端断开后的服务可用性，
并不单独证明设备任务已立即停止或缓存已全部释放。

设备正确性还应比较固定模型、固定输入、相同采样参数下的 baseline 与插件
输出，覆盖缓存压力、抢占恢复、取消、混合状态模型和不同请求长度。
greedy 输出差异需区分正常数值扰动与状态/顺序错误，不能仅用 HTTP 200 验收。

`scripts/compare_outputs.py` 可对两个配置分别运行同一组自然语言契约：
先逐个请求，再双并发；第二轮交换单请求顺序，检查前一请求状态是否泄漏。
长输入在计数请求已输出首个内容事件、且仍未完成时发射。脚本保留完整合成文本、
SSE、usage、功能 oracle、重复稳定性及文本哈希；`--require-exact` 可额外要求
两个服务逐文本一致。客户端请求重叠仍需 scheduler trace 确认 P 实际跨迭代。
`--baseline-report` 可复用已完成回执中的 baseline 端点，只请求新 candidate；
复用前验证模型、重复次数、case 内容及功能结果、覆盖范围和文本哈希。
该脚本用于正确性比较，复用历史 baseline 的时间数据不构成同条件性能比较。

同一 prompt 的单请求结果在相同配置下应先具有可重复性，再比较两个模式的
单请求和并发结果。并发 batch 形状可改变 BF16 数值路径，文本不一致本身既
不能证明状态损坏，也不能证明只是正常数值误差。需要定位首个分歧，必要时在
相同 batch 形状和强制相同历史 token 的条件下比较 logits。少量契约全部通过
只说明这些用例通过，不能替代长期随机、取消与内存压力测试。

性能测试应固定相同模型、并行方式、缓存预算、请求到达序列和输出长度，
分别报告 TTFT、逐 token TBT、请求级 TPOT、吞吐量及给定 SLO 下的 goodput。
一个 SSE content event 可能携带多个 token，因此 inter-event gap 只能作为
流式响应间隔，不能直接冒充逐 token TBT。所有性能结论都需要保存原始回执，
未经测量不报告收益倍数。

[`test_decoupled_scheduler.py`](../tests/test_decoupled_scheduler.py) 使用真实
vLLM scheduler 和 KV allocator，覆盖 D 跨轮输出、完整 P 成本、槽位预留、
KV 抢占、部分/全部取消、晚到取消、首 token 即结束、错序回执、keep pause
提前拒绝以及原生 `EngineCore.step()` 的零 token continuation/ACK 路径。
这些主机侧回归不能替代真实 worker 的模型输出、设备访存和多卡通信验证。
trace 的 `decoupled_schedule` / `decoupled_complete` 记录 cohort、pending P、
本轮 D 和 ACK；后续 P 层执行不会再次记入 prompt token admission。
