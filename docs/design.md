# MuxPD 接入设计与验证边界

本项目面向 HUST 的 vLLM / vllm-ascend 环境，以可选择的插件方式引入
MuxWise 的阶段分离机制。用户显式选择 MuxPD 调度器后，prefill 处理完整输入，
每次模型调用只包含 prefill 或 decode 请求，取消 token chunk 和 P/D 混合批次。

## 实现层次

| 层次 | 行为 | 当前状态 |
| --- | --- | --- |
| 同阶段批次调度 | 完整 prefill；decode 与 prefill 使用不同批次；共享原引擎内的缓存管理 | 第一版实现路径；端到端正确性和性能以测试回执为准 |
| 串行执行 | 每次运行一个同阶段批次；目标配置为 TP=2、EP 开启 | 当前默认接入路径；没有实现空间并发 |
| 按层交错执行 | 实验性双 runner 保留 prefill 的中间状态，在层边界安排 decode | 正在开发，待正确性、状态隔离及设备验证 |
| NPU 空间并发 | 独立流同时运行，分别控制 Cube / Vector 资源 | 待设备时间线与逐算子控核验证 |
| SLO 自适应分区 | 用本机延迟数据和争用测量选择分区及每轮 prefill 层数 | 待实现与校准；不能使用 NVIDIA 的回归参数 |

串行同阶段调度消除了 mixed batch 和 token chunk，但长 prefill 仍会阻塞
decode。它是接入与对照基础，不能称为完整 MuxWise，也不能据此声称改善了
TBT 或实现了算力隔离。实验性双 runner 不自动意味着并发安全或性能更好。

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

性能测试应固定相同模型、并行方式、缓存预算、请求到达序列和输出长度，
分别报告 TTFT、逐 token TBT、请求级 TPOT、吞吐量及给定 SLO 下的 goodput。
一个 SSE content event 可能携带多个 token，因此 inter-event gap 只能作为
流式响应间隔，不能直接冒充逐 token TBT。所有性能结论都需要保存原始回执，
未经测量不报告收益倍数。
