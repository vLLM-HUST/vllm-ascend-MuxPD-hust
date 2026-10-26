# MuxPD for HUST vLLM Ascend

维护主体：[vLLM-HUST](https://github.com/vLLM-HUST)。缺陷、兼容性和发布问题请提交到
[本仓库 Issues](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues)。

`0.2.0a1` 提供可安装的 ECPA 0.3 Bundle，包含成对的 scheduler、admission policy
和 worker。安装不自动启用；Manager 显式配置后统一投影三个入口，并根据实际输出
记录运行状态。安装、启用、停止和恢复原生服务见 [ECPA 部署说明](docs/ecpa-deployment.md)。
版本发布和实机验收由 [Issue #21](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/21)
跟踪；这次接入本身不构成新的性能收益结论。

新增默认关闭的 [异步 Decode 调度入口](docs/experimental-async-decode.md)：
纯 D 阶段复用原生异步队列，进入 P 前排空队列，完整 P 仍通过独立模型 batch
推进。该入口需要新的成对 scheduler/worker 和显式配置，原有同步默认不变。
完整普通 API 启动命令、适用边界和本轮实验入口见该说明；Manager 尚未对这条
路径完成资格验证，不能直接套用现有 Manager 启动命令。

Historical data for the migrated batch is in [benchmark custody](EVIDENCE.md),
with pinned hashes and an external-directory downloader.

当前实现提供默认关闭的 [group-FX 入口](docs/experimental-prefill-group-fx.md)和
[确定性通信选项](docs/ecpa-deployment.md#实验性确定性通信配置)。
[#11](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/11#issuecomment-6405978497)
已按显式 TP4 配置与冻结输入的有界调查完成关闭；历史失败与默认配置的数值限制仍保留。
source98 的普通 worker 和 Manager 服务桥接、后续性能与生命周期记录统一见
[中央 follow-up 入口](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ef849ef49577c491fee68317ffc96d52796cd379/reports/muxpd-followup-20261025)。
完整质量、实用原版默认后端参考和实际混合 profile 已完成，见
[当前结果与适用范围](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ebddb585390c333ac1f8c6ea40cfeead3c4321af/reports/muxpd-followup-20261025/quality-reference-v5)。
f7 raw-P 与 group-P 在固定题集的完整输出一致，但两者均未通过预先冻结的
答案不退化门槛；总正确数增加不能抵消原生控制均答对题目上的新增错误。
该已知质量问题继续由 [#23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23) 跟踪。
上一轮同步接入的价值是长 prefill 插入时较短的最坏出词停顿；相对实用原版参考，吞吐、
首字延迟和常规出词间隔的尾部表现均更差。
因此 [PR #10](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/pull/10)
仅提供显式启用的实验优化，不能视为通用替代或答案质量通过。

上述 source98 设备桥接实际使用
`98eac12dc2595a8742a81b58e96b899fed905194`，属于上一轮固定源码证据。
本轮异步调度的测量另存于
[调度实验报告发布路径](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/8b5a7d57c94bb34b1756ee7c867dbe72094bcd9c/reports/muxpd-scheduling-20261026/)
（以报告中的源码和配置身份为准）。已发布包、默认 raw-P 配方和实验候选
分别评估；异步调度修改没有解决上述答案质量问题。

将 MuxWise 的 P/D 解耦机制适配到现有 HUST vLLM，使用原生 KV 分配、抢占、
取消、采样与 OpenAI API。安装包本身不修改 vLLM；通过显式 scheduler/worker
类启用，无需改动基线源码。

**状态：raw-P 固定环境集成已验证；可选实验路径按各自证据范围评估。** 当前实现包含完整 prefill、独立 P/D 模型
batch；串行策略具有在线延迟估计和等待阈值。实验 worker 使用独立输入/metadata 与共享
权重/KV，通过 transformer 层边界切换。完整层 event fence 是正确性起点，
不代表设备 kernel 重叠或论文中的空间资源分区。`decoupled` 模式将完整 prefill
保留在 worker 中，允许多轮 decode 在其完成前独立提交结果；取消时等待 worker
确认停止访问 KV 后再释放缓存。

| 模式 | P/D 执行与结果提交 |
|---|---|
| `serial` | 每轮只有完整 P 或 D，作为参考实现 |
| `layered` | 独立 P/D 模型 batch，按层切换，同轮提交结果 |
| `decoupled` | 完整 P 跨轮保存，D 每轮独立完成，延迟提交 P 结果 |

`stream_mode=fenced` 对两流建立完整事件依赖；实验性 `overlap` 保留共享初始化、
同组 collective 和采样完成所需依赖，取消普通层边界的跨流等待，并隔离两条
lane 的临时 workspace。设备并行程度必须由
NPU profiler 验证，不能由 Python 协程或调度 trace 推断。
当前固定配方未启用 stream 资源配额；它沿用开启内部格式的原生 V1 runner，
不满足该配额接口要求关闭内部格式的前提，因此不表示论文的 Cube/Vector 空间分区。

以下性能段落记录历史配置结果；版本、负载和代价见各自报告，不能直接赋予当前实验候选。

TP4 调优加入独立 decode 图、完整 P 编译、按 cohort token 工作量分配层数，
并减少连续 D 的 host 调用与事件依赖。两组负载各六轮测量的聚合吞吐高于
原生 graph 基线；自然语言 16/16 对照一致，但部分合成序列仍有差异，且部分
延迟指标未改善。完整结果与限制见 [TP4 性能报告](docs/performance-tp4.md)。
这不是对任意负载或全部指标不退化的保证，也不是论文硬件分区的实现。

八卡 TP8 迁移验证复用同一配置：每方每负载六轮，混合负载吞吐基本持平
（177.302 → 177.314 tokens/s），decode-heavy 为 243.153 → 244.002 tokens/s。
阶段波动大于合并收益，不能宣称稳定加速；mixed 的请求平均 TPOT P95 上升
0.87%。自然语言 16/16 对照一致，合成用例文本 33/56 一致。
方法、原始回执及输出稳定性限制见 [TP8 实验报告](docs/performance-tp8.md)。

持续 D 加长 P 的 TP8 争用实验取得明确的延迟取舍：启用可选
`decode_budget_guard` v2 后，三种长 P 负载各六轮的 D 最大事件间隔 P95
从原生 16K 的 385–398 ms 降到 66–68 ms，所有请求满足本轮 D 100 ms /
P TTFT 1 s 观察目标；吞吐低 1.96%–3.54%，P TTFT P95 增加 22.62%–28.49%。
原生 2K chunk、旧插件、无注入 P 控制及所有重复轮次均保留。
自然语言为 16/16 文本精确一致，合成 token IDs 仍有重复及跨配置差异。
这不是硬 SLO 保证或设备空间分区；guard 默认关闭，完整启用示例、图表与
全部证据见 [TP8 争用实验报告](docs/performance-tp8-contention.md)。

持续到达的容量验证使用独立 Poisson plan 和严格原始 SSE 回执，区分 pooled
TBT P99 与每请求双 SLO goodput，并调优原生 chunk budget / async 基线。
实验性 **isolated graph overlap** 的固定 8 层入口与运行方法见
[Goodput 实验说明](docs/goodput-experiments.md)。默认 graph 脚本仍为 fenced；
新路径尚无持续容量收益结论。同条件 OMP1 的逐档结果、goodput 改善信号与
高负载吞吐代价见 [TP8 持续到达探索进展](docs/performance-tp8-goodput-exploration.md)。

针对短输出流与长 prefill 争用的 TP4 交互场景，新增显式入口
[`serve-qwen35-streaming.sh`](scripts/serve-qwen35-streaming.sh)：raw P 四层切片、
独立 decode 图和通信组。三个独立到达 seed、交换卡组的六个确认窗口中，
满足 TTFT≤1 s 且每个单 token 间隔≤100 ms 的请求由 **117/378 增至 368/378**，
Goodput 为原生 16K 对照的 **3.15×**；固定窗口吞吐约 **+0.48%**，视为基本持平。
收益是减少每请求最坏停顿，代价是 TTFT 和 pooled token-gap P99 增加。
适用负载、全部对照和复现证据见 [TP4 Goodput 报告](docs/performance-tp4-goodput.md)。
这是特定到达率下的交互 SLO 收益，不是最大吞吐或任意负载加速保证。

早期 [TP2 验证](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/2144d58de108cddaff3800fdcbf12dc9d3ab1d18/archive/mod-consumers-20261023/muxpd/manifest.json) 和
[TP4 eager 验证 (archive member: `evidence/2026-10-10-tp4/README.md`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) 保留原始回执与未取得收益的结果。
不同 TP、eager/graph、缓存设置的测量不能直接混用。

## 固定运行基线

| 组件 | 验证目标 |
|---|---|
| vllm-hust | `d0f22d2bda562156e4dbf433ce645e1769b4f804` |
| vllm-ascend-hust | `03766ac696fde5ab1980d80ca0b8543d3580c989` |
| 模型 / 设备 | Qwen3.5-35B-A3B / Ascend 910B2，BF16，TP2+EP、TP4+EP、TP8+EP |
| 执行 | V1 NPU runner，纯文本；默认同步，异步 D 须使用独立实验入口；eager 或显式 decode graph |
| 底层依赖 | 复用已运行环境，不重新解析 torch / torch-npu / CANN |

参考正式 [ASPLOS 2026 论文 v3](https://arxiv.org/html/2504.14489v3) 与作者指定
[`slo_config` 分支](https://github.com/ykcombat/sglang/tree/eeac5148f456428a05823c330e1d40afb0ef2438)。
实现与参考差异见 [设计](docs/design.md) 和 [来源](docs/references.md)。

## 安装与启用

以下命令针对服务器现有 `/root/workspace/runtime/env.sh`；其他路径可设置
`MUXPD_WORKSPACE`。使用 `--no-deps` 保持已验证的 Ascend 软件栈。

```bash
source /root/workspace/runtime/env.sh
uv pip install --python /root/workspace/.venv/bin/python --no-deps -e .

# 串行参考实现：每批只做 P 或 D，不切分 prompt。
ASCEND_RT_VISIBLE_DEVICES=2,3 scripts/serve-qwen35.sh \
  --batch-admission-policy-config '{"trace_path":"local-results/serial-{pid}.jsonl"}'
```

默认监听 `127.0.0.1:8002`。示例使用逻辑 NPU `2,3`；实际可见编号以当前容器为准，
设备重新挂载后不要沿用旧逻辑/物理映射。启动前需停止占用所选设备的旧进程。脚本显式关闭 chunked prefill、
async scheduling 和 prefix caching，并将 token budget 设置为完整 16K 上下文。
**Qwen3.5 的 Mamba `align` 模式本身会拆 prefill**，因此使用 `none`。
这会改变缓存条件，性能比较必须保留相同缓存设置的对照组。

启动脚本默认 TP=2；四卡使用 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 MUXPD_TP_SIZE=4`。
TP4 占用所选四张 NPU，TP8 占用八张，不能与占用相同设备的服务同时运行。
早期四卡环境的完整启动与复测命令见
[TP4 记录 (archive member: `evidence/2026-10-10-tp4/README.md`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)。

独立 decode 图使用以下入口。P 保持完整 prompt，不进入设备图；D 使用自己
的静态输入、workspace 和捕获流；两者共享模型权重与 KV。该入口关闭详细 trace，
启用自适应层配额、prefill-first、完整 P 编译和 262144 layer-tokens 工作量预算。
该配置已在本机 TP4 和 TP8 测量；其他 TP 或模型需重新验证。图的收益应与同样开启 decode 图的原生服务比较，
调查与逐项证据见 [TP4 性能报告](docs/performance-tp4.md) 和 [TP8 实验报告](docs/performance-tp8.md)。

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 MUXPD_TP_SIZE=4 \
  scripts/serve-qwen35-decode-graph.sh

# 八卡：与上面的四卡示例择一启动。
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MUXPD_TP_SIZE=8 \
  scripts/serve-qwen35-decode-graph.sh
```

```bash
# 实验分层执行：两套 runner 输入与元数据，模型权重/KV 共用。
uv pip install --python /root/workspace/.venv/bin/python 'greenlet>=3.1,<4'
ASCEND_RT_VISIBLE_DEVICES=2,3 VLLM_USE_V2_MODEL_RUNNER=0 scripts/serve-qwen35.sh \
  --scheduler-cls vllm_muxpd.multiplex_scheduler.LayeredMuxPDScheduler \
  --worker-cls vllm_muxpd.worker.MuxPDNPUWorker \
  --batch-admission-policy-config \
  '{"execution":"layered","layers_per_slice":1,"trace_path":"local-results/layered-{pid}.jsonl"}'

# 跨轮执行：完整 prompt 不切 token，decode 在 P 完成前持续推进。
ASCEND_RT_VISIBLE_DEVICES=2,3 VLLM_USE_V2_MODEL_RUNNER=0 scripts/serve-qwen35.sh \
  --scheduler-cls vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler \
  --worker-cls vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker \
  --batch-admission-policy-config \
  '{"execution":"decoupled","stream_mode":"fenced","layers_per_slice":8,"trace_path":"local-results/decoupled-{pid}.jsonl"}'
```

显式 `scheduler_cls` 是插件入口；不依赖 `vllm.general_plugins` 全局 monkey patch。
不要仅将 `MuxPDPolicy` 加到原生 scheduler：必须和本仓库 scheduler 配对。
不满足配置契约会报错，不能静默退回 mixed batch。

## 验证与性能记录

```bash
source /root/workspace/runtime/env.sh
/root/workspace/.venv/bin/python -m pytest tests -q
/root/workspace/.venv/bin/python -m ruff check .

/root/workspace/.venv/bin/python scripts/smoke_serving.py \
  --base-url http://127.0.0.1:8002 --output local-results/smoke.json
/root/workspace/.venv/bin/python scripts/benchmark_serving.py \
  --label muxpd-serial --output local-results/benchmark.json
/root/workspace/.venv/bin/python scripts/check_trace.py local-results/serial-*.jsonl
```

CPU 测试覆盖策略与协程；原生 Scheduler 测试覆盖 KV 容量、抢占重算、取消、
完整 prompt 和调度入口。实机脚本覆盖流式/并发生成、2305-token prompt 和取消后
继续服务。benchmark 保存合成负载的原始 SSE 时间、TTFT、平均 TPOT、吞吐和
给定 SLO 下的 goodput；SSE 事件间隔不会被冒充为单 token 延迟。

## 配置和支持边界

`--batch-admission-policy-config` 接收严格 JSON。主要字段：

| 字段 | 默认值 | 作用 |
|---|---:|---|
| `max_decode_steps` | 8 | 串行模式下连续 D 的上限 |
| `max_prefill_wait_ms` | 100 | 串行 P 等待阈值，仍受 KV 可用容量约束 |
| `ttft_slo_ms` / `tpot_slo_ms` | 1000 / 100 | 策略目标，不是服务保证 |
| `max_prefill_requests` | 8 | 单批 P 请求数上限 |
| `max_prefill_batch_tokens` | 0 | 软 batching 上限；0 使用完整预算 |
| `ewma_alpha` | 0.2 | 串行模式的调度至完成延迟在线估计权重 |
| `layers_per_slice` | 1 | 基础完整 transformer 层数；decode budget guard 可将实际配额降至 1 |
| `prefill_layer_token_budget` | 0 | decoupled 的层工作量预算；非零时按完整 P cohort 的 token 总数计算基础层配额 |
| `execution` | serial | 必须与 scheduler/worker 类匹配 |
| `stream_mode` | fenced | `overlap` 是需独立验证的双流执行后端 |
| `adaptive_layers` | false | decoupled 按本机耗时和 TTFT 预算放大层配额 |
| `decode_budget_guard` | false | adaptive fenced 按观测的 D 和 P 片段耗时限制层配额；仅是延迟预算启发式 |
| `prefill_first` | false | adaptive fenced 判断 P 可在本轮完成时先执行 P，下一轮偿还 D 执行机会 |
| `decode_graph` | false | decoupled 的独立 FULL_DECODE_ONLY 图；默认 fenced，实验 overlap 须 isolated 通信及受支持的 P 路径 |
| `compile_full_prefill` | false | decode graph 模式中，新 P 可一次完成时复用原生编译路径；需启动预热 |
| `compile_prefill_groups` | false | 默认关闭的四层 FX P 分组；仅支持专门的固定 q4/fusion 配方，见 [group-FX 说明](docs/experimental-prefill-group-fx.md) |
| `async_decode_pipeline` | false | 纯 D 使用原生异步队列，P 前排空；须显式异步 scheduler/worker 配对，见 [启动说明](docs/experimental-async-decode.md) |
| `require_hccl_deterministic` | false | 要求启动前的 HCCL 环境匹配显式确定性通信配置；不是任意输出确定性保证 |
| `overlap_communication` | shared | 实验 `isolated` 为 overlap 的 P 创建独立 TP/EP 通信组 |
| `trace_path` | null | 可选 JSONL 文件；支持 `{pid}`，不记录 prompt 内容 |

`prefill_layer_token_budget` 非零时，基础层配额为
`max(layers_per_slice, min(模型层数, budget // cohort完整P token数))`，再裁剪到剩余层数。
分母包括 cohort 全部成员，取消部分成员后也不缩小；它不是 KV 容量或硬时间上限。
默认 0 保留固定基础配额。`adaptive_layers=true` 在获得完整 P-only 和 D-only 样本后，
可根据截止时间进一步放大配额。它优先减少 P 的首 token 等待，可能增加个别 D 的响应间隔，不保证
TPOT 达标，也不进行硬件资源分区。详细算法见 [设计](docs/design.md)。

可选 `decode_budget_guard=true` 只允许与 `execution="decoupled"`、
`adaptive_layers=true` 和 `stream_mode="fenced"` 同时使用；其他组合会报错。
组件与现有 graph 启动脚本均默认不启用。它在工作量/TTFT 配额计算后增加 D
延迟预算限制，可以低于 `layers_per_slice`，但每次至少推进一层，避免 P 永久
等待。缺少匹配的 P 或 D 耗时样本时先推进一层；此时 `prefill_first` 不能绕过
限制。完整 P 的 token 准入不变：即使它占满本轮 token 预算而没有选中 D，
已有 D 仍计入预算；只有没有存活的可运行 D 时才一次完成剩余 P。

估计区分 P 首层准备成本与续层成本，按原 cohort 的完整 token 长度组合匹配；
D 按 batch 大小及上下文长度分桶。有效的已完成 lane 计时优先，缺少该计时时
使用已完成整步 host 耗时的保守后备估计，整网编译 P 不训练逐层成本。
P 续片优先使用已完成 fenced 续片的“整步 host 耗时减 P 设备跨度”估计非 P
成本；它包含 D、通信及其他开销，不是纯 D 耗时。没有匹配样本时使用独立 D
观测；新 P 和 `prefill_first` 始终使用独立 D 观测，不复用续片残差。
取消、暂停和晚到 abort 不用于学习。它不增加同步来获取设备时间，也不保证
TTFT、逐 token TBT 或平均 TPOT 达标：未知形状、单层已经超预算、跨 rank
等待等仍会影响实际延迟。trace 会记录预算冲突和 TTFT 可行性；回执/配额校验
不能代替实际服务测量。配置示例及估计范围见
[decode budget guard 设计](docs/design.md#可选-decode-budget-guard)。
P、独立 D、续片非 P 残差三个估计缓存各限 256 个条目，按最近使用顺序逐出；
逐出后使用可用的后备观测或重新冷启动，不改变请求、KV 或取消状态。
trace 记录估计来源、条目数和累计逐出次数。

初始延迟系数是可配置启动值，不能作为 MuxWise 论文 contention predictor 的复现。
PP/DP/CP、speculative decoding、KV/EC transfer、streaming-input continuation
不在当前范围。实验 worker 进一步限制 Qwen3.5、A2、BF16、无 LoRA、
无结构化输出/logprobs、无 FlashComm/共享专家侧流/EPLB。decoupled 拒绝 MFU
指标选项，避免拆分结果退休后向原生统计模块传入不完整的 FLOPs 元数据。
`decoupled` 当前只允许一个未完成的 P cohort，保留其 KV 与请求槽位；不支持
存在未完成 P 时的 `keep` 暂停，需先 drain 或 abort。完整 prompt 过长或 KV
不足仍会造成等待，层调度不能消除内存容量限制。

默认 worker 仍要求 eager。`decode_graph=true` 必须配合 `--no-enforce-eager`、
`FULL_DECODE_ONLY`、编译 mode 3 和明确的捕获大小（当前脚本为 1/2/4/8）；
实验 overlap 图要求 isolated 通信；普通 raw P 配方见
[Goodput 实验说明](docs/goodput-experiments.md)，默认关闭的四层分组 P 则受
[group-FX 专用约束](docs/experimental-prefill-group-fx.md)限制。不支持 shared overlap 图、
ENPU、offload 或运行中重新捕获。图只覆盖 D，首次单 token
P 也强制 `NONE`。`trace_path=null` 不构造逐轮诊断记录；需要生命周期证据时
显式开启 trace，并在计时报告中说明。
Ascend 原生 GraphParams 包含进程级状态；当前不支持在同一 worker 进程内销毁后
重新初始化图服务，重启服务需使用新的 worker 进程。

`compile_full_prefill` 保持 P 的图模式为 `NONE`，只开放原生编译函数的调用。
新 P 的层预算必须覆盖整个模型；已暂停的 P 继续通过 Python 层边界推进。
整网返回来源记为 `full_forward_return`，与 `layer_hooks` 区分；两者都必须
经过原生采样同步后才能确认完成和释放 KV。该来源字段本身不证明 FX 命中或
设备图执行，性能与执行路径需结合启动记录和 profiler 判断。

回滚：停止本插件服务，使用原 `/root/workspace/runtime/serve-qwen35.sh`。
基线源码与原启动脚本不由安装过程修改。
