# MuxWise 来源与迁移依据

## 正式论文与 artifact

- Yukang Chen 等，*Towards High-Goodput LLM Serving with Prefill-decode
  Multiplexing*，ASPLOS 2026 Volume 2；
  [DOI: 10.1145/3779212.3790236](https://doi.org/10.1145/3779212.3790236)。
- [arXiv:2504.14489v3](https://arxiv.org/html/2504.14489v3)，2026-02-07。
  早期 v1/v2 采用 Drift 名称与不同实验结果；本项目使用 v3 的设计描述。
- [Zenodo artifact](https://zenodo.org/records/18062118)。
- 论文 Appendix A.4 指向
  [ykcombat/sglang 的 slo_config 分支](https://github.com/ykcombat/sglang/tree/slo_config)。
  本次核对固定到 `eeac5148f456428a05823c330e1d40afb0ef2438`
  （2025-12-16，`upload configuration`）。

MuxWise 将同一进程内的 P/D 执行与共享内存结合，按层推进 prefill，并通过
事件查询完成合并。它用独立延迟预测与争用修正选择 decode 资源，将其余资源
分给 prefill。该模型必须在目标硬件和模型上校准；论文的 GPU 实验不构成
Ascend 性能或 SLO 保证。[论文 §3](https://arxiv.org/html/2504.14489v3#S3)

## 可追溯的代码入口

以下链接均固定到上述 artifact commit。

| 入口 | 作用 |
| --- | --- |
| [multiplexing.py](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/multiplex/multiplexing.py) | `SchedulerMultiplexMixin`，分别持有 decode batch 和 split-prefill batch；decode 先发射，event query 和跨 rank 完成确认后合并 |
| [pdmux_context.py](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/multiplex/pdmux_context.py) | `PDMuxConfig`，预创建资源分区和 stream groups，包含纯 P / 纯 D 两种状态 |
| [greenctx_stream.cu](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/sgl-kernel/csrc/spatial/greenctx_stream.cu) | CUDA Driver API 的资源拆分、GreenContext 和 nonblocking stream 创建 |
| [llama.py](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/models/llama.py#L496) | `forward_split_prefill()`，保存 hidden states/residual，按层继续执行 |
| [tp_worker.py](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/managers/tp_worker.py#L287) | split-prefill 的 `ForwardBatch` 保留与后续复用 |
| [model_runner.py](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/model_executor/model_runner.py#L1705) | 独立 prefill workspace 与按分区维护的 decode attention backend |
| [cuda_graph_runner.py](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/model_executor/cuda_graph_runner.py#L481) | 每个 decode stream 分别 capture，以 stream index 和 batch size 保存 graph 与 output buffers |
| [parallel_state.py](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/distributed/parallel_state.py#L1440) | prefill 与 decode 使用独立 TP communicator，通信采用当前 stream |
| [server_args.py](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/server_args.py#L2745) | artifact 拒绝 PP、chunked prefill、PD disaggregation 和原 overlap scheduler 的组合 |

## 论文与公开实现的差异

`slo_config` 的 `adjust_stream_groups()` 仍有 temporary demo 注释。
它按 decode batch size 阈值或 `decode_bs_divisor` 选资源分区；每轮层数为
`max(1, split_forward_token_budget // extend_num_tokens)`。代码中未包含
论文完整的 contention guard、在线修正和可选 prefill 抢占机制。
这里的 token budget 用于决定每轮执行多少层，并非将 prompt 切成 token chunks。

另一个公开分支 `slo_scheduler` 的 commit 为
`a068b3e484b09325d926c920643d59ca20084ca0`（2025-09-02）。其中
[predictor.py](https://github.com/ykcombat/sglang/blob/a068b3e484b09325d926c920643d59ca20084ca0/python/sglang/srt/multiplex/predictor.py)
包含硬编码的 A100 / H800 / RTX3090 回归参数；核对时未在该分支的调度路径中
发现 predictor 调用。它只能作为预测接口的研究参考。

默认 `main` 的核对 commit 为 `3f3578ee7d1fc7a0290b99dc1440d70979f107cb`
（2025-06-27），不是本项目采用的 artifact 基线。以上日期和状态对应本次
固定 commit，不推断后续分支是否继续更新。

## Ascend 的对应能力

- [torch_npu.set_stream_limit 官方说明](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-set_stream_limit.md)：
  分别设置 stream 的 Cube / Vector 核数；仅支持相应的 Ascend C 算子控核，
  需要 `allow_internal_format=False`；可通过 profiler 的 `BlockDim` 检查。
- [CANN 资源限制优先级与 capture 注意事项](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_1879.html)：
  stream 限制优先于进程限制；资源设置作用于后续任务，graph capture 需提前设置。
- [Ascend C kernel launch 约束](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00021.html)：
  算子启动的核数必须遵守可用核数，否则可能破坏其他 stream 的资源限制。

这些 API 为迁移提供候选机制，是否满足本机算子集合的空间并发需求必须实测。
当前设计的完成情况与边界见 [design.md](design.md)。

## 本项目的对应实现与差异

| Artifact 机制 | 本项目入口与适配 |
| --- | --- |
| 独立 P/D batch 与 split-prefill 状态 | [`worker.py`](../src/vllm_muxpd/worker.py) 使用共享权重和缓存的两个 runner；[`lanes.py`](../src/vllm_muxpd/lanes.py) 保留按层恢复的调用栈 |
| P 跨轮保留，D 每轮独立输出 | [`decoupled_scheduler.py`](../src/vllm_muxpd/decoupled_scheduler.py) 保存原始 P 成本与 cohort；[`decoupled_worker.py`](../src/vllm_muxpd/decoupled_worker.py) 分别采样 D 和当轮已完成 P |
| split-prefill 完成与合并 | 本项目使用显式 cohort 完成/取消 ACK，再调用原生 vLLM retirement；未照搬 SGLang 的 batch merge 状态结构 |
| GreenContext 与 SM 分区 | 尚未移植；`stream_mode=overlap` 仅为独立 NPU 流实验，不提供核数分区 |
| 独立 TP communicator | 默认 shared overlap 由 [`overlap.py`](../src/vllm_muxpd/overlap.py) 对现有 device process group 加 collective 事件门控；可选 isolated overlap 由 [`communicators.py`](../src/vllm_muxpd/communicators.py) 把 P 的 TP/EP 调用路由到 sibling groups，D 保留原生组；指定 TP4+EP eager 配置已验证 |
| 独立 workspace / decode graph | overlap 仍要求 eager；另一路 decoupled fenced 由 [`decode_graph.py`](../src/vllm_muxpd/decode_graph.py) 为 D 独立 capture/replay 原生 ACL graph，隔离静态输入、metadata 与 workspace；P 始终为 graph mode `NONE` |
| 按 token 工作量决定 P 层数 | `prefill_layer_token_budget` 默认 `0`；非零时按完整 cohort token 数计算基础层配额，保留 `layers_per_slice` 下限并限制到剩余层；输入仍是完整 prompt |
| 资源预测与争用修正 | 论文的硬件分区预测器与 contention guard 未接入；可选 `adaptive_layers` 将本机独立步骤 EWMA/deadline 配额与上述基础配额取较大值，未校准时仍可使用 token-work 配额 |
| 完整 P 的执行开销 | 可选 `compile_full_prefill` 允许 fresh、完整层配额 P 使用原生 FX 编译，P 仍为 graph mode `NONE`；部分 P 保持逐层 raw 执行，这不是 artifact GreenContext 或 P graph 的复刻 |

当前实现保留 artifact 的工作量配额含义，但没有照搬其参数或调度器。
`prefill_layer_token_budget` 使用实际接纳 cohort 的完整 token 总数，包含抢占
重算输入；它不会因部分取消或层数推进而缩小，也不是硬计算量上限。
`adaptive_layers` 的 deadline 可进一步提高配额；独立 P/D 延迟样本未齐备时，
仍使用基础配额。`prefill_first` 只在 decoupled adaptive fenced 下可用：
新 P 可在预览为完整 forward 时先执行，随后偿还被延后的已有 D 债务，避免
连续新 P 饿死 D。该顺序选择不等于论文的资源分区或 SLO 保证。

组件的 `decode_graph`、`compile_full_prefill`、`adaptive_layers` 和
`prefill_first` 默认均关闭；[图模式启动脚本](../scripts/serve-qwen35-decode-graph.sh)
显式启用它们并使用 `262144` 的 token-work 预算。独立编译 P 的阶段实验和
最终组合配置的性能结果不同，具体比较口径、六轮汇总及 TTFT/TPOT 取舍见
[TP4 性能报告](performance-tp4.md)。这些数据不支持把组合收益分别归因到
某一个开关，也不证明复现了论文的收益。

## HUST 原生图路径与生命周期适配

本轮固定的原生代码为 HUST vLLM
[`d0f22d2bda562156e4dbf433ce645e1769b4f804`](https://github.com/vLLM-HUST/vllm-hust/tree/d0f22d2bda562156e4dbf433ce645e1769b4f804)
和 vllm-ascend
[`03766ac696fde5ab1980d80ca0b8543d3580c989`](https://github.com/vLLM-HUST/vllm-ascend-hust/tree/03766ac696fde5ab1980d80ca0b8543d3580c989)。
关键入口是原生
[`NPUModelRunner`](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/worker/model_runner_v1.py)、
[`ACLGraphWrapper`](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/compilation/acl_graph.py)
和 [GDN metadata builder](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn_attn_builder.py)。
插件复用其 D metadata 更新和 capture/replay，不重新实现 attention kernel。

插件图模式要求 `VLLM_COMPILE` + `FULL_DECODE_ONLY`，仅支持 decoupled
fenced；它不支持 graph + overlap。D 有独立 wrapper、静态缓冲、workspace
和 capture/replay stream。P dispatcher 与 forward context 强制 graph mode
`NONE`，包括单 token 的新 P、profile 和 dummy run。可选完整 P 编译只改变
`skip_compiled`；启动预热先于 D capture，运行时只允许新 P 整网成功返回后
报告 `full_forward_return`，不能将挂起的 P 中途切为 compiled 执行。
具体 guard、完成条件和观测计数见 [设计文档](design.md#独立-decode-acl-graph)。

原生模块持有进程全局 `GraphParams`。插件关闭时释放自己的 D graph 并恢复
包装，由 owner 负责共享 KV 清理；它不清除原生全局 `GraphParams`。因此
同一进程内关闭后重新初始化 graph worker 会被拒绝，需要重启进程。
该限制不能被描述成已支持任意模型卸载、热更新或进程内多引擎复用。

vLLM 的调度发射即递增 computed token 数，与分层执行完成不是同一时刻。
因此 decoupled scheduler 显式保护 pending P，不以 computed token 数推断其
已经能够 decode。取消路径沿用原生 KV manager 的 `pop_blocks_for_free()`
引用保留语义，ACK 后才 `free_blocks()`；原生 host 请求可先删除，设备侧
仍持有的行和缓存必须晚于 ACK 回收。

这是对 HUST vLLM 生命周期的额外适配，不是 paper artifact 中已有的 vLLM
实现。部分取消需等整个 P forward 结束；整组取消同步 P stream 后 ACK；
pending P 时拒绝 keep / `PAUSED_ALL`，禁止原生暂停或 cache reset 提前回收
悬挂状态。具体协议及测试边界见 [design.md](design.md#跨迭代-prefill-协议)。
本节的实现映射本身不证明设备并发、性能优于 baseline 或达到 SLO。
shared TP2 compute 区间重叠有 [指定 capture 的独立证据](../evidence/2026-10-10/README.md#device-profile-independent-pd-compute-intervals)；
TP4 的 shared/isolated 通信、独立 D graph、完整 P FX 执行和优化组合另有
[本轮原始回执及源码清单](../evidence/2026-10-10-tp4-optimization/README.md)。
isolated8 已通过指定 TP4+EP eager 正确性与生命周期检查，profile 中事件数
减少，但计算交叠没有增加，性能也未超过原生；不能将它与 graph fenced 的
结果拼成未验证的 graph overlap 支持。上述证据范围均不扩展到每核资源隔离、
任意负载利用率或论文性能结果。

论文和 artifact 的逐项差异详见
[固定源码对照](../evidence/2026-10-10-tp4-optimization/original-comparison.md)。
原生 eager、原生 graph 与插件 graph 的单 token P 状态复用诊断见
[三路调查](../evidence/2026-10-10-tp4-optimization/one-token-prefill-investigation.md)：
原生 graph 在该探针中出现单 token P 的首输出不稳定；双 token P 的后续差异
在三路均存在。它是固定版本的兼容性线索，尚不足以确认缓存泄漏或将所有文本
差异归因为同一个问题。
