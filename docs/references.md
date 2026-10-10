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
