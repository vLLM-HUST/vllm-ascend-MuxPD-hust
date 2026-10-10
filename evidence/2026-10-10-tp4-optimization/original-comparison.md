# MuxWise 原始实现与 Ascend 插件：TP4 性能调查

记录日期：2026-10-10。本文为只读源码及既有结果分析；未启动 NPU、服务请求或性能测试。当前工作分支 `perf/tp4-execution-overhead` 上的未提交优化不在本次基线比较内。

## 可复现版本

| 对象 | 固定版本 | 来源 |
| --- | --- | --- |
| 正式论文 | arXiv 2504.14489 **v3**，2026-02-07，ASPLOS 2026 | [正式版本](https://arxiv.org/html/2504.14489v3)，[DOI](https://doi.org/10.1145/3779212.3790236) |
| 公开 artifact | `ykcombat/sglang`，`slo_config`，`eeac5148f456428a05823c330e1d40afb0ef2438` | [固定 commit](https://github.com/ykcombat/sglang/tree/eeac5148f456428a05823c330e1d40afb0ef2438)；本地 `/root/workspace/references/muxwise-sglang` |
| 原仓库 main | `3f3578ee7d1fc7a0290b99dc1440d70979f107cb` | 2026-10-10 `git ls-remote --heads origin` 核对；不是上述 artifact 分支 |
| predictor 实验分支 | `slo_scheduler`，`a068b3e484b09325d926c920643d59ca20084ca0` | [固定 commit](https://github.com/ykcombat/sglang/tree/a068b3e484b09325d926c920643d59ca20084ca0) |
| 插件比较基线 | `c9969f8ea6483b839a17765a8b841512be90a7b9` | [固定 commit](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/tree/c9969f8ea6483b839a17765a8b841512be90a7b9) |
| HUST vLLM | `d0f22d2bda562156e4dbf433ce645e1769b4f804` | `/root/workspace/vllm-hust` |
| HUST vLLM Ascend | `03766ac696fde5ab1980d80ca0b8543d3580c989` | `/root/workspace/vllm-ascend-hust` |

本机模型为 Qwen3.5-35B-A3B BF16、Ascend 910B2、TP4+EP、V1 eager。论文主要结果使用 NVIDIA 设备及不同模型/工作负载；不能直接转用其加速比。

## 已核实的实现差异

| 机制 | 原始 artifact | 插件 c9969f8 | 性能含义及边界 |
| --- | --- | --- | --- |
| Decode graph | [`cuda_graph_runner.py:481`](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/model_executor/cuda_graph_runner.py#L481) 按分区和 batch size 捕获；`:875` 整网 replay | [`worker.py:178`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/c9969f8ea6483b839a17765a8b841512be90a7b9/src/vllm_muxpd/worker.py#L178) 强制 eager | 图缺失会保留每算子的 host launch 开销，但 matched native 也 eager，因此不能单独解释当前插件相对 native 的退化。 |
| Host 编排 | [`multiplexing.py:121`](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/multiplex/multiplexing.py#L121) 先启动完整 D，再执行连续 N 层 P；[`llama.py:496`](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/models/llama.py#L496) 保存显式 layer cursor/hidden/residual | [`decoupled_worker.py:84`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/c9969f8ea6483b839a17765a8b841512be90a7b9/src/vllm_muxpd/decoupled_worker.py#L84) mixed overlap 逐层交替；[`lanes.py:85`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/c9969f8ea6483b839a17765a8b841512be90a7b9/src/vllm_muxpd/lanes.py#L85) 每次 step 切换上下文、stream、greenlet 并记录 event | 插件额外控制开销明确存在；其占总退化的比例需要 host/device 对照 profile，不能只由低设备活动率推断。 |
| 通信 | [`parallel_state.py:1453`](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/distributed/parallel_state.py#L1453) 新建 P TP communicator，并使用当前 stream | [`worker.py:277`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/c9969f8ea6483b839a17765a8b841512be90a7b9/src/vllm_muxpd/worker.py#L277) 共用 TP/EP group；[`overlap.py:75`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/c9969f8ea6483b839a17765a8b841512be90a7b9/src/vllm_muxpd/overlap.py#L75) 每次 collective 无条件 wait/record event | 纯 D 阶段也承担 gates 开销；共享 group 的正确顺序限制了跨阶段通信重叠。原始分支并不等于已经完整解决本机 MoE EP 通信。 |
| 资源分区 | [`pdmux_context.py:105`](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/multiplex/pdmux_context.py#L105) 预建 P-only、D-only 和多个 SM 分区；[`greenctx_stream.cu:56`](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/sgl-kernel/csrc/spatial/greenctx_stream.cu#L56) 使用 CUDA GreenContext | [`worker.py:259`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/c9969f8ea6483b839a17765a8b841512be90a7b9/src/vllm_muxpd/worker.py#L259) 创建普通 NPU streams，没有 Cube/Vector 配额 | 已观察到双流 compute 交叠，但不证明资源隔离或 GreenContext 等价行为。 |
| P 层配额 | [`multiplexing.py:137`](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/multiplex/multiplexing.py#L137) 使用 `max(1, split_forward_token_budget // extend_num_tokens)`，默认预算 65536 | [`decoupled_scheduler.py:353`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/c9969f8ea6483b839a17765a8b841512be90a7b9/src/vllm_muxpd/decoupled_scheduler.py#L353) 固定配额或 TTFT slack/EWMA | 固定 8 层使 40 层 P 在持续有 D 时至少跨 5 轮；原始公式对小 P 常一次完成。固定配额可能增加短请求 TTFT。 |
| Batch composition | [`multiplexing.py:55`](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/multiplex/multiplexing.py#L55) 保持独立 running D 与 split P batch | [`decoupled_scheduler.py:184`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/c9969f8ea6483b839a17765a8b841512be90a7b9/src/vllm_muxpd/decoupled_scheduler.py#L184) 一个 pending P cohort 期间只调度 D，保留 P slots/KV | 两者均不把 P+D 输入拼成一次模型 batch；插件 cohort 未完成时新 P 排队。该开销不能与原生混合 batch 的一次权重访问/算子启动直接等同。 |
| P 完成同步 | [`multiplexing.py:174`](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/multiplex/multiplexing.py#L174) query P event，全部 TP ranks 确认后再合并 | [`decoupled_worker.py:224`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/c9969f8ea6483b839a17765a8b841512be90a7b9/src/vllm_muxpd/decoupled_worker.py#L224) P Python forward 返回后同轮同步采样 | 最后一轮 D ACK 可能等待 P 尾部；这是可改善协议点，尚未单独量化收益。 |
| Prefix reuse | 原版保留共享 KV/tree cache；`multiplexing.py:106` 可见 tree cache 管理 | [`config.py:114`](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/c9969f8ea6483b839a17765a8b841512be90a7b9/src/vllm_muxpd/config.py#L114) 禁用 prefix caching，Mamba cache 为 none | 插件保留阶段间 KV，尚不能利用跨请求 prefix reuse；目前 matched native 也关闭该功能，不能用它解释本次 matched 差值。 |

论文 §3.2–3.4 采用 decode SLO 优先、时长匹配的 P 层配额 `ceil(Td * num_layers / Tp)`、竞争估计及可选 P 抢占。当前插件的 TTFT slack 策略不等价。公开 artifact 也没有完整复现全部论文策略：`multiplexing.py:24` 明示 temporary demo；`slo_scheduler` 的 [`predictor.py:141`](https://github.com/ykcombat/sglang/blob/a068b3e484b09325d926c920643d59ca20084ca0/python/sglang/srt/multiplex/predictor.py#L141) 仅有 NVIDIA 型号参数，该分支 multiplex 目录内没有 scheduling 调用 Predictor 的证据。

## 既有 TP4 证据与归因限制

- `local-results/tp4/native.json`、`overlap.json`、`adaptive-fenced.json`：各 2 次、每次 8 请求，prompt 128/2048/512/4096，输出 64，间隔 0.12 秒。按总 output tokens / 总 wall time 聚合，约为 native 32.10、fixed overlap 21.18、adaptive fenced 32.29 token/s。小样本和 run 间差异不支持稳定收益结论。
- overlap 两轮分别为 24.85、18.45 token/s；native 为 31.63、32.58；adaptive fenced 为 32.15、32.44。需要新的多重复对照，不能把 overlap 的变化全部算作代码固定成本。
- 离线读取 `overlap-trace-403973.jsonl` 与 `adaptive-fenced-trace-423474.jsonl`，按相同步号的 `decoupled_schedule`/`decoupled_complete` 时间差分类：纯 D、batch=8 为 106 轮/中位 288.37 ms 与 122 轮/225.63 ms；纯 D、batch=1 为 494 轮/243.97 ms 与 455 轮/222.24 ms。该窗口混有 smoke、正确性、benchmark、profile，**不是 matched microbenchmark**，仅用于证明调查不能只关注 mixed P/D 轮。
- TP4 profile 报告约 12.7% compute active、各 rank P/D compute overlap 2.86–4.49%。这是有限捕获窗口的设备区间统计；不能解释成生产利用率、单核占用或全部空隙为 host 开销。
- `cpu-snapshot.json` 的 cgroup 配额相当于 16 CPU 核；累计 throttling 计数不是本次请求发生 CPU throttling 的因果证据。
- 0.1 秒 TPOT SLO 下三种模式的既有测试均未达到 goodput；论文 goodput 倍数不能用来解释这里的裸吞吐。

离线复核纯阶段时间的核心规则：`kind='D'` 当 schedule 的 `pending_prefill_ids` 为空，按 `len(decode_ids)` 分桶；每轮毫秒数为 `(complete.time - schedule.time) * 1000`。勿将不同时段相减或跨 rank 相加。

## 可实施优先级

1. **P0：减少冗余 collective gates。** 同 group、同 stream 连续调用可省 wait；保留 collective 后精确 tail。直接延迟记录 tail 到下一次跨流调用会把后续 compute 也串进依赖，不能作为等价优化。纯 D 快路径需要保留模式转换时的正确依赖。
2. **P0：减少固定小配额对短 P 的延迟。** 使用本机实测 P/D 成本及 shape 决定配额，分别记录时长匹配和 TTFT 策略；不可把 NVIDIA 常数当成本机模型参数。
3. **P1：独立 P/D HCCL groups 与 D 整网 graph。** 先验证组切换和 MoE EP，再开放并发 graph；decode graph 首版可限制为 fenced，见相邻设计文档。
4. **P1：P 完成采用 query/延迟采样 ACK。** 增加“全部层已提交、设备尚未完成”状态，保持 KV pin；需要同步更新 receipt、取消、释放协议。
5. **P2：资源配额及 contention predictor。** [`set_stream_limit` 官方文档](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-set_stream_limit.md) 仅承诺支持 Ascend C 算子的核数限制，要求 `allow_internal_format=False`，不能直接视作 SM 独占分区。需先逐算子核实有效性和干扰，再谈 SLO 保证。

以上为实现差异和待验证因果假设，不是已完成优化或已测得新性能收益。
