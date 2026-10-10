# TP4 性能调查与优化验证

本报告记录 2026-10-10 在四张 Ascend 910B2、Qwen3.5-35B-A3B 上的 TP4 调查。
当前 work256 配置的全部六轮实测，相比原生 graph 的两阶段六轮汇总，混合
负载吞吐高 **3.79%**，decode-heavy 高 **1.27%**，TTFT P95 都降低。
它使用完整 P、独立 P/D model batch、decode graph、预算 262144 的层配额及
可选完整 P 编译，同时优化连续 D 路径并关闭 trace/profiler。收益属于该联合
配置，不能归因于某一个开关，也不能直接推广到其他模型、设备或负载。

| 负载 / 全部 6 轮 | 原生 tokens/s | work256 tokens/s | 原生 → work256 TTFT P95 / s | 原生 → work256 请求平均 TPOT P95 / s |
| --- | ---: | ---: | --- | --- |
| 混合 | 175.925 | 182.600 | 0.756 → 0.669 | 0.03994 → 0.03849 |
| Decode-heavy | 241.934 | 245.011 | 0.783 → 0.680 | 0.02719 → 0.02769 |

**并非所有指标都改善。** Decode-heavy 的请求平均 TPOT P95 比六轮原生高
1.82%；候选第二组三轮吞吐 244.152 tokens/s，也略低于原生 initial 的
244.227。小幅收益没有统计显著性证明，所有阶段及两次原生基线都保留，
没有择取较弱基线。公式、逐轮值与输入散列见
[完整汇总](../evidence/2026-10-10-tp4-optimization/graph-work256-summary.json)。

自然语言 16/16 文本精确一致，合成请求传输通过但文本仅 40/56 跨端点一致；
不能作任意 prompt 的等价承诺。同源码 trace-on 诊断已确认 12K P 按 **21+19 层**
完成、取消在 21 层后收到 ACK，并验证 4 次 P-first 后下一轮 D 服务偿还，见
[原始行摘录与校验](../evidence/2026-10-10-tp4-optimization/graph-work256-traced-proof.json)。
这些是独立生命周期诊断，不计入上面的吞吐汇总。
组件默认 `compile_full_prefill=false`、`prefill_layer_token_budget=0` 保持不变。
以下保留旧配置、优化机制和各阶段失败或未改善的结果，避免把当前收益
误归于某一个步骤或把 CPU mock 测试当成硬件证明。

## 环境与比较口径

模型为 BF16、40 层混合 full attention/GDN 的 Qwen3.5-35B-A3B，使用四张
Ascend 910B2、TP=4、EP 开启、V1 同步调度、language-model-only。
以下 eager 对照使用相同的 max model length/token budget 16384、max sequences 8、
GPU memory utilization 0.85、APC 关闭、`mamba_cache_mode=none`。
插件关闭 token chunking，模型 forward 始终使用独立 P/D batch。
原生保留自身调度和 mixed batch 行为。

- HUST vLLM：`d0f22d2bda562156e4dbf433ce645e1769b4f804`。
- HUST vllm-ascend：`03766ac696fde5ab1980d80ca0b8543d3580c989`。
- 旧插件比较基线：`c9969f8ea6483b839a17765a8b841512be90a7b9`。
- 新一轮候选为 `perf/tp4-execution-overhead` 的运行时快照；具体文件 SHA256
  见 [native manifest](../evidence/2026-10-10-tp4-optimization/native-manifest.json) 和
  [P-first manifest](../evidence/2026-10-10-tp4-optimization/pfirst-manifest.json)。
  后续源码改动不能自动归入这轮测量。
- 模型配置、包版本及旧 TP4 环境见
  [environment.json](../evidence/2026-10-10-tp4/environment.json)。

原生 manifest 中有被后续参数覆盖的 TP2/APC/2048-token 参数；有效配置应按
最后一次参数值及实际服务配置核对，不能只截取命令行前半段。
最初 TP2、APC 开启、Mamba align 的常驻服务不属于这里的匹配对照。

每轮使用 8 个固定合成 token-ID 请求，每请求输出 64 tokens；温度为 0、
seed 为 42、`ignore_eos=True`。每种输入长度先以 8 个输出 tokens 预热。
混合负载循环使用 prompt 长度 `128,2048,512,4096`，请求间隔 0.12 s；
decode-heavy 负载全部为 128-token prompt，同时发起。

度量来自 [benchmark_serving.py](../scripts/benchmark_serving.py)：

- 聚合吞吐 = 所有重复的完成 tokens 总数 / 所有重复的 wall time 总和，
  包括请求到达间隔、prefill 和尾部完成时间；不是稳态 decode 吞吐。
- TTFT = 客户端收到首个非空文本 SSE 的时间。它可能晚于首个生成 token，
  不代表设备首 token 时间。
- 每请求平均 TPOT = `(最后非空文本事件时间 - 首个非空文本事件时间) /
  (completion_tokens - 1)`；下表的 P95 是这些请求平均值的 P95，
  不是逐 token 延迟的 P95。
- 最大 SSE 内容事件间隔也按请求聚合。文本事件数可能少于生成 tokens，
  不能把内容事件间隔等同于逐 token TBT。
- 分位数采用 nearest rank：排序后取第 `ceil(n × 0.95)` 项。
  新一轮每配置 24 个请求的 P95 是第 23 项；旧两轮 16 请求的 P95 是最大值。
- benchmark 保存结果前要求 `[DONE]`、usage、非空文本和预期完成 token 数。
  该版本保留内容事件时间，不保存原始 HTTP/SSE body、实际 dispatch 时刻或
  finish reason；这些字段不能从旧报告中补推。

计时和 profiler 采集分开。三个重复不能确立统计显著性；执行顺序没有随机化，
自适应估计还会跨重复保留，因此不能将重复看成独立冷启动样本。

## 旧 TP4 结果：定位问题的起点

旧对照每种模式两轮、每轮 8 请求，均为 eager。完整输入散列、公式、逐轮
结果和输出稳定性见
[benchmark-summary.json](../evidence/2026-10-10-tp4/benchmark-summary.json)。

| 模式 | 聚合 tokens/s | 两轮 tokens/s | TTFT P95 / s | 请求平均 TPOT P95 / s |
| --- | ---: | --- | ---: | ---: |
| 原生 | 32.102 | 31.633 / 32.584 | 0.710 | 0.240 |
| fixed overlap，8 层 | 21.175 | 24.852 / 18.446 | 3.836 | 0.374 |
| adaptive fenced，8 层下限 | 32.291 | 32.148 / 32.435 | 1.182 | 0.240 |

adaptive fenced 的聚合吞吐只比原生高约 0.6%，而 TTFT 更高；这不能作为
稳定加速的结论。fixed overlap 的两轮差异很大，也不能把全部差值归为固定
代码成本。合成请求存在跨重复文本 hash 变化，不能只归因于 BF16，也不能
据此承诺任意 prompt 的输出等价。

这组 8 个 prompt 共 13568 tokens，小于 16384 的 token budget。原生可以
将它们放入预算，不需要靠 prompt token chunk 才能容纳这组输入。
因此，本例不能用“原生 chunk 更小”单独解释差值。

## 已核实的性能机制

**固定配额会延长 P cohort 的占用。** 持续有 D 时，40 层模型配额为 8 层，
一个 P cohort 需要五轮才能完成；当前协议在它完成前不接收新 P cohort。
源码与调度 trace 支持这条等待链。它能解释短 P 为什么也会排队，但没有
单独消融来量化它占 TTFT 差值的比例。

**完整 P 和 D 独立 forward 有结构成本。** 原生可在一次模型 batch 内同时
处理 P/D；插件为了保留阶段隔离，需要分别执行 forward。即使 adaptive
把当前 P 配额提高至全部剩余层，原先 D-first 的一轮仍先运行 D，再完成 P，
之后统一退休结果。提高层配额不能消除这次完整 D 的等待。

**shared overlap 存在额外同步操作，纯 D 也受影响。** 四个 rank 的设备/host
profile 均观察到更多 event record/wait；按 forward-equivalent 归一后仍成立。
这是已测得的操作数量差异，不能直接换算成同等比例的吞吐损失。

**eager 与原始 artifact 的 decode graph 路径有实质区别。** 缺少 graph 会保留
算子级 host launch 和 Python 调用，但旧的匹配原生也为 eager，所以
“插件没有 graph”本身不足以解释它相对 eager 原生的额外退化。
后续 graph 候选必须与原生 graph 对照。

旧 trace 中纯 D 的 schedule→receipt 时间也更长，说明调查不能只关注
P/D 同时存在的轮次。但该 trace 混有 smoke、正确性、benchmark 和 profile，
不构成匹配 microbenchmark。cgroup 累计 throttling 计数也不能证明本次
请求的退化来自 CPU throttling。

## Profile：计数、耗时与设备区间分开

[逐 rank 审计](../evidence/2026-10-10-tp4-optimization/native-vs-shared-profile.json)、
[审计说明](../evidence/2026-10-10-tp4-optimization/native-vs-shared-profile.md) 和
[复现脚本](../evidence/2026-10-10-tp4-optimization/native-vs-shared-profile-audit.py)
保留了 trace/profiler metadata SHA256。

两个 profile 使用一个 128-token prompt 的 D 请求，在其首次文本输出后
加入 12000-token P；捕获限于八个引擎迭代，不覆盖完整请求生命周期。
原生导出了 8 个 forward-equivalent，旧 shared overlap 导出了 9 个，
因此原始累计值包含不同工作量。

没有直接导出整网或 decoder-layer forward 的 Python 范围。
forward-equivalent 是由每次整网 40 个 MoE、30 个 GDN 调用，以及 logits
all-gather 次数一致推得；MoE 范围不能冒充整层耗时。

| 每 forward-equivalent 的 host 操作 | 原生 | 旧 shared overlap | 说明 |
| --- | ---: | ---: | --- |
| `vllm::all_reduce` | 81.000 | 81.000 | 调用数量相同 |
| `Event::record` | 122.000 | 215.667 | 增加 76.8% |
| `Event::wait` | 121.000 | 205.444 | 增加 69.8% |
| `aclrtRecordEvent` | 368.750 | 462.333 | CANN API 计数 |
| `aclrtStreamWaitEvent` | 285.000 | 369.444 | CANN API 计数 |

四个 rank 的上述计数一致。Python wrapper、CANN API、设备 record/wait task
可能是同一次同步的不同层级，不能把三者相加。

rank 0 的 `all_reduce` CPU inclusive 累计为原生 108.937 ms、shared 309.207 ms；
减去同线程内严格包含的 CPU 子范围后，分别为 40.487 ms、137.241 ms。
这种 exclusive 估计保留未埋点工作和可能的协程暂停时间，不等于 CPU 指令执行
时间，也不等于设备通信耗时。MoE、GDN、all-reduce 等嵌套 inclusive 值不能相加。

| 设备计算区间统计，四个 rank 的范围 | 原生 | 旧 shared overlap |
| --- | ---: | ---: |
| compute kernel 区间并集 / ms | 371.430–373.606 | 369.224–372.634 |
| 首个 compute 开始至最后一个结束 / ms | 2680.627–2681.479 | 2915.699–2917.378 |
| compute 并集 / 上述窗口 | 13.85%–13.94% | 12.66%–12.78% |

计算区间只包括硬件轨道上显式 AI_CORE/AI_VECTOR/MIX 类型且时长为正的任务，
排除可识别的通信任务。不同 rank/capture 的时间轴分别处理，没有跨设备合并。
比例是带 profiler 开销的窗口覆盖率，不是 AICore 占用率；空白还可能包含
通信和等待，不能全部归为 host launch。

这组导出没有可明确归为 HCCL kernel 的硬件名称/类型，不意味着没有通信。
SDMA、Notify Wait、Write Value 及 Communication track 范围均单独保留。
后者可能包含等待，不能当作有效通信计算时间加入 compute 并集。
旧 shared profile 的 physical stream 45/44 映射和交集证明了 P/D compute
区间交叠，仍不证明资源隔离、核占用分区或端到端收益。

## 本轮 eager P-first 候选

候选使用 `execution=decoupled`、`stream_mode=fenced`、
`layers_per_slice=8`、`adaptive_layers=true`、`prefill_first=true`。
结果来自 [eager-summary.json](../evidence/2026-10-10-tp4-optimization/eager-summary.json)。
下面是该运行快照的整体结果，不是 `prefill_first` 单独开关的消融实验。

| 负载 / 模式 | 聚合 tokens/s | 三轮 tokens/s | TTFT P95 / s | 请求平均 TPOT P95 / s | 最大内容事件间隔 P95 / s |
| --- | ---: | --- | ---: | ---: | ---: |
| 混合 / 原生 eager | 29.910 | 28.360 / 30.477 / 31.030 | 0.774 | 0.267 | 0.476 |
| 混合 / P-first eager | 32.420 | 32.341 / 32.483 / 32.437 | 0.974 | 0.242 | 0.681 |
| Decode-heavy / 原生 eager | 32.590 | 32.577 / 32.640 / 32.553 | 0.737 | 0.240 | 0.364 |
| Decode-heavy / P-first eager | 33.698 | 33.928 / 33.941 / 33.236 | 0.697 | 0.234 | 0.574 |

混合负载本次聚合吞吐高 8.4%，TTFT P95 却高 25.8%；decode-heavy 吞吐高
3.4%，TTFT P95 低 5.4%。两种负载的最大内容事件间隔 P95 都变差。
这些是本次三轮的描述性结果，没有统计显著性结论。
不能把本次原生 29.910 tokens/s 与旧原生 32.102 tokens/s 混用来计算加速比。

本轮四组各 24 请求均在 1 s 内收到首个非空文本 SSE；这只是实测范围，
不是硬上界。benchmark 原配置的联合 SLO 是 TTFT ≤ 2 s、请求平均 TPOT ≤ 0.1 s；
四组均为 0/24 满足，联合 goodput 均为 0。不能以裸吞吐增加宣称这条 SLO
已被满足，也不能将论文 goodput 倍数套用于本机结果。

P-first 的实现让即将一次完成的 P 在原生 admission **之前**选择 P-only；
已有 D 暂时不进入 allowlist，因此不会提前增加其 `num_computed_tokens`。
P 结果退休后形成 decode debt，下一次可服务的 transaction 优先 D。
若 P 分配 KV 失败，回退到 D，不建立债务。

源码审计还修复了 full-budget P-only 路径未记债的问题：连续长 P 请求如果
只生成一个 token，原本可能不断释放 slot 而绕过已有 D。现在该路径明确记录
`full_budget_prefill_first`，完成后也必须偿还 D；deadline 路径单独记录
`deadline_prefill_first`。取消、抢占、完成会清理对应债务；严格 worker receipt
验证先于退休和债务提交。原生 KV 不足时仍可能抢占刚完成 P 的请求。

真实 native scheduler/KV CPU fixture 的相关回归曾完成 **134 passed**，
覆盖 D token 计数、下一轮服务、KV 失败回退、取消和 full-budget 饥饿路径。
这提供状态机证据，不代表 NPU kernel 正确性，也不保证 wall-clock TPOT：
一次完整 P 仍可能让 D 等待超过 100 ms。
详见 [调度审计](../evidence/2026-10-10-tp4-optimization/scheduler-findings.md)。

## Graph8 阶段性实测：仍未追平原生 graph

本节对应 [原生 graph manifest](../evidence/2026-10-10-tp4-optimization/native-graph-manifest.json)
和 [graph8 manifest](../evidence/2026-10-10-tp4-optimization/graph8-manifest.json) 的快照。
两者使用 `FULL_DECODE_ONLY`、compilation mode 3、capture sizes `[1,2,4,8]`；
graph8 为 fenced、固定 8 层配额、`adaptive_layers=false`、`decode_graph=true`，
P 始终 eager。**它不是上节的 adaptive P-first eager 配置。**
负载、预热、输出 token 数、三次重复及聚合方法与上节相同。

| 负载 / 模式 | 聚合 tokens/s | 三轮 tokens/s | TTFT P95 / s | 请求平均 TPOT P95 / s | 最大内容事件间隔 P95 / s |
| --- | ---: | --- | ---: | ---: | ---: |
| 混合 / 原生 graph（initial） | 177.033 | 175.794 / 173.668 / 181.838 | 0.756 | 0.0395 | 0.405 |
| 混合 / graph8 | 168.801 | 168.919 / 168.986 / 168.499 | 0.813 | 0.0375 | 0.119 |
| Decode-heavy / 原生 graph（initial） | 244.227 | 244.028 / 244.443 / 244.209 | 0.734 | 0.0272 | 0.360 |
| Decode-heavy / graph8 | 229.832 | 225.650 / 228.778 / 235.276 | 0.809 | 0.0279 | 0.0926 |

本次 graph8 聚合吞吐分别低于匹配原生 graph **4.65% / 5.89%**，TTFT P95
也更高，仍有待优化。最大内容事件间隔的改善不能自动等同于逐 token TBT
改善。四组各 24 请求均满足该 benchmark 的 TTFT ≤ 2 s、请求平均 TPOT ≤ 0.1 s，
但这条请求平均 SLO 仍不能保证每个 token 的延迟 ≤ 0.1 s。
Graph 的两侧吞吐都远高于各自 eager 数据，说明 graph 路径对本机成本很关键；
不同候选配置和运行时段之间的倍数不能作为插件相对原生的加速比。

[归档的启动快照](../evidence/2026-10-10-tp4-optimization/graph8-profile-proof.json)
记录四个 rank 各捕获 4 个 graph entries，启动时 `capture_successes=1`、
`capture_failures=0`。启动快照的 replay 计数为 0，
因为当时尚未服务请求；不能把它解读为运行期没有 replay。
[四 rank profiler 证据](../evidence/2026-10-10-tp4-optimization/graph8-profile-proof.json) 直接
提取到每个 rank **7 次** `AscendCL@aclmdlRIExecuteAsync` 完整记录，来源为
CANN 设备运行时 API，而非客户端标记或 worker 自报计数。导出中的该轨道
标为 `CANN / CPU`：它记录 graph execute 提交，不是设备 kernel 时长。
该独立证据不依赖 worker shutdown 日志；API 强制终止 worker 时可能不会
执行插件的 shutdown hook。

[提取脚本](../evidence/2026-10-10-tp4-optimization/graph8-profile-proof-audit.py) 和 JSON 保留
每份 trace/profiler metadata 的 SHA256、全部命中事件、启动捕获日志及
profile workload：P128/D64 后加入完整 P12000/输出8，最多捕获八个引擎迭代。
每 rank 的 7 次记录只覆盖这个窗口，不是服务器生命周期总数；四个 TP rank
不能相加解释为 28 个独立 decode step。这些 API 记录也不单独标识请求、D batch
size 或输出正确性，因此仍需结合上述运行配置、SSE 和调度回执。

正确性证据必须分开解释：

- [自然语言对照](../evidence/2026-10-10-tp4-optimization/native-graph-vs-graph8.json) 的
  16/16 个对比文本精确一致，两侧既定 oracle 均通过，重复和 isolated/paired
  文本也稳定。这仅覆盖这些固定测试输入。
- [合成 SSE 对照](../evidence/2026-10-10-tp4-optimization/native-graph-vs-graph8-serving.json)
  两侧各 56 请求均通过严格传输、usage、finish reason、输出长度校验；
  跨端点文本只有 **37/56** 精确一致。28 个 case 分组各有两个重复，原生
  8/28 分组不稳定，插件 7/28 不稳定。单 token P 在插件侧稳定、原生侧不稳定，
  原因尚未确定；不能以此宣称修复了原生问题，或将差异全部归因于 BF16。
- [长 P 进度回执](../evidence/2026-10-10-tp4-optimization/graph8-progress.json) 记录完整
  12000-token P 首次文本之前出现 4 个 D 内容事件；
  [取消回执](../evidence/2026-10-10-tp4-optimization/graph8-cancel.json) 通过。
  [严格 trace checker](../evidence/2026-10-10-tp4-optimization/graph8-trace-check.json)
  验证 1153 轮层进度回执、40 层总数，并记录 1 次 P 取消退休。
  这些支持跨迭代进度和生命周期，不证明 fenced 下设备 kernel 重叠。

后续 hot-path 改动必须形成新的 manifest 和匹配回执。本节只记录上述已落盘
结果，未把继续优化、全量输出等价或最终性能目标标记为完成。

## Graph-adaptive 后续快照与状态复用探针

[graph-adaptive manifest](../evidence/2026-10-10-tp4-optimization/graph-adaptive-manifest.json)
记录了冻结的运行源码散列。该候选沿用 decode-only graph、fenced 及 8 层下限，
开启 adaptive layers 和 P-first，关闭 scheduler trace，并跳过同 stream 的冗余
wait。它同时包含多个改动，不能从下面的整体结果拆出每一项的收益。
原生 graph 对照复用上一节已完成的三轮，没有在这两种候选之间随机交替测量。

| 负载 / graph-adaptive | 聚合 tokens/s | 三轮 tokens/s | TTFT P95 / s | 请求平均 TPOT P95 / s | 最大内容事件间隔 P95 / s |
| --- | ---: | --- | ---: | ---: | ---: |
| 混合 | 171.400 | 168.262 / 171.777 / 174.268 | 0.833 | 0.0369 | 0.377 |
| Decode-heavy | 236.116 | 238.814 / 238.694 / 231.010 | 0.815 | 0.0267 | 0.0955 |

数据见 [mixed](../evidence/2026-10-10-tp4-optimization/graph-adaptive-mixed.json)
和 [decode-heavy](../evidence/2026-10-10-tp4-optimization/graph-adaptive-decode-heavy.json)。
两种负载各 24 请求均满足 benchmark 的 TTFT ≤ 2 s、请求平均 TPOT ≤ 0.1 s。
聚合吞吐较 graph8 提高，但仍分别低于复用的原生 graph 对照约 **3.2% / 3.3%**，
TTFT P95 也仍高于原生。这是另一个阶段性快照，不能写成已经追平或超越。

[自然语言对照](../evidence/2026-10-10-tp4-optimization/native-graph-vs-graph-adaptive.json)
的 16/16 文本与原生 graph 精确一致，两侧 oracle 通过。
进一步的 [状态复用探针](../evidence/2026-10-10-tp4-optimization/graph-adaptive-state-reuse.json)
完成全部 30 请求及传输检查，记录实际生成 token IDs。六个 case 每个在背景
4096-token P 请求前后各重复两次，背景请求输出 8 tokens；它检验前后输出，
没有单凭客户端请求顺序证明某个物理 KV slot 必然复用。

| 固定 prompt IDs / 输出上限 | 背景请求前 | 背景请求后 | 前后首次 token 序列 |
| --- | --- | --- | --- |
| `[100]` / 1 | 稳定 | 稳定 | 相同 |
| `[100]` / 16 | 稳定 | 稳定 | 相同 |
| Hello `[9419]` / 1 | 稳定 | 稳定 | 相同 |
| Hello `[9419]` / 16 | 稳定 | 稳定 | 相同 |
| Hello/Hi `[9419,12675]` / 1 | 稳定 | 稳定 | 相同 |
| Hello/Hi `[9419,12675]` / 16 | **不稳定** | 稳定 | **不同** |

最后一项在背景请求前的两次输出于 token index 1 首次不同（索引从 0 起）；
前后首次输出于 index 2 不同。不能用其余稳定 case 掩盖这项差异，不能直接
归因于 BF16。后续补齐的三路对照见下节，同类后段差异也出现在原生 eager。

旧 `graph8-state-reuse.json` 在 54 个预期请求中只留下 41 个回执，原因是操作时
提前停止 API。它明确标为 incomplete，并已从性能/正确性结论及完整回执归档中
排除；这不是一次可用于评估模式优劣的完整试验。仅在
[archive-index.json](../evidence/2026-10-10-tp4-optimization/archive-index.json)
保留排除原因和计数。

## 单 token P：完整三路诊断与尚未确定的原因

[独立调查](../evidence/2026-10-10-tp4-optimization/one-token-prefill-investigation.md)
现已核对原生 eager、原生 graph、插件 graph-adaptive 三路同 payload 的完整
token-ID 回执，各 **30/30** 请求传输、usage、finish reason 和 token 数均通过。
两份新增原生证据为
[eager](../evidence/2026-10-10-tp4-optimization/native-eager-state-reuse.json) 与
[graph](../evidence/2026-10-10-tp4-optimization/native-graph-state-reuse.json)；
eager 启动参数见
[manifest](../evidence/2026-10-10-tp4-optimization/native-eager-state-manifest.json)。
每路四个单 token P 分组共 16 次目标响应，另有双 token 对照及六次背景请求。

| Prompt / 输出上限 | 原生 eager | 原生 graph | 插件 graph-adaptive |
| --- | --- | --- | --- |
| `[100]` / 1 或 16 | 前后重复均稳定 | 重复及前后比较均从生成 index 0 不同 | 前后重复均稳定，与 eager 完整序列一致 |
| Hello `[9419]` / 1 或 16 | 前后重复均稳定 | 重复及前后比较均从生成 index 0 不同 | 前后重复均稳定，与 eager 完整序列一致 |
| HelloHi `[9419,12675]` / 1 | 稳定，首 token 为 0 | 稳定，首 token 为 0 | 稳定，首 token 为 0 |
| HelloHi `[9419,12675]` / 16 | 四次输出中 3 条独特序列 | 四次输出中 3 条独特序列 | 四次输出中 3 条独特序列 |

原生 eager 与插件的全部 16 条单 token 目标响应完整序列逐条相同；原生 graph
在这四个分组的重复及前后比较都从首个生成 token 分歧。自然 `Hello` 的单
token ID 也复现，因此不能仅解释为 `[100]` 的文本显示问题。这支持较窄的
判断：本次单 token 首步不稳定与原生 graph 路径相关。
双 token/max16 的后段差异则在三路都出现，原生 eager 自身也复现；不能把它
归为 graph 专有问题，或因单 token 结果较好而宣称插件全面修复正确性。

源码调查指出候选机制：原生 dispatcher 按 query shape 接受单 token 新 P
为 uniform decode，而 GDN metadata 仍按实际 phase 归为 P；decode graph 的
静态 state 索引更新又受 `num_prefills == 0` 限制。插件强制 P 使用 NONE 路径，
可以规避这一候选路由。当前没有逐请求 cache 索引、内容或 logits 记录，也未
进行仅修改 phase eligibility 的隔离实验，因此**尚不能确诊旧 KV/state 被错误
读取**。每组只有两次 before/after 重复，结论不扩展为所有输入的确定性保证。
[上游 Issue 草稿](../evidence/2026-10-10-tp4-optimization/one-token-prefill-issue-draft.md)
保留最小 payload、固定源码链接、散列和剩余验证项。

相关上游 [Issue #40](https://github.com/vLLM-HUST/vllm-ascend-hust/issues/40)
记录 CANN 9.1 下 BF16/D256 paged FIA 在 eager 和 graph 重复相同输入时出现
较大数值错误，其测量形状为 B16、每请求 Q3、Q heads 8。该形状与本轮不同，
尚未建立它与这里双 token 后段差异的因果联系，也不能将单 token 的 graph
阶段分类疑点直接视为同一根因。

## 第二次原生 graph 基线与六轮汇总

在前述 graph 候选和 isolated8 之后，重新启动原生 graph 并以相同有效参数
再测三轮。[bracket manifest](../evidence/2026-10-10-tp4-optimization/native-graph-bracket-manifest.json)
标识 `native-tp4-full-decode-only-bracket`，记录时间为 04:39 UTC；首轮
`native-tp4-full-decode-only` 记录于 04:06 UTC。两次的有效模型、TP4/EP、
编译模式、capture sizes、APC/Mamba 配置及 benchmark 参数一致。
原始回执是 [bracket mixed](../evidence/2026-10-10-tp4-optimization/native-graph-bracket-mixed.json)
和 [bracket decode-heavy](../evidence/2026-10-10-tp4-optimization/native-graph-bracket-decode-heavy.json)。

| 负载 / 原生 graph 阶段 | 聚合 tokens/s | 各轮 tokens/s | TTFT P95 / s | 请求平均 TPOT P95 / s | 最大内容事件间隔 P95 / s |
| --- | ---: | --- | ---: | ---: | ---: |
| 混合 / initial，3 轮 | 177.033 | 175.794 / 173.668 / 181.838 | 0.756 | 0.0395 | 0.405 |
| 混合 / bracket，3 轮 | 174.831 | 173.453 / 177.494 / 173.605 | 0.733 | 0.0403 | 0.399 |
| 混合 / 全部 6 轮 | 175.925 | 保留上面两阶段全部轮次 | 0.756 | 0.0399 | 0.405 |
| Decode-heavy / initial，3 轮 | 244.227 | 244.028 / 244.443 / 244.209 | 0.734 | 0.0272 | 0.360 |
| Decode-heavy / bracket，3 轮 | 239.684 | 237.639 / 235.006 / 246.719 | 0.784 | 0.0279 | 0.385 |
| Decode-heavy / 全部 6 轮 | 241.934 | 保留上面两阶段全部轮次 | 0.783 | 0.0272 | 0.377 |

[baseline summary](../evidence/2026-10-10-tp4-optimization/native-graph-baseline-summary.json)
保存输入 SHA256、阶段、每轮数值、请求数、token 总数、wall time 总和及公式。
六轮聚合吞吐使用每负载 3072 个完成 tokens 除以六轮总 wall time；P95 使用全部
48 个请求排序后的第 46 项，不平均两个阶段的 P95。两阶段各 24/24 请求满足
既定联合 SLO，合计 48/48；混合负载合并 request goodput 为 2.749 请求/s，
decode-heavy 为 3.780 请求/s。摘要也保留合成文本 hash 的跨轮变化。

bracket 相对 initial 的吞吐略低，两阶段的运行内波动仍可见。此前 graph8 和
graph-adaptive 段落中的百分比明确使用 initial，不能事后改为择取较弱的
bracket 来放大收益；后续候选应同时面对两阶段和全部六轮汇总。这些顺序执行
的重复仍不是随机化实验，不能据此建立统计显著性。

[原生 graph 的长 P probe](../evidence/2026-10-10-tp4-optimization/native-graph-progress.json)
完成 D 输入128/输出64及 P 输入12000/输出8。在客户端 P 开始到首个 P 文本
事件之间，记录 **1 个 D 内容事件**；graph8 的对应探针为 4 个。
这展示两种执行方式在各自窗口中的进度差异，不代表相同窗口长度、实际设备
D batch 或 kernel 重叠，也不能单凭事件数量判定吞吐、TTFT 或 TPOT 更好。

## Graph-compiled：本次未获得性能提升

[graph-compiled manifest](../evidence/2026-10-10-tp4-optimization/graph-compiled-manifest.json)
记录 04:45 UTC 的冻结版本：decode graph、fenced、adaptive/P-first、8 层下限，
开启 `compile_full_prefill`、开启 scheduler trace，并包含 D-only worker fast path。
该候选同时改变了编译尝试、trace 和执行路径，不能把整体结果归因于单个开关。
它尚未包含后述 token-work quota。

| 负载 / graph-compiled | 聚合 tokens/s | 三轮 tokens/s | TTFT P95 / s | 请求平均 TPOT P95 / s | 最大内容事件间隔 P95 / s |
| --- | ---: | --- | ---: | ---: | ---: |
| 混合 | 165.425 | 160.409 / 167.337 / 168.773 | 0.832 | 0.0380 | 0.123 |
| Decode-heavy | 225.658 | 226.485 / 228.119 / 222.447 | 0.825 | 0.0282 | 0.0966 |

原始回执见 [mixed](../evidence/2026-10-10-tp4-optimization/graph-compiled-mixed.json)
和 [decode-heavy](../evidence/2026-10-10-tp4-optimization/graph-compiled-decode-heavy.json)。
两组各 24/24 请求满足既定联合 SLO，但吞吐没有追平任何一次原生 graph 基线：
混合分别低于 initial/bracket 约 **6.56% / 5.38%**，decode-heavy 低约
**7.60% / 5.85%**；对六轮合并原生基线，分别低约 **6.0% / 6.7%**。
TTFT P95 也高于两次基线。该联合配置不支持把 `compile_full_prefill` 改为默认开启；
其默认值仍为 false。三轮不构成统计显著性结论。

[自然语言对照](../evidence/2026-10-10-tp4-optimization/native-graph-vs-graph-compiled.json)
16/16 文本精确一致，两侧固定 oracle 和重复稳定性通过。
[合成 SSE 对照](../evidence/2026-10-10-tp4-optimization/native-graph-vs-graph-compiled-serving.json)
两侧各 56 请求传输通过，跨端点文本 **36/56** 精确一致；28 个重复分组中
原生 8 组不稳定、候选 10 组不稳定。不能将传输通过解释为全量输出等价。
[状态复用探针](../evidence/2026-10-10-tp4-optimization/graph-compiled-state-reuse.json)
完成全部 30 请求：四个单 token P 分组的前后重复均稳定，双 token/max16 分组
在背景请求前、后各自都不稳定，前后首次输出也不同。

[长 P](../evidence/2026-10-10-tp4-optimization/graph-compiled-progress.json) 中仍有
4 个 D 内容事件出现在完整 12000-token P 首次文本之前；
[取消](../evidence/2026-10-10-tp4-optimization/graph-compiled-cancel.json) 与
[严格 trace](../evidence/2026-10-10-tp4-optimization/graph-compiled-trace-check.json)
通过。trace 包含 99 个 cohort、1497 轮回执、160 次 P 完成及 1 次取消，
其中 `full_forward_return=71`、`layer_hooks=136`、`none=1290`。
整网返回来源只允许本轮新建且完整结束的 P；旧/暂停/部分 P 必须使用真实 layer
hooks。`full_forward_return` 证明 native 整网 forward 成功返回和状态就绪，
不单独证明 FX 命中或 device graph 执行；`prefill_compiled_calls` 也只是允许
编译调用的尝试计数。

后续 [四 rank profile 审计](../evidence/2026-10-10-tp4-optimization/graph-compiled-profile-proof.md)
在这个八迭代窗口内确认每个 rank **7 次** `aclmdlRIExecuteAsync`，以及 **1 次**
实际 `npu_fx_compiler inference` runtime range。后者由已安装 Torch-NPU 的
`_CompiledFxGraph.__call__` 包围 `run_kernel`，类别为 `cpu_op`，不是只依据
插件开关或尝试计数推断。每个 range 内包含 40 MoE、30 GDN、10 full-attention、
270 GEMM 调用，支持一次完整 FX forward 的运行证据。

[JSON](../evidence/2026-10-10-tp4-optimization/graph-compiled-profile-proof.json) 与
[审计脚本](../evidence/2026-10-10-tp4-optimization/graph-compiled-profile-proof-audit.py)
保留 marker 实现源码散列、原始事件、时钟换算和调度回执。将 profiler 时间
转换为 monotonic 后，四 rank 的 FX range 都完整落在 scheduler step 1433、
cohort 98 的 **128-token P** schedule/complete 区间中，其来源为 full_forward_return。
请求归属来自时钟对齐后的区间包含关系，FX marker 本身没有 request ID。
随后 cohort 99 的 **12000-token P** 在 steps 1435–1439 按累计层数
8/16/24/32/40 使用 layer_hooks，窗口内没有第二次 FX entry。

因此，本窗口证明的是四个 TP rank 协作执行的一次完整 FX P 和七轮 D replay，
不能把 rank 相加成四个独立 P，也不能将生命周期 `prefill_compiled_calls=72`
当成 72 次实际命中。FX range 的 365.403–380.948 ms 是插桩下的 host 时长，
不是 NPU kernel 时长、occupancy 或无 profiler 的 TTFT。四 rank 自动停止均
有 RECORD 状态警告；结论只覆盖完整导出的事件，不是整个服务生命周期统计。

## Token-work 层配额与 work256 六轮验证

固定 8 层让短 P 也可能跨五轮。D graph 缩短 D 耗时后，1 s TTFT 目标仍可能
允许很多轮，因此纯 deadline 策略未必会增大短 P 配额。为验证 artifact 的
按 P token 数调整层数思路，新增默认关闭的 `prefill_layer_token_budget=0`。
启用后先计算：

```text
base_quota = max(layers_per_slice,
                 min(model_layers, prefill_layer_token_budget // cohort_tokens))
effective_quota = min(remaining_layers, base_quota)
```

整数除法向下取整；随后已校准的 deadline 策略可以继续增大配额。分母为最初
实际获 KV 分配的整个 P cohort 的完整 scheduled-token 总数，包括重算时已有
输出 tokens，既不按单请求分别计算，也不随已完成层数或部分取消而缩小。
fresh admission 预判使用候选 P token 总和；若 KV 仅接纳一部分 P，分配后按
实际 cohort 重算。已准入 D 不会因后验配额变化被删除。

这是 layer-token 工作量启发式，不是硬资源或时间上限：配置的层数下限可让
实际工作量超过预算，且没有 D 竞争时原协议仍允许一次完成剩余 P。
默认 0 保持原有 fixed/deadline 数值路径。非零只允许 decoupled；它既不切分
prompt tokens，也不把 P/D 放进同一模型 batch。
若 P-first 已开启且 fresh 配额覆盖全模型，可在 EWMA 未校准时先选择 P-only；
完成后仍记录并偿还 D 债务。只有 fresh 完整 forward 才可尝试 compiled P。

40 层、8 层下限、预算 262144 的**计算示例**：整个 cohort 为 4096 tokens 时
配额为 40；两条 4096 合计 8192 时为 32；12000 时为 21，下一轮裁剪为剩余 19。
整个 cohort 一次完成的 token 阈值是 6553。这些数值是公式结果，不是实测时长
或性能承诺；65536/131072/262144 需要本机匹配测量，不能套用 NVIDIA 参数。

配置、native scheduler/KV fixture 和 trace checker 的针对性测试为
**188 passed**，覆盖 KV 部分准入、重算 token、部分取消、deadline 下限、D 债务
与 KV 失败回退；另用 checker 验证了四份真实 fixture 生成的 trace。
新增 `work_quota` 记录来源、预算、完整 cohort token 数、配置下限和基础配额；
checker 从最初 schedule 的 token map 复算，并拒绝同 worker 的协议降级。
旧 trace 可按旧协议验证，不能冒充新增配额的证据。

[work256 manifest](../evidence/2026-10-10-tp4-optimization/graph-work256-manifest.json)
记录 05:01 UTC 的冻结来源；实际加载路径也在 manifest 核实。首组计时同时
包含预算 262144、compiled full P、D-only fast path 及连续 D 同流时省去冗余
入口 fence、trace 关闭和不配置 profiler。它是联合改动，不能拆出单个开关的收益。
其余 TP4/EP、输入分布、生成长度和 benchmark 口径与两次原生 graph 基线相同。

| 负载 / work256 stage 1 | 聚合 tokens/s | 三轮 tokens/s | TTFT P95 / s | 请求平均 TPOT P95 / s | 最大内容事件间隔 P95 / s |
| --- | ---: | --- | ---: | ---: | ---: |
| 混合 | 181.517 | 178.678 / 182.767 / 183.174 | 0.687 | 0.0386 | 0.377 |
| Decode-heavy | 245.876 | 245.187 / 244.940 / 247.517 | 0.680 | 0.0276 | 0.347 |

数据见 [mixed](../evidence/2026-10-10-tp4-optimization/graph-work256-mixed.json)、
[decode-heavy](../evidence/2026-10-10-tp4-optimization/graph-work256-decode-heavy.json)
及 [stage 1 summary](../evidence/2026-10-10-tp4-optimization/graph-work256-stage1-summary.json)。
摘要保留来源散列、全部三轮及各自与 initial、bracket、六轮合并基线的比较：

| 相对原生基线 | Mixed 吞吐变化 | Decode-heavy 吞吐变化 |
| --- | ---: | ---: |
| initial，3 轮 | +2.53% | +0.68% |
| bracket，3 轮 | +3.82% | +2.58% |
| 全部 6 轮 | +3.18% | +1.63% |

两组各 24/24 请求满足联合 SLO，goodput 为 2.836 / 3.842 请求/s。TTFT P95
低于两次原生基线；decode-heavy 的请求平均 TPOT P95 却比 initial 高 1.45%，
比六轮合并高 1.43%，因此不能写成所有延迟指标均改善。
混合负载 index 1/2/5、decode-heavy index 7 的文本 hash 跨重复变化。
这些是首组三轮的原始结果；stage 1 summary 保留当时的快照，后续结果不覆写它。

在正确性探针之后，同一冻结服务又完成三轮确认，仍保留已有 adaptive 估计。
原始输入为 [confirm mixed](../evidence/2026-10-10-tp4-optimization/graph-work256-confirm-mixed.json)
和 [confirm decode-heavy](../evidence/2026-10-10-tp4-optimization/graph-work256-confirm-decode-heavy.json)。

| 负载 / 确认阶段 | 聚合 tokens/s | 三轮 tokens/s | TTFT P95 / s | 请求平均 TPOT P95 / s | 最大内容事件间隔 P95 / s |
| --- | ---: | --- | ---: | ---: | ---: |
| 混合 | 183.697 | 184.288 / 182.934 / 183.874 | 0.664 | 0.0383 | 0.373 |
| Decode-heavy | 244.152 | 243.248 / 243.351 / 245.875 | 0.680 | 0.0278 | 0.353 |

[六轮完整汇总](../evidence/2026-10-10-tp4-optimization/graph-work256-summary.json)
按每负载全部 3072 completion tokens / 总 wall time 计算；各延迟分位数取全部
48 请求的 nearest rank 第 46 项。合并 mixed 为 **182.600 tokens/s**、
TTFT P95 **0.66888 s**、请求平均 TPOT P95 **0.03849 s**；decode-heavy 为
**245.011 tokens/s、0.68011 s、0.02769 s**。两负载各 48/48 满足联合 SLO，
goodput 为 2.853 / 3.828 请求/s。

| 全六轮候选相对原生基线 | Mixed 吞吐变化 | Decode-heavy 吞吐变化 |
| --- | ---: | ---: |
| initial，3 轮 | +3.14% | +0.32% |
| bracket，3 轮 | +4.44% | +2.22% |
| 全部 6 轮 | +3.79% | +1.27% |

候选 decode-heavy 确认阶段略低于原生 initial；六轮请求平均 TPOT P95 比
initial 高 1.84%、比原生六轮合并高 1.82%。不能写成所有指标不退化。
两阶段是顺序运行且中间插入了其他诊断，不是六个独立冷启动重复；没有统计
显著性结论。六轮合成文本 hash 在 mixed index 1/2/3/5 和 decode-heavy
index 1/7 存在变化，均在摘要和原始回执保留。

[自然语言](../evidence/2026-10-10-tp4-optimization/native-graph-vs-graph-work256.json)
16/16 文本与原生 graph 精确一致；
[合成 SSE](../evidence/2026-10-10-tp4-optimization/native-graph-vs-graph-work256-serving.json)
两侧各 56 请求传输通过，文本 40/56 精确一致，原生 8/28、候选 9/28 分组
跨重复不稳定。[状态复用](../evidence/2026-10-10-tp4-optimization/graph-work256-state-reuse.json)
30/30 完整，四个单 token 分组共 16 条完整序列与 graph-adaptive 历史逐条相同；
双 token/max16 在背景请求后仍不稳定，且前后首次输出不同。

[长 P 进度](../evidence/2026-10-10-tp4-optimization/graph-work256-progress.json)
记录完整 12000-token P 首次文本前 1 个 D 内容事件，与 21+19 两轮的设计
预期相容，但客户端事件不能单独证明层配额；
[trace-off 取消](../evidence/2026-10-10-tp4-optimization/graph-work256-cancel.json)
验证约 0.1 s 后的客户端取消和 D 继续响应。trace 关闭时不能据此声称 worker
已发送取消 ACK 或 KV 已按协议释放；下述独立 trace-on 诊断补齐协议证据。

同一冻结运行时源码（现已提交为 `4caeebaf16c611806de2d5275748ead2a05eba59`）
另启 trace-on 服务，见
[诊断 manifest](../evidence/2026-10-10-tp4-optimization/graph-work256-traced-manifest.json)。
先用一个 12000-token P 请求及 8-token 输出预热校准纯 P 耗时，
[校准回执](../evidence/2026-10-10-tp4-optimization/graph-work256-traced-calibration.json)
**仅作诊断准备，不加入性能对照**。这不是六轮计时的额外重复，也不证明冷启动
未校准时必定采用同一层配额。随后
[进度探针](../evidence/2026-10-10-tp4-optimization/graph-work256-traced-progress.json)
和 [取消探针](../evidence/2026-10-10-tp4-optimization/graph-work256-traced-cancel.json)
均通过；自然语言 [再次 16/16 精确一致](../evidence/2026-10-10-tp4-optimization/native-graph-vs-graph-work256-traced.json)。

[严格 checker](../evidence/2026-10-10-tp4-optimization/graph-work256-traced-check.json)
验证 466 个层进度回执、22 个 cohort、21 次 P 完成和 1 次取消；
20 个回执来自 `full_forward_return`、4 个来自 `layer_hooks`，其余 442 个无 P；
24 个 P 调度回执携带 layer-token 配额。
[紧凑证据](../evidence/2026-10-10-tp4-optimization/graph-work256-traced-proof.json)
保留完整 trace SHA256 和 51 个带行号的原始 JSONL 行，
[提取脚本](../evidence/2026-10-10-tp4-optimization/graph-work256-traced-proof-audit.py)
验证以下实际路径：

- Cohort 4：step 21 一次准入全部 12000 tokens，P 累计到 21/40 层；step 22
  不再重复准入 P tokens，只推进剩余 19 层并完成。两轮都是 `layer_hooks`；
  P 完成之前已退休一轮 D，客户端也观察到 1 次 D 内容事件。
- Cohort 6：step 86 推进 21/40 层，step 87 收到取消控制并返回取消 ACK，
  累计层数保持 21，未产生 P 完成；step 88 cohort 已清空，D 继续服务。
- 4 次 `token_budget_prefill_first` 都只准入 P，延后 D 不在该轮 token map；
  完成后建立 D 债务，紧接下一轮 `repay_decode` 服务这些 D 并清债。前后
  `computed_before` 与上一轮实际调度 tokens 对得上，没有 P-only 轮额外推进 D。

这些回执验证 host 生命周期与提交的层边界，不证明设备空间重叠、实际 KV 内容
或每次完整 P 都命中 FX。证据针对当前机器和测试范围；组件默认仍为关闭编译、
layer-token budget 0，本机 graph 启动脚本显式选择已测的 compile+262144 配置。

## Isolated8 独立通信组的 eager 实测

[isolated8 manifest](../evidence/2026-10-10-tp4-optimization/isolated8-manifest.json)
对应冻结源码的 decoupled、overlap、独立 P/D 通信组、固定 8 层配额。
该快照没有后续 D-only fast path，也没有 decode graph、adaptive layers 或 P-first。
其余模型、TP4/EP、输入、预热、三个重复和统计口径与 eager 对照相同。
不能将它的结果归于后续源码，或用 graph 对照计算独立通信组的收益。

| 负载 / isolated8 | 聚合 tokens/s | 三轮 tokens/s | TTFT P95 / s | 请求平均 TPOT P95 / s | 最大内容事件间隔 P95 / s |
| --- | ---: | --- | ---: | ---: | ---: |
| 混合 | 28.934 | 28.575 / 29.118 / 29.116 | 2.958 | 0.238 | 0.451 |
| Decode-heavy | 31.729 | 31.870 / 31.600 / 31.720 | 1.794 | 0.232 | 0.296 |

原始回执见 [mixed](../evidence/2026-10-10-tp4-optimization/isolated8-mixed.json)
和 [decode-heavy](../evidence/2026-10-10-tp4-optimization/isolated8-decode-heavy.json)。
聚合吞吐分别比本报告的原生 eager 三轮低约 3.3% / 2.6%，TTFT P95 也更高；
两组各 24 请求的联合 SLO 均为 0/24。没有同一时间、同一 hot path 的 shared8
消融，不能将差值单独解释为换 communicator 的收益或损失。
混合负载 index 1/2/5、decode-heavy index 3/7 在三个重复中的文本 hash 不同；
这同样阻止对任意合成输入作重复一致性承诺。

[自然语言对照](../evidence/2026-10-10-tp4-optimization/native-vs-isolated8.json)
复用原生 eager 的基线报告，16/16 文本精确一致，两侧 oracle、重复稳定性和
isolated/paired 对照均通过。
[长 P 回执](../evidence/2026-10-10-tp4-optimization/isolated8-progress.json)
记录完整 12000-token P 首次文本之前的 4 个 D 内容事件；
[取消](../evidence/2026-10-10-tp4-optimization/isolated8-cancel.json) 和
[trace checker](../evidence/2026-10-10-tp4-optimization/isolated8-trace-check.json)
通过，后者检查 930 轮层数回执、40 层总数、72 次 P 完成及 1 次取消。
该冻结版本没有新增的 `progress_source` 字段，checker 只验证当时的层数协议。
这些是输出和生命周期证据，设备区间另由下述 profile 验证。

[四 rank isolated/shared 审计](../evidence/2026-10-10-tp4-optimization/isolated-vs-shared-profile.md)
比较两次限定为八个引擎迭代的独立 probe。两侧每 rank 都包含 **9 个 forward
等价量**、360 次 MoE、270 次 GDN、16086 个计算 kernel；forward 等价量来自
40 MoE/30 GDN 层及 logits all-gather 次数交叉核对，没有直接整网计时 range。
每个 forward 的 all-reduce 均为 81 次。归一化事件数量如下，四个 rank 一致：

| 路径 | Python event record / forward | Python event wait / forward | CANN record / wait / forward |
| --- | ---: | ---: | --- |
| 旧 shared8 | 215.667 | 205.444 | 462.333 / 369.444 |
| isolated8 | 125.333 | 123.444 | 372.000 / 287.444 |

Python record/wait 分别减少 **41.9% / 39.9%**，与去除每 collective 的 P/D gate
和逐层 host 切换一致。但两侧执行顺序也不同：isolated 连续提交 P slice，再
提交完整 D；旧 shared 保留逐层交替。因此这不是只改变 communicator identity
的消融，不能把端到端收益全归给独立通信组。

通过日志核实的物理 stream 为 P=45、D=44；对明确标为 AI_CORE、AI_VECTOR、MIX
的计算任务先各自求区间并集，再求 P/D 交集，排除通信、DMA 和 host range。
isolated 四 rank 的 P/D 计算交叠为 **9.703–12.246 ms**，占计算活跃并集
**2.63%–3.30%**；旧 shared 为 **10.649–16.660 ms、2.86%–4.49%**。
本次没有增加计算重叠，不能称为更好的空间并行，也没有证明 Cube/Vector 资源隔离。
计算活跃并集占 profile 窗口约 13.86%–13.96%，旧 shared 为 12.66%–12.78%；
这是区间覆盖率，不能解读为 AI Core occupancy 或用于替代无 profiler 的吞吐。

[JSON](../evidence/2026-10-10-tp4-optimization/isolated-vs-shared-profile.json) 与
[提取脚本](../evidence/2026-10-10-tp4-optimization/isolated-vs-shared-profile-audit.py)
保存 source hash、stream/通信组映射日志、各 rank 原计数和 trace/CSV 精确交叉核对。
CPU inclusive 包含嵌套操作，不可相加；exclusive 只减掉同线程包含的 cpu_op，
仍含未插桩时间。HCOM API 是 host 时长，communication-track 并集包含等待。
导出中没有显式归类为 HCCL 的硬件任务，不代表没有通信：SDMA、Notify Wait、
Write Value 另行保留。
四 rank 自动停止都报告 RECORD 状态警告；所选计算区间均位于导出的
PROFILING_ENABLE/DISABLE 之间，且工作量计数一致，但仍不能证明整个请求周期
及每条 profiler 记录均完整捕获。

## 与 MuxWise 的差异及未完成项

正式论文和固定 artifact 来源见 [references.md](references.md)；详细对照见
[original-comparison.md](../evidence/2026-10-10-tp4-optimization/original-comparison.md)。
论文使用的 NVIDIA 资源分区及预测参数不能直接移植为 Ascend 性能承诺。

| 机制 | 原始 artifact / 论文 | 本机验证状态 |
| --- | --- | --- |
| Decode graph | artifact 按资源分区和 batch size 捕获，整网 replay | graph8 已捕获、运行并完成阶段性匹配对照；吞吐仍低于原生 graph，合成文本非全精确 |
| P 层配额 | artifact 使用 token budget / P tokens；论文另有时长匹配、竞争估计与 SLO 策略 | 可选 layer-token 基础配额叠加本机 EWMA/deadline；work256 六轮有小幅吞吐收益，部分 TPOT 指标退化，不等价于论文 contention predictor |
| 通信组 | artifact 为 P 建立独立 TP communicator | isolated8 已完成 TP4 eager、正确性与四 rank profile；事件数下降，计算重叠未增加，本轮性能未优于原生 |
| 资源分区 | CUDA GreenContext 分割 SM 资源 | 目前普通 NPU streams；未证明 Cube/Vector 资源隔离 |
| Prefix reuse | artifact 保留共享 KV/tree cache | 当前 Qwen3.5 验证要求 APC off、Mamba none；本次原生也相同 |
| P 完成与回收 | artifact 通过 event query 和跨 rank 确认完成 | 插件依赖严格完成/取消 ACK 释放 KV；进一步延迟采样协议尚未验证 |

公开 artifact 本身包含 temporary demo 策略，不等于论文完整 predictor。
当前优化也未引入 NVIDIA 参数、token chunking 或 P/D mixed model batch。

后续 graph 优化仍必须使用原生 graph 的匹配基线，分别保留原生和候选的
capture sizes、编译模式、采样/输入参数及每轮结果。客户端 SSE 成功、文本
hash 相同或并发发出八个请求，都不能证明设备 graph replay 或实际 D batch=8。
[graph serving 检查器](../scripts/check_decode_graph_serving.py) 提供固定合成 token IDs、
单 token P、并发组和缩减请求组的严格传输回执；还需 worker graph snapshot、
调度 trace 和必要的设备 profile 相互对应。目前没有结果支持将
graph8/isolated 推荐为已全面优于原生的默认配置。

## 证据文件与校验

已完成回执、manifest、紧凑 profile/trace 审计及脚本已归档到
[evidence/2026-10-10-tp4-optimization](../evidence/2026-10-10-tp4-optimization/README.md)。
[归档索引](../evidence/2026-10-10-tp4-optimization/archive-index.json) 记录原路径、
大小和 SHA256；文件为原产物的逐字节副本。未复制残缺回执、完整 profiler、
大型 scheduler trace、server log 或未完成的数据。紧凑 profiler 报告保留
完整 trace 的源散列和选定事件，但不能替代完整 trace 重算；复现输入位置见
归档说明。本轮二十份 benchmark 输入的 SHA256 为：

| 文件 | SHA256 |
| --- | --- |
| `native-mixed.json` | `3b361649815a35f42534c1996eb923562f9e2ebaf10f28e048d87f63a017a9d5` |
| `native-decode-heavy.json` | `d3721df7c077c466a1c1be2f4a84d3a20819fe6bb428e49d9ddd9b297dc573c4` |
| `pfirst-mixed.json` | `f1a9b02cffd9fccff3053424efe86f58b042e4cbd779eee0722a62da11d7c15b` |
| `pfirst-decode-heavy.json` | `b16b08af68cbf076532750a7fb6ec872e29a74e835e4ce881bf1e8ea2a3c5d1a` |
| `native-graph-mixed.json` | `1fb2c066347a4f19b2d44677e743ef755017f53aa6abc22fa4f3771b164254d9` |
| `native-graph-decode-heavy.json` | `9f6cc8d5d3f6c5928bfedf1fa42c7bcd25e8c773987fd8d06e0ee8e33fb1da5e` |
| `graph8-mixed.json` | `036a51cf28136a51ae61c75484828675deb179828e81f2fa1e7197e8d74fbba4` |
| `graph8-decode-heavy.json` | `e9cb85bc59ec1430dead24a45e5fadfe6b3fc4fe2f6127ad5eecf401d6aa8854` |
| `graph-adaptive-mixed.json` | `29f7d2ac10160605b42d3cb94c8aab5fde8a4aae9a7d9ea8a37638217865c971` |
| `graph-adaptive-decode-heavy.json` | `190e0f66b6234d3011aeca01a608ac469ba93287952b01366dc23c33bf15ffcf` |
| `isolated8-mixed.json` | `3b057f12cfd0eb7d3814bced71898083ba8a73a8c0378bf2af2b4dea0aa791e8` |
| `isolated8-decode-heavy.json` | `c7b7379b59f75c093ae51b83e701772ff2116e33c9ab51b1debe6e2234324534` |
| `native-graph-bracket-mixed.json` | `1be9428c43db5555e45a33c73855fc79d9ab9b024b5d4fb053bd60bc5895dab8` |
| `native-graph-bracket-decode-heavy.json` | `f8c5e3560573ef39010e88d4266dc3bc3893729e508218fc5b90827e5ca048cb` |
| `graph-compiled-mixed.json` | `972d9c9e6886b921169fadb551e5de1ecbf8bc5b1d844c53f7a463632f62f720` |
| `graph-compiled-decode-heavy.json` | `8fa574f2f814e1edaa39e71d2d4b08889f5b9fc53aad2afd0a13cd0baa1475c2` |
| `graph-work256-mixed.json` | `3620aa4e15c06c3cd41c3bddb1b04f4172c01368a75fab3579c24a3283ff961e` |
| `graph-work256-decode-heavy.json` | `29fa467c8a7960988f9b819f6f81b59eb3a6cd672bd12b032786908ea4ea4eb0` |
| `graph-work256-confirm-mixed.json` | `e2e013064350358edd2233e9a6b65f21b95d0869ca5ad7d6d9ba21e5d20faf0b` |
| `graph-work256-confirm-decode-heavy.json` | `4031dea6dc89c76390765662897bb27b14b8024aaea4da00ccbaed5219d089d3` |

`eager-summary.json` 的 SHA256 为
`eebcf27a5a0d85b66add9a7585e5e95b600b7ff63de1dafcb2a792edd5707dd8`。
Profile 的每个 trace、profiler metadata 和分析脚本散列保存在上述审计 JSON，
无需通过累计 CPU inclusive 时间或相加各 rank 的设备时间重新估算吞吐。


最终验证见 [validation-final.json](../evidence/2026-10-10-tp4-optimization/validation-final.json)：
冻结运行时文件散列一致；完整 native 环境测试 **451 passed、2 skipped，39.07 s**；
不含 torch/vLLM 的隔离 CI 环境 **229 passed、130 skipped，1.24 s**，后者明确
跳过原生调度器和 worker 测试。Ruff、空白检查和 graph 启动脚本语法检查通过，
测试命令与原日志 SHA256 均保留。

TP4 试验结束后已恢复原有两组 TP2 服务，见
[恢复记录](../evidence/2026-10-10-tp4-optimization/restored-services.json)：
8000 原生使用逻辑卡 0、1（eager、APC on、Mamba align）；8002 MuxPD 使用
逻辑卡 2、3（adaptive fenced eager、APC off、Mamba none）。
[原生 smoke](../evidence/2026-10-10-tp4-optimization/restored-native-smoke.json) 与
[插件 smoke](../evidence/2026-10-10-tp4-optimization/restored-muxpd-smoke.json) 均通过；
这些恢复检查不是 TP4 性能样本。原生 core 仓库保持干净，Ascend 仓库仅保留
用户原有的 `csrc/build_aclnn.sh` 修改，未改动该文件。
