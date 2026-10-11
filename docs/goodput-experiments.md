# 持续到达 Goodput 实验与 graph overlap 使用说明

本页说明如何验证 TP8 下的持续处理能力，不预设插件优于原生 vLLM。
[上一轮有限争用实验](performance-tp8-contention.md) 证明了缩短 decode
停顿需要付出的吞吐与 prefill 延迟代价；四条 D 加少量长 P 的有限队列，
不能直接给出可持续到达率或论文意义的容量收益。

本页所述 graph overlap 是实验性实现：`da9a8b1` 引入独立通信组下的 D
graph / raw P，`71e70b4` 调整为先提交 D，再执行 P 的主机层调度，
`62c6353` 加入持续到达基准。本说明起草时，这些改动位于尚未合并的实验
分支，相对已发布的 `70ba12d` 仍需独立验证；运行时必须记录实际 commit、
源文件 SHA 和全部参数。
默认 [graph 启动脚本](../scripts/serve-qwen35-decode-graph.sh) 仍使用
`fenced`，没有自动切换为 overlap。

## 方法与论文的关系

[MuxWise 正式论文 §4.1、§4.2.3](https://arxiv.org/html/2504.14489v3#S4)
通过递增 Poisson 到达率评估尾部 TBT 和容量，并预先调优 chunked-prefill
的 token budget。它使用 pooled TBT P99；这与“每条请求的最大 gap”不同。
例如 256-token 输出只有一次长停顿，仍可能满足 pooled P99，而该请求的
最大 gap 已经超标。因此本基准保留两套指标，不相互替代，也不把平均 TPOT
当成尾部 token 间隔。[作者基准参考](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/benchmark/pdmux/bench_serving.py#L338)

当前插件没有移植论文的 CUDA Green Context 空间分区、完整离线争用预测器
或跨请求 KV 复用机制。实验性双流允许调度与执行重叠，但实际 kernel 交叠
需另行采集设备 profile。通信组隔离不是 AI Core 分区，客户端时间也不能
证明设备并行。

## 实验性 TP8 graph overlap

先确认所选八卡空闲，并安装当前待验证分支。以下是**固定基础 8 层**的
候选入口，不是默认推荐配置，也不是既有实机诊断参数的原样复现：

```bash
source /root/workspace/runtime/env.sh
uv pip install --python /root/workspace/.venv/bin/python --no-deps -e .

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MUXPD_TP_SIZE=8 \
  VLLM_SERVER_PORT=8002 scripts/serve-qwen35-decode-graph.sh \
  --batch-admission-policy-config \
  '{"execution":"decoupled","stream_mode":"overlap","overlap_communication":"isolated","decode_graph":true,"compile_full_prefill":false,"layers_per_slice":8,"prefill_layer_token_budget":0,"adaptive_layers":false,"prefill_first":false,"decode_budget_guard":false,"trace_path":null}'
```

后面的完整 policy JSON 覆盖启动脚本自带 policy。各项约束如下：

| 配置 | 含义 |
| --- | --- |
| `execution=decoupled` | 完整 P 保存在 worker 中，D 可以跨多轮独立返回 |
| `stream_mode=overlap`、`overlap_communication=isolated` | P 使用独立 TP/EP 通信组；保留初始化、KV 准备与采样所需依赖 |
| `decode_graph=true` | D 使用独立静态输入、workspace 和捕获流；脚本提供 `FULL_DECODE_ONLY` 与 1/2/4/8 capture sizes |
| `compile_full_prefill=false` | P 使用可逐层暂停的原始模型调用；其 graph mode 始终为 `NONE`，单 token P 也一样 |
| `layers_per_slice=8`、`prefill_layer_token_budget=0` | 关闭按 token 工作量扩大配额；有 D 时基础片段为 8 层，末片裁剪到剩余层数 |
| `adaptive_layers=false` | 不用 TTFT 启发式放大片段 |
| `prefill_first=false`、`decode_budget_guard=false` | 这两个当前只支持 adaptive fenced 的策略不能直接套到 overlap |
| `trace_path=null` | 正式计时不启用逐轮 trace；生命周期诊断单独运行 |

“固定 8 层”不意味着每一轮都必须推进 8 层：没有活跃 D 时调度器可以完成
剩余 P，取消/暂停有其独立控制语义。`prefill_layer_token_budget=262144`
即使配合 `adaptive_layers=false`，仍会按 cohort token 数扩大基础层配额，
例如单条 12K P 可以得到 21 层；不能把这一配置称为固定 8 层。

上述路径仍限制为已支持的 Qwen3.5 / A2 / BF16、V1 同步 runner、纯文本、
TP+EP，禁用 prefix caching、Mamba `align`、FlashComm、MC2、共享专家
侧流、EPLB、offload 与推测解码。P/D 共用权重和 KV 所有权；取消必须等待
worker ACK，不能为了 goodput 提前释放 P 的 KV。存在未完成 P 时的 `keep`
暂停以及同一 worker 进程内销毁后重新初始化图服务，仍不受支持。
具体拒绝条件以 [配置校验](../src/vllm_muxpd/config.py)、
[worker 校验](../src/vllm_muxpd/worker.py) 和
[图校验](../src/vllm_muxpd/decode_graph.py) 为准。

`da9a8b1` 的早期 P 先提交路径已完成本机自然语言 16/16 文本精确一致、
graph serving、取消及满 token budget 的诊断；这不是 token IDs 全等证明，
也不自动覆盖 `71e70b4` 的 D 先提交路径或上面的固定 8 层参数。新版本的
正确性、生命周期与容量结果需各自记录，当前不宣称持续 goodput 或总吞吐
收益。

上述 P 先提交和 D 先提交两组诊断均使用 `prefill_layer_token_budget=262144`
的 work256 配置，以保持提交顺序消融的配额一致；它们均不是固定 8 层。
后续干净计时分别保留 work256 与 `prefill_layer_token_budget=0` 的固定 8 层
配置，不能将两者合并成同一个候选。

## 公平原生基线

原生服务使用同一模型、TP8/EP8、16K context、maxseq=8、图捕获大小、
内存比例和缓存策略。不能用插件启动脚本冒充原生：它会装入插件
scheduler/worker。以下是匹配同步条件的原生 16K 入口，与插件择一启动：

```bash
source /root/workspace/runtime/env.sh
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=0 \
  /root/workspace/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /root/workspace/models/Qwen3.5-35B-A3B \
  --served-model-name Qwen3.5-35B-A3B --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --enable-expert-parallel --dtype bfloat16 \
  --max-model-len 16384 --max-num-seqs 8 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 --language-model-only \
  --no-enable-prefix-caching --mamba-cache-mode none \
  --enable-chunked-prefill --no-async-scheduling --no-enforce-eager \
  --compilation-config \
  '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}'
```

探索 token budget 为 `256,512,1024,2048,4096,8192,16384`；修改的是
`--max-num-batched-tokens`，context 上限保持 16384。公开全部尝试，使用探索
数据选择适合预登记 SLO 的原生参数，再用新 seed 确认。只测试 2K 与 16K
不足以断言胜过调优后的 chunked-prefill。

此外，验证原生默认/启用 `--async-scheduling` 的可用性并纳入最佳基线；
不能因为插件要求同步，就把原生可用的 async 优化排除。若某配置在固定
Ascend 版本上不支持，保留启动错误与原因，不称其已完成性能比较。更大的
maxseq=16/32 和匹配 capture sizes 是双方共同验证的后续矩阵，不单方面
为候选或原生放宽并发。

原生 prefill 分段图也必须纳入候选。固定 Ascend 源码
[`03766ac` 的 platform.py](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/platform.py#L555)
支持 `FULL_AND_PIECEWISE`，而固定 vLLM 源码的
[默认 attention 切分点](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/config/compilation.py#L755)
包含 GDN。GDN 不支持任意 prefill 的整网 FULL 捕获，不等于其外侧不能
使用分段图。上述 `FULL_DECODE_ONLY` 启动命令只是一项对照，不能称为
最佳原生配置。

本机后续事前登记的分段图候选为 token budget 512/1024、maxseq=8、
async、OMP=1，保留各自的 D-only 对照。512 的捕获尺寸为
`[1,2,4,8,128,136,256,512]`，1024 另加 `1024`，上限设为对应 budget；
136 覆盖 P128 加 1–7 个 D 的总 token 数。这是有限的初始候选，尚未验证
实机可用性或确定最优尺寸。配置 `mode=3`、`FULL_AND_PIECEWISE`、
`use_inductor_graph_partition=false`，关闭 `enable_sp`、`fuse_gemm_comms`，
保留默认 `splitting_ops`；平台会关闭 `npugraph_ex` 和 static kernel。
因此该比较包含编译路径变化，不能描述为只改一个 graph 开关。

先在独立诊断进程确认最终模式、八个 rank 的真实 P/混合批分段 replay、
长请求连续状态及取消后恢复，再移除 observer 并重启进行干净计时。
捕获内存不足、模式降级、正确性失败都须保留，不能换回 D-only 后仍标为
分段图。编译后的 FX 调用本身不能证明 ACL graph replay。

## 生成与复用到达计划

[benchmark_goodput.py](../scripts/benchmark_goodput.py) 支持独立 plan 文件。
默认负载为 80% `128 input / 256 output` 和 20% `12000 input / 8 output`，
到达为 Poisson，类别独立采样。arrival、class、prompt 使用独立 RNG，
不同模型配置重放同一个 plan SHA；不等待已有请求完成后再生成到达。

探索示例，先创建 plan，再分别对各个已启动服务重放：

```bash
python3 scripts/benchmark_goodput.py --plan-only \
  --output local-results/tp8-goodput/plan-rate075.json \
  --request-rate 0.75 --warmup 20 --duration 60 --seeds 101

python3 scripts/benchmark_goodput.py \
  --plan local-results/tp8-goodput/plan-rate075.json \
  --base-url http://127.0.0.1:8000 --label native16k-sync \
  --output local-results/tp8-goodput/native16k-rate075.json

# 停止原生并启动候选后，重放同一计划。
python3 scripts/benchmark_goodput.py \
  --plan local-results/tp8-goodput/plan-rate075.json \
  --base-url http://127.0.0.1:8002 --label overlap-graph-fixed8 \
  --output local-results/tp8-goodput/overlap-rate075.json
```

命令中的 `python3` 应来自含 `httpx` 的已验证环境；也可显式使用
`/root/workspace/.venv/bin/python`。不运行服务时，`--plan-only` 不需要 endpoint。
输出路径必须不存在，脚本拒绝覆盖既有证据。`--plan` 不允许同时传入 rate、
duration、seed、模型或长度等生成参数，避免声称相同 trace 却更改请求。

未指定 `--plan` 时脚本也可直接生成并执行，同时保存 `.plan.json`。
可调整 `--long-fraction`、`--long-input-tokens`、`--long-output-tokens`、
`--short-input-tokens` 和 `--short-output-tokens`；所有输出长度必须至少为 2。
`--long-fraction 0` 是短请求控制，`--long-input-tokens 8192` 是 8K 验证。
本版本只生成 Poisson 到达，尚未实现同均值突发 trace 生成选项。

## 探索与确认分离

| 项目 | 探索 | 确认 |
| --- | --- | --- |
| seed | 101 | 1001、1002、1003 |
| 连续暖场 | 20 s | 60 s |
| 到达窗口 | 60 s | 300 s；边界另验证 600 s |
| 初始 rate，requests/s | 0.25、0.5、0.75、1、1.5、2 | 根据探索预先冻结的边界及邻近点 |
| 用途 | 排查与参数选择 | 对未参与调优的新 trace 比较 |

探索按相同规则细化各系统相邻通过/失败 rate 的中点，不只细化候选有利区间。
正式确认前冻结候选/基线配置、源码、arrival plan 和比较阈值；平衡运行顺序，
在末尾补原生 bracket。绘图、CPU 测试、trace 和 profiler 不与计时混跑。
更改候选、阈值或协议后不得继续混用同一确认结果池。

默认 `--drain-timeout 120` 从固定到达窗末尾开始计算；每请求另有
`--request-timeout 600`。暖场是连续到达前缀，进入测量窗前不排空请求，
所有暖场回执单独保留。到达窗口结束后停止注入，等待 drain 或取消到期
请求。未完成或控制错误会保留失败证据并停止后续 seed，避免在未排净服务上
继续下一组试验；这类运行不能获得容量比较资格。

## 指标、边界与失败证据

所有请求都经过两个阶段，因此**每条测量请求同时检验 TTFT 和 decode gap**，
不再按“短 D 请求只看 gap、长 P 请求只看 TTFT”的角色方式计算 goodput。
固定公开 `100/200 ms gap × 1/2 s TTFT` 四组阈值，不根据结果选择唯一展示项。

| 输出字段 | 解释 |
| --- | --- |
| `pooled_observed_event_gap_s.p99` | 所有测量到达请求的相邻 SSE token 事件间隔合并后的 P99 |
| `paper_like_pooled_gap_p99_passed` | 完整、单 token 可观察的请求满足 pooled gap P99；这是延迟判据，不自动证明容量 |
| `pooled_gap_and_ttft_p99_passed` | 在上述条件外加入 scheduled TTFT P99 阈值，明确是额外约束 |
| `single_token_request_goodput_per_s` | 同时满足 TTFT 和每请求最大 gap 的完成请求数 / 固定到达时长 |
| `event_gap_request_goodput_per_s` | 允许多 token 事件的事件级观察；不能冒称逐 token SLO |
| `fixed_window_output.validated_tokens_per_s` | 固定观察窗内实际收到的 token 数 / 窗口时长，只计最终通过传输校验的响应 |
| `completed_cohort_output_tokens` | 测量窗内到达请求的完整输出，包含 drain，不能与固定窗口 token 数混用 |
| `eligible_for_capacity_comparison` | 完整运行、无控制错误、全部传输有效、单 token 可观察、dispatch 合格且未超 drain；仍需检查跨 rate 稳定性 |

TTFT 的主口径从**计划到达**开始计算，另存实际 coroutine 开始到首 token
的 TTFT。`send_lag_s` 是计划到达到 coroutine 入口的差，早于 JSON 编码、
建连和 socket 写入，不是真实 wire-send 时间。默认 `--max-send-lag 0.05`
限制客户端调度迟滞；超限不移动计划时间修补结果。

gap 时间是完成 SSE frame 的 HTTP 字节到达时间。多 token chunk 不拆出
虚构的 token 时间；即使每个 frame 只含一个 token，多个 frame 也可能共享
一次网络 read 的时间戳。所有百分位使用 `(n-1)*p/100` 线性插值。完整请求
的 gap 包括观察窗结束后的 drain，不能裁掉尾部停顿来改善 SLO。

输出 tokens/s 按固定观察窗口统计，所以可以包含暖场请求在窗口内实际
生成的部分；arrival-cohort goodput 只评价测量窗内到达的请求。后者的分母
不会因为 drain 较短、较长或失败而改变。失败/超时/格式错误请求不进入
goodput 分子，但仍保留在所有计划到达请求的达标率分母中。

`backlog_samples` 记录客户端已计划到达减去已终止请求，以及尚未开始的数量；
它不是服务端的 waiting/running 队列。需要结合 backlog 漂移、失败、排空时间、
独立 seed 与延长窗口判断可持续性，单次低 P99 或短窗口 goodput 较高不足以
证明稳定容量。原生与插件的合成输出可能不同，影响 MoE 路由；另行记录
文本 oracle、token 重复稳定性及跨配置差异，不将传输成功等同数值等价。

每次执行保留三份文件：

- `.json`：计划、完整请求/事件、SSE 原始 body、指标、错误、producer/parser SHA。
- `.events.jsonl`：追加的生命周期与原始 HTTP body chunks，字节先保存再解析；
  周期性 flush，不承诺断电时最后一段仍存在。
- `.progress.json`：计时期间的小型 backlog 快照，避免不断重写完整 raw JSON。

plan 单独持久化，最终报告还记录 journal 的字节数与 SHA。异常中断时应同时
保留 report、progress 和 journal；不能只复制最后一次成功 summary。
正式性能结论前，须用独立审计从原始字节重新核对传输、计划、分母、窗口和
SLO。验证工具不导入 torch，也不需要 NPU：

```bash
/root/workspace/.venv/bin/python -m pytest tests/test_benchmark_goodput.py -q
/root/workspace/.venv/bin/ruff check \
  scripts/benchmark_goodput.py tests/test_benchmark_goodput.py
```

运行上述 CPU 检查时应避开任何推理计时窗口。完整原生启动参数、模型路径、
设备状态、源码 hash、所有失败配置和探索/确认选择过程，应随最终证据一起
发布；本使用说明不替代实际运行 manifest 或结果报告。
