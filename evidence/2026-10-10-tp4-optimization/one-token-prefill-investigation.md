# TP4 单 token prefill 的图模式兼容性调查

记录日期：2026-10-10。调查只读取已完成回执、启动日志和固定版本源码，没有发送 NPU/API 请求或修改原生代码。现已完成原生 eager、原生 graph、插件 graph 三路同 payload token-ID 对照，各 30 条。Issue 草稿见 [one-token-prefill-issue-draft.md](one-token-prefill-issue-draft.md)，尚未发布。

## 当前结论

原生 `FULL_DECODE_ONLY` 的四组单 token prompt 测试，在重复请求中都从**第 0 个生成 token**出现差异；原生 eager 和插件 graph 的对应四组在本次所有重复及长 prefill 前后均一致，且两者全部 16 条目标响应的完整 token 序列逐条相同。自然文本 `Hello` 的单 token ID 也复现了原生 graph 现象，所以不能仅归为 `[100]` 的 UTF-8 文本显示问题。本次三路中，单 token 首步不稳只在原生 graph 观察到。

源码存在值得验证的具体不一致：原生图 dispatcher 在非 speculative 模式下按 query shape 判定 uniform decode，能够接纳一个 token 的新 prefill；GDN builder 却会按 `is_prefilling` 把同一请求认作 prefill。GDN 的 decode 图持有静态 state 索引，而这些 buffer 仅在 `num_prefills == 0` 时更新。这个条件链**支持检查错误路由及状态复用的假设，尚未直接证明某次请求读了哪个旧 cache block**。

三路的双 token prompt 在 `max_tokens=16` 时都存在第 1 个或更后生成 token 的差异，**原生 eager 同样复现后段不稳**。这是另一项尚未解释的现象。不能把全部后段差异、不自然文本或所有 MoE 数值差异都归因于单 token prefill 路由，也不能宣称插件解决了全部生成正确性问题。

## 证据与实验条件

| 项目 | 原生 eager | 原生 graph | 插件 graph |
| --- | --- | --- | --- |
| 完整回执 | [native-eager-state-reuse.json](native-eager-state-reuse.json) | [native-graph-state-reuse.json](native-graph-state-reuse.json) | [graph-adaptive-state-reuse.json](graph-adaptive-state-reuse.json) |
| UTC 时间 | 04:54:45–04:55:45 | 04:40:26–04:40:36 | 04:25:52–04:26:06 |
| 请求数 / transport 校验 | 30 / 全通过 | 30 / 全通过 | 30 / 全通过 |
| fingerprint 尾部（共同前缀 `vllm-0.1.dev1+gd0f22d2bd-tp4-ep-`） | `09336364` | `30ad75cc` | `aa038972` |
| 启动参数 | [native-eager-state-manifest.json](native-eager-state-manifest.json) | [native-graph-bracket-manifest.json](native-graph-bracket-manifest.json) | [graph-adaptive-manifest.json](graph-adaptive-manifest.json) |

三路共有：Qwen3.5-35B-A3B、BF16、Ascend 910B2、TP4+EP、max model/batched tokens 16384、max seqs 8、prefix cache 关闭、mamba cache mode `none`、同步调度、language-model-only。两种 graph 服务使用 `VLLM_COMPILE + FULL_DECODE_ONLY`，capture sizes `[1,2,4,8]`；原生 eager 使用 `--enforce-eager`。manifest 保留完整 argv，重复选项以服务日志里的最终解析配置为准。未随归档上传的本地日志路径分别为 `local-results/perf-tp4/native-eager-state-server.log`、`native-graph-bracket-server.log`、`graph-adaptive-server.log`。

差异不能忽略：原生使用 native scheduler，日志显示 chunked prefill 开启；插件关闭 chunking，并使用独立 P/D runner、fenced stream、adaptive quota、prefill-first。诊断 prompt 最长 4096，单客户端严格串行，未到 16384 token 预算，但这并不是只更改一行 dispatcher 的受控实验。插件本次加载的是本地冻结目录 `local-results/perf-tp4/graph-adaptive-source/`（各文件 hash 见 manifest），没有后来新增的 `compile_full_prefill` 路径。

诊断脚本为 [scripts/probe_graph_state_reuse.py](../../scripts/probe_graph_state_reuse.py)，producer SHA-256 三路均为 `07fc9dedcd3c56db77e5aca3e28eb4f9c944fcb762902bf81fda239b6b3d8da4`。每组执行 `before × 2 → 4096-token 背景 P + 8-token D → after × 2`；每路六组、24 次目标请求、6 次背景请求。生成参数完全一致：

```json
{
  "model": "Qwen3.5-35B-A3B",
  "temperature": 0,
  "seed": 42,
  "ignore_eos": true,
  "add_special_tokens": false,
  "skip_special_tokens": false,
  "stream": false,
  "return_token_ids": true
}
```

我独立核对了请求 sequence、相邻请求的 monotonic 起止时间和 token/usage 长度：三次客户端执行各自均没有请求重叠，各 30/30 条都返回精确的输入 token IDs、预期数量的输出 token IDs、`finish_reason=length` 与正确 usage。此检查不声称没有其他外部客户端，也不把 `before` 解释为新进程或干净 cache；前面已执行的测试会留下状态。

回执文件 SHA-256：

- 原生 eager：`69a706bb87043e12804bf13006da02ef5c4030d8ff3a8fab53397f496ced01c7`
- 原生 graph：`29650c7a95242f7cd06b7ea712f1faad0b22b5f09f64b84802c233e7e939825a`
- 插件 graph：`b22e9436c58cfcb03afa254c24536d3c60c356330e6231c3e58e0a93fcac598e`

## Token ID 结果

以下下标只计生成 token，`0` 是首个生成 token。`稳定` 表示被比较的完整输出 token 序列相同；每个 before/after 仅两次重复，不能据此声明所有输入都确定。

每格依次为 `before 重复首差异 / after 重复首差异 / before[0] 与 after[0] 首差异`。

| prompt / max_tokens | 原生 eager | 原生 graph | 插件 graph |
| --- | --- | --- | --- |
| `[100]` / 1 | 稳定 / 稳定 / 相同 | 0 / 0 / 0 | 稳定 / 稳定 / 相同 |
| `[100]` / 16 | 稳定 / 稳定 / 相同 | 0 / 0 / 0 | 稳定 / 稳定 / 相同 |
| `[9419]` (`Hello`) / 1 | 稳定 / 稳定 / 相同 | 0 / 0 / 0 | 稳定 / 稳定 / 相同 |
| `[9419]` / 16 | 稳定 / 稳定 / 相同 | 0 / 0 / 0 | 稳定 / 稳定 / 相同 |
| `[9419,12675]` (`HelloHi`) / 1 | 稳定 / 稳定 / 相同 | 稳定 / 稳定 / 相同 | 稳定 / 稳定 / 相同 |
| `[9419,12675]` / 16 | 1 / 1 / 1 | 1 / 15 / 15 | 1 / 稳定 / 2 |

首个生成 token 的四次结果按 `before[0], before[1], after[0], after[1]` 排列：

| 目标组 | 原生 eager 首 token IDs | 原生 graph 首 token IDs | 插件 graph 首 token IDs |
| --- | --- | --- | --- |
| `[100]` / 1 | `8, 8, 8, 8` | `248046, 6797, 38, 29108` | `8, 8, 8, 8` |
| `[100]` / 16 | `8, 8, 8, 8` | `248046, 245065, 38, 211649` | `8, 8, 8, 8` |
| `Hello` / 1 | `11, 11, 11, 11` | `220, 113190, 9419, 99505` | `11, 11, 11, 11` |
| `Hello` / 16 | `11, 11, 11, 11` | `29108, 108317, 9419, 142853` | `11, 11, 11, 11` |
| `HelloHi` / 1 | `0, 0, 0, 0` | `0, 0, 0, 0` | `0, 0, 0, 0` |
| `HelloHi` / 16 | `0, 0, 0, 0` | `0, 0, 0, 0` | `0, 0, 0, 0` |

四个单 token 组中，原生 graph 每组的四条完整序列也各不相同；原生 eager 和插件 graph 每组只有一条独特序列，且两者逐条完全一致。双 token / 16 组三路各有三条独特序列，首 token 都始终为 `0`；原生 eager 与插件该组的四对完整序列逐条均不同。因此双 token 对照只在首 token/max1 上是阴性对照，不能把它整体写成稳定的阴性对照。

## 固定版本源码链条

原生 checkout pin：vLLM-HUST/vllm-ascend-hust `03766ac696fde5ab1980d80ca0b8543d3580c989`；vLLM-HUST/vllm-hust `d0f22d2bda562156e4dbf433ce645e1769b4f804`。以下链接按 pin 固定。

1. [NPUModelRunner:2783](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/worker/model_runner_v1.py#L2783) 计算 `is_all_decode`，但非 speculative 时条件使用 `(is_all_decode if self.speculative_config else True)`，随后只检查每请求 query 长度和总 shape。一个新请求、一个 prompt token 可以满足 uniform decode。`:2805` 的 `force_eager` 则明确返回 `NONE`。
2. [同 runner:2966](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/worker/model_runner_v1.py#L2966) 按 `num_computed_tokens < num_prompt_tokens` 构造 `is_prefilling`。新单 token prompt 为 true，与上面的 shape 分类可能不同。
3. [Ascend GDN builder:457](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn_attn_builder.py#L457) 调用 `split_decodes_and_prefills(..., treat_short_extends_as_decodes=False)`；[vLLM helper:621](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/attention/backends/utils.py#L621) 将 `is_prefilling` 并入 prefill 分类。因此非 speculative 新单 token 请求的正常 metadata 是 P。
4. [GDN capture metadata:513](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/attention/backends/gdn_attn.py#L513) 明确声明 full graph 仅支持 Mamba decode；NPU runner 在 [3055](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/worker/model_runner_v1.py#L3055) 的 capture 阶段调用此入口。实际 replay 在 [ACLGraphWrapper:266](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/compilation/acl_graph.py#L266) 提交既有图，不重新执行 GDN 的 Python prefill 分支。
5. [Ascend GDN builder:704](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn_attn_builder.py#L704) 仅在 `num_prefills == 0` 且非 speculative decode 满足图大小时，向静态 `non_spec_state_indices_tensor` 和 `non_spec_query_start_loc` 写本轮数据。若一个 metadata 为 P 的请求误走 D graph，这个静态更新分支不会发生；这是旧索引仍被图引用的候选路径，尚需实机记录地址和值验证。
6. 正常 P 的初态语义明确：[builder:783](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn_attn_builder.py#L783) 按 context length 生成 `has_initial_state`；[gdn.py:407](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn.py#L407) 读取相关 SSM 行并清零无初态行，运行 chunk P 后写回。清零 kernel 在 [utils.py:88](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/triton/fla/utils.py#L88) 对 `has_initial_state=False` 行真正写零。正常 conv P 也传 initial-state mask（[gdn.py:285](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn.py#L285)）。
7. D 分支不同：conv 在 [gdn.py:306](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn.py#L306) 传 `initial_state_mode_opt=None`；recurrent GDN 在 [434](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn.py#L434) 接受共享 SSM 和 state 索引，没有 P 的 `has_initial_state` 清零分支。
8. 不能假定 allocator 在此配置兜底清零：虽然 [KVCacheConfig:943](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/kv_cache_interface.py#L943) 为 Mamba 请求 zeroing，Ascend worker 只有 [911](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/worker/worker.py#L911) 的 `eagle3 && num_speculative_tokens > 1` 才初始化 `_kv_block_zeroer`。本次无 speculative；[GPUModelRunner:1137](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/worker/gpu_model_runner.py#L1137) 对未初始化 zeroer 的调用不做清零。即使加零，也未必能修复错误 state index，不能把“统一清零”直接当最终修复。

插件这次冻结的 `decode_graph.py` SHA 为 `d3a927053ba2cf97fbffcd7ec805d21aeacee93f7c8dea9d33bef3ad735acfcc`；本地路径 `local-results/perf-tp4/graph-adaptive-source/src/vllm_muxpd/decode_graph.py` 的 370 行强制 P dispatch `force_eager=True`，380–382 行强制 `NONE / skip_compiled=True / capturing=False`。因此一个 token 的 P 也执行正常 P 路径，独立 D wrapper 才持有 graph。这能说明插件规避了上述候选路由，不足以单独证明观察到的稳定性只由这一个差别造成。

## 最小复现及剩余证据

最小目标 payload 为 `/v1/completions` 非 SSE 请求，使用上述 generation 参数，加 `prompt: [9419], max_tokens: 1`。本次原生直接相邻两次便得到不同首 token；完整 bracket 还覆盖 `[100]`、`[9419,12675]`、max1/max16 和长 P 前后。现成完整命令（此调查未执行）为：

```bash
python scripts/probe_graph_state_reuse.py \
  --base-url http://localhost:8000 \
  --output local-results/perf-tp4/native-eager-state-reuse.json \
  --repeats 2 --long-prompt-tokens 4096 --long-max-tokens 8
```

该命令只有在协调启动了原生 eager 服务之后才是 eager 对照；脚本本身不能设置服务图模式。主代理已运行并完成上述 eager 报告，本调查没有再次发请求。

同参数 native eager 的完整 30 条回执已经补齐，并支持“单 token 首步不稳与此 native graph 路径相关”的较窄判断。下一步应做仅修改 phase-aware graph eligibility 的隔离实验，并记录新请求首步的 `computed/prompt tokens`、runtime mode、batch descriptor、GDN P/D 数、动态/静态 state indices 与 `has_initial_state`。对 hybrid GDN，候选防护应按实际 `is_prefilling` 或 metadata 限制 FULL，而不仅仅用 `num_computed_tokens > 0`，后者不能覆盖有上下文的短 prefill。

当前没有记录上述逐请求内部 metadata、cache 内容或 logits margin；没有排除 MoE/BF16 数值、其他 decode 图更新路径或背景状态差异。`temperature=0, seed=42` 也不构成所有算子的逐 bit 确定性保证。双 token 长生成的后段差异需要独立调查，不能随单 token首步问题一并结案。

建议将兼容性 Issue 的主要归属放在 `vLLM-HUST/vllm-ascend-hust`：现象在未选用插件 scheduler/worker 的原生服务中复现，候选 dispatcher 与 GDN static-buffer 更新也位于该仓库。插件仓库可在证据文档中交叉链接，跟踪当前 P 强制 NONE 的防护和仍未解决的双 token 后段现象；不需要先创建两个重复根因 Issue。本次只准备草稿，发布由主代理完成。
