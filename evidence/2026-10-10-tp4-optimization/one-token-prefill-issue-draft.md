# Issue 草稿：TP4 hybrid GDN 单 token prefill 在 FULL_DECODE_ONLY 下首 token 随重复请求变化

**状态：仅供审阅，未发布。原生 eager / 原生 graph / 插件 graph 三路同 payload 对照已各完成 30 条；不应把当前假设写成已确认根因。**

建议主要发布到 `vLLM-HUST/vllm-ascend-hust`：原生服务未启用插件 scheduler/worker 时即可复现，候选路由及 GDN 更新也在该仓库。插件仓库可用证据文档交叉跟踪防护，不必创建重复根因 Issue。

建议标题：`Compatibility: investigate one-token GDN prefill routing under FULL_DECODE_ONLY (TP4, Qwen3.5)`

在固定 HUST vLLM / Ascend 版本上，原生 `FULL_DECODE_ONLY` 服务对同一个单 token prompt 的严格串行 greedy 请求，从第 0 个生成 token 就出现重复差异。原生 `--enforce-eager` 和 MuxPD 的“P 强制 NONE、独立 D graph”路径在本次同 payload 测试中未观察到这一首步现象，两者四组单 token 测试的全部 16 条完整响应 token 序列也逐条相同。源码同时存在 graph shape 分类与 GDN prefill metadata 分类不一致的候选路径，建议跟踪此兼容性问题。

## 环境与输入

- Qwen3.5-35B-A3B，BF16，Ascend 910B2，TP4 + EP，V1 同步，language-model-only。
- HUST vLLM `d0f22d2bda562156e4dbf433ce645e1769b4f804`；HUST Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989`。
- max model/batched tokens 16384，max seqs 8；prefix cache 关闭，mamba mode `none`；无 speculative。
- 两种图服务为 `VLLM_COMPILE + FULL_DECODE_ONLY`，capture sizes `[1,2,4,8]`；原生 eager 使用 `--enforce-eager`。
- 完整服务参数与冻结插件源码由 manifest 记录；原生 chunked-prefill 开启、插件关闭，不把两者称作只改一项配置的 A/B。

最小请求发往 `POST /v1/completions`：

```json
{
  "model": "Qwen3.5-35B-A3B",
  "prompt": [9419],
  "max_tokens": 1,
  "temperature": 0,
  "seed": 42,
  "ignore_eos": true,
  "add_special_tokens": false,
  "skip_special_tokens": false,
  "stream": false,
  "return_token_ids": true
}
```

`9419` 为此模型 tokenizer 的 `Hello`；避免仅凭非法 UTF 字节解码后的文本比较。严格等待每次响应结束后再发下一次。使用仓库 `scripts/probe_graph_state_reuse.py --repeats 2` 可执行完整六组 bracket：一个/两个 token prompt、生成 1/16 token、4096-token 长 P + 8-token D 的前后重复。

## 当前观测

三路各 30/30 请求 transport、token IDs、usage 校验通过；四个单 token prompt 测试组各含 before 两次、长背景一次、after 两次。

| 测试 | 原生 eager 四次首 token IDs | 原生图模式四次首 token IDs | 插件图模式四次首 token IDs |
| --- | --- | --- | --- |
| `[100]`, max1 | `8, 8, 8, 8` | `248046, 6797, 38, 29108` | `8, 8, 8, 8` |
| `[100]`, max16 | `8, 8, 8, 8` | `248046, 245065, 38, 211649` | `8, 8, 8, 8` |
| `[9419]`, max1 | `11, 11, 11, 11` | `220, 113190, 9419, 99505` | `11, 11, 11, 11` |
| `[9419]`, max16 | `11, 11, 11, 11` | `29108, 108317, 9419, 142853` | `11, 11, 11, 11` |

`[9419,12675]` 的首 token 在三路、max1/max16 全部为 `0`。但其 max16 完整序列在三路都出现后续差异：原生 eager 的 before/after 重复及跨历史比较均从第 1 个生成 token 分歧；原生 graph 首差异分别为 1、15、15；插件为 1、after 重复相同、跨历史 2。三路该组各有三条独特序列。这部分不是 graph 专有现象，不应归因于单 token prefill，也未被本插件证明修复。

期望验证的是：新请求初始状态与实际 prefill metadata 一致，不应仅因 query length=1 而重用不适用的 decode 图。这里没有提出 BF16/MoE 在所有情况下都必须逐 bit 确定的普遍要求。

## 候选源码路径，尚待隔离验证

- [NPU runner:2783](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/worker/model_runner_v1.py#L2783)：非 speculative 时 uniform decode 不要求 `is_all_decode`，一个新单 token P 可满足 shape 条件。
- [GDN builder:457](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn_attn_builder.py#L457)：使用 `treat_short_extends_as_decodes=False`，实际 metadata 仍可识别为 P。
- [GDN builder:704](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn_attn_builder.py#L704)：静态 decode state indices/query-start buffers 只在 `num_prefills == 0` 时更新。
- [GDN P:407](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/03766ac696fde5ab1980d80ca0b8543d3580c989/vllm_ascend/ops/gdn.py#L407)：P 会依据 `has_initial_state` 清零初态；D recurrent 路径复用已有 state。原生非 speculative 配置也不初始化分配时 `_kv_block_zeroer`，不能假设自动清零兜底。

尚未逐请求记录 runtime mode、静态/动态 state index 或实际 cache 内容，因而不能断言已经证明跨请求 cache 污染。MuxPD 规避候选路由的结果也不是单变量修复实验。

## 补证与验收

1. 已补原生 eager 同 payload 30 条报告，三路原始回执保持不改写；四组单 token 的 eager/插件完整响应一致。
2. 对原生 dispatcher 增加仅 hybrid GDN 的 phase-aware FULL 准入隔离实验，记录首步 `is_prefilling`、GDN P/D metadata 和静态 state 更新，验证上述候选原因。
3. 验证一个 token 的新 P、正常 D、短剩余 P、长 P 前后与取消后 cache 复用；真实 D 保持 graph replay，P 禁止误入 D graph。
4. 双 token 长生成差异单独追踪，采样日志或 logits 数值证据不足前不关闭其调查。

发布时附：`native-graph-state-reuse.json`、`graph-adaptive-state-reuse.json`、`native-eager-state-reuse.json`、对应 manifests、脚本 producer SHA `07fc9dedcd3c56db77e5aca3e28eb4f9c944fcb762902bf81fda239b6b3d8da4`。完整调查及每个回执 SHA 见同目录 `one-token-prefill-investigation.md`。所有请求均为合成测试数据；本草稿没有发送外部 Issue。
