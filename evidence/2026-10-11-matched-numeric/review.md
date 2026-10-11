# 三配置匹配 21 请求：独立数值审计

通过仅表示原始传输、请求体、来源和保留历史一致，不表示数值等价或因果结论。所有首次分歧位置从 0 开始。

审计状态：`passed_evidence_consistency`。

| 配置 | 完整传输／Added | 初始 P128 两次全序列一致 | P511 后四次一致 | 六次不同序列数 |
|---|---:|---|---|---:|
| native16k_cap512 | 21 / 21 | True | False | 2 |
| fixed4_raw | 21 / 21 | True | False | 2 |
| short_piecewise | 21 / 21 | False | False | 2 |

跨配置按相同序号、相同请求体（仅 ID 不同）逐条对齐：

- `native16k_cap512` → `fixed4_raw`：11/21 条完整输出 token 序列相同；这是描述统计，不是精度评分。
- `native16k_cap512` → `short_piecewise`：11/21 条完整输出 token 序列相同；这是描述统计，不是精度评分。
- `fixed4_raw` → `short_piecewise`：18/21 条完整输出 token 序列相同；这是描述统计，不是精度评分。

各次短输出 token IDs、全部 15 对 P128 首次分歧、初始／后四次分组、原 126 初始对照及逐请求跨配置分歧均保留在 JSON；不复制大 prompt 数组。

解释边界：

- Audit passed means complete, internally consistent transport and retained request history, not output equivalence.
- Initial P128 pair and post-P511 repetitions have different histories; keep them separate.
- Three preserved configuration bundles differ in graph mode, asynchronous scheduling, communication and runtime implementation; no single-mechanism causal inference.
- No numerical-error, cache-contamination, semantic-quality or performance conclusion follows from token equality/divergence alone.
- Added/pause history does not prove identical internal model batches. These are finite sequential, nonrandomized diagnostics.

离线审计源码 SHA：`1b4e85ab0d8876385a94540ad30d6af7182ecd792600339f2794e5315d5564f3`。原报告、manifest、客户端源码、raw SSE、日志和 journal 的 SHA 均在 JSON。
