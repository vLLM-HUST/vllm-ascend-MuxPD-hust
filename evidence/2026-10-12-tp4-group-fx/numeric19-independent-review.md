# TP4 group-FX：固定 19 请求的独立数值复算

[独立审计 JSON](numeric19-independent-audit.json) 的 SHA256 为 `0eb97183288e3255f015005dc4ae45e56d0b44cdb526d677dd41814f5cfdaa37`。raw-a 与 candidate-b 各 19/19 完整，均通过原始 SSE、journal、计划和阶段顺序复算；这表示证据和传输合同完整，不表示数值等价。

[审计器](audit_groupfx_numeric.py) SHA256 为 `e9beb0536e7178b753adf7e3618188488d05739eebb6706916aefbbe20cb8a50`。它没有导入 producer 或 SSE parser：按原始字节找到 LF/CRLF 帧界，将帧末字节映射到累计 HTTP chunk 的接收时间，独立解析 UTF-8/JSON、prompt/token IDs、usage、finish、DONE。核对每个请求的原始字节与 chunk 时间、固定 payload、响应 ID、19 次唯一开始/结束、11 个依次排空的阶段和两次四路 D 首 token 门槛。两份 journal 各 1,233 条事件，重算结果与 producer 的数值摘要一致。

跨角色完整 token 序列 **9/19 相同**。全部五条 12K P（单独三次、混合两次）在两个角色中一致。P128 首三次和末三次在两个角色各自都出现两种序列，首次差异为零基索引 4 的 `248068` / `74455`，因此不能用某一条重复作为唯一“正确输出”。

需要单列的混合 D 差异如下；这些观察不能因为已有 P128 重复差异就被豁免。

| D lane | 跨角色首差异（两轮相同） | raw-a 自身两轮首差异 | candidate-b 自身两轮首差异 |
|---|---|---|---|
| 0 | 索引 3：223 → 224 | 索引 74 | 索引 114 |
| 1 | 索引 65：31838 → 10597 | 两轮完全一致 | 索引 104 |
| 2 | 两轮跨角色均完全一致 | 两轮完全一致 | 两轮完全一致 |
| 3 | 第一轮一致；第二轮索引 26：7309 → 16421 | 索引 26 | 索引 80 |

lane 0 的早期索引 3 和 lane 1 的索引 65 差异在两轮跨角色比较中重复出现，早于本轮观测到的相应角色内部重复差异。应保留为候选相关的新疑点；两轮、单一服务历史/卡组下的结果不能区分配置、执行形态、卡组或其他来源，也不足以认定缓存污染或数值误差。首次分歧之后，两条自回归历史已经不同，后续 token 差异不再是同一条件输入下的比较。

证据边界：outbound `X-Request-Id` 没有单独捕获为 wire bytes，其与 body ID 相等仅由冻结 producer 源码支持；body ID 与 raw SSE 的 `cmpl-…` ID 则独立核对。四路 D 门槛证明客户端流仍活跃，不证明实际模型 batch、物理重叠或服务中没有其他客户端。运行配置与 group-FX 实际命中需要另外的 manifest/生命周期证据。这里没有作性能或语义正确性结论。

[CPU 验证回执](numeric-auditor-selftest.json) 记录 9 个 fixtures 通过（9.59 秒）、AST/Ruff/格式检查通过和实际离线复算 exit 0。初次 fixture namespace 导入失败日志保留；修复仅限测试夹具。未进行 HTTP/NPU 操作。
