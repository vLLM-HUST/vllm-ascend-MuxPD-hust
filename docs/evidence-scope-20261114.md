# 证据导航、兼容与发布范围

本次处理 [#38](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/38)。
[benchmark 固定报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ae8666e4c3178dbece5f7a32ba3712ccec5e4e17/reports/tech-outcomes-20261117/muxpd/README.md)
是本轮实验、计算方法、收益与代价的唯一归档入口；原件不复制到本仓库。
本页只说明使用边界、责任及发布范围。

## 使用和兼容边界

- 默认模式使用 `scripts/serve-qwen35-muxwise.sh`。服务器、插件、模型、设备及工作负载的完整身份与重放条件见
  [中央环境说明](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ae8666e4c3178dbece5f7a32ba3712ccec5e4e17/reports/tech-outcomes-20261117/muxpd/ENVIRONMENT.md)。
  当前源码、其他模型、其他并行度或不同运行时不会自动继承该记录的资格。
- `decode_budget_guard` 是需要显式配置的历史可选能力；其 fenced 路径与当前默认模式分别引用。
  使用说明见 [guard 文档入口](performance-tp8-contention.md)，所有实验与适用边界见该入口指向的中央报告。
- ROW 为默认关闭的可选布局，配置见 [decode attention layout](decode-attention-layout.md)。
  一致性证明与性能证据分开；不能将 ROW 的一致性条件和默认模式的性能结果组合成同一配置的结论。
- Manager 集成与普通源码入口有各自的资格范围；启用、取消、回滚按
  [ECPA 部署说明](ecpa-deployment.md)执行。

## 质量 issue 的当前状态

[#23 的关闭记录](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23#issuecomment-6703514893)
和 [中央固定报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ce52719c023b766910a198b401ba154369f3945d/reports/muxpd-quality-v2-sync-tp4-20261106/REPORT-v4.md)
说明重新定义的 ROW 对 native 单请求一致性条件及其证据范围。
旧门限下的失败仍保留，关闭不意味着默认模式获得逐题或逐 token 等价保证。
有关 token 级、文本级证据的区别和取消复测的范围决定，以中央报告为准。

## 贡献与发布说明

项目负责人仍为毛瀚云（[Irisuko](https://github.com/Irisuko)），指导老师罗瑞坤；责任身份沿用
[#37](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/pull/37)。vLLM-HUST 提供仓库与发布基础设施。
本次证据整理与自动复算不转移 MOD 所有权，也不把自动化协助归为设备实验作者。

本次是文档和中央归档更新，没有修改运行时代码、入口默认值、依赖或包版本，没有发布新的 wheel。
既有安装包和 Manager 集成保持原有资格范围。新增配置的兼容、质量和性能需要独立证据。
