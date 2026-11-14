# 成果补证、兼容与发布范围（2026-11-14）

本次处理 [#38](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/38)。
[benchmark 固定报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ae8666e4c3178dbece5f7a32ba3712ccec5e4e17/reports/tech-outcomes-20261117/muxpd/README.md)
保留本地 wl38 的全部六种负载、TP2/TP4 两臂原始结果、请求身份、冻结筛选规则、SLO 计算和复核脚本，
并重算 TP8 固定归档。本仓库维护实现、使用方法与版本边界；实测原件统一放在 benchmark 仓库。

## 证据与运行配置的对应关系

| 对象 | 运行版本与配置 | 可引用范围 |
| --- | --- | --- |
| wl38 默认模式 | MuxPD `43fcaf4279640bcc086119b0671cb7fd4228836a`；Core `d0f22d2bda562156e4dbf433ce645e1769b4f804`；Ascend `c182a9ed1ed8ca773512d6549e15de6fd471aa00`；Qwen3.5-35B-A3B BF16；TP2/TP4+EP；`serve-qwen35-muxwise.sh`，16 序列、16K 上下文、FULL_DECODE_ONLY 图、默认 TND 布局 | 同卡、同请求、每配置单轮；吞吐和 SLO 收益、首 token/普通间隔/最坏停顿代价一并引用。到达和固定输出长度属于派生设置，没有答案质量评测。 |
| 历史 TP8 guard v2 | MuxPD `258aef6ce52b7fd77cfe9a7f70d1ea29a13f7d41`；Core 同上；Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989`；8×910B2；TP8+EP8；同步 fenced guard 配方 | 三种合成争用负载、每种六轮。D 最大出词事件间隔与 P TTFT、吞吐取舍见固定报告。不是 wl38 默认模式的 TP8 验证，不代表硬 SLO 或完整 MuxWise 空间分区。 |
| ROW 一致性 | `43fcaf4` 的显式 ROW + 单 prompt prefill；与同配置 native 单请求相比 | TP4 200/200 完整 token、文本、结束原因一致；TP2 200/200 文本、结束原因一致，未保存 token 数组。ROW 默认关闭，其性能未测，不能借用 wl38 默认模式的收益。 |

当前机器只读盘点及历史运行身份的区别、模型/tokenizer 历史摘要缺项、完整命令与数据源版本，
见 benchmark 的 [环境与溯源说明](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ae8666e4c3178dbece5f7a32ba3712ccec5e4e17/reports/tech-outcomes-20261117/muxpd/ENVIRONMENT.md)。
当前缓存版本不补写成过去运行的模型哈希，当前 main 也不继承旧版本的测量资格。

## 质量 issue 的当前状态

[#23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23#issuecomment-6703514893)
已经依据重新定义的 ROW 对 native 单请求一致性条件关闭，
[固定报告 v4](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ce52719c023b766910a198b401ba154369f3945d/reports/muxpd-quality-v2-sync-tp4-20261106/REPORT-v4.md)
记录 TP4 token 级验证，以及维护者接受 TP2 既有文本级证据、取消 TP2 token 复测的范围决定。

旧门限下的失败仍成立：TP4 同步默认 raw/group 的 QUALITY-V2 系统性丢失，以及 ROW 相对 batched native 的旧 FAIL 均保留。
默认模式并未获得逐题或逐 token 等价保证。此次性能归档不能证明旧失败已修复，也不能组合成 ROW 同时具有默认模式性能收益的结论。

## 贡献、兼容与发布说明

- 项目负责人仍为毛瀚云（[Irisuko](https://github.com/Irisuko)），指导老师罗瑞坤；责任身份沿用已合入的 [#37](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/pull/37)。
  vLLM-HUST 提供仓库与发布基础设施。本次证据整理与自动复算不转移 MOD 所有权，也不把自动化协助归为设备实验作者。
- 兼容声明只覆盖上述实测的运行时与配置。TP2/TP4 默认模式、历史 TP8 guard、ROW、旧 Manager 集成有各自的证据范围；需要切换入口时按对应说明显式配置。
- 本次是文档和实验归档更新，没有修改运行时代码、入口默认值、依赖或包版本，没有发布新的 wheel。
  既有 `0.2.0a1` 的 [ECPA 部署与安装资格](ecpa-deployment.md)保持其原有范围；源码默认模式的性能数字不授予旧安装包、Manager 或其他运行时新的性能资格。
- 后续使用其他模型、TP8 默认模式、新版本或 ROW 性能，需要独立运行身份与测量；本次未开展这些实验。
