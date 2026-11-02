# TP8 持续到达探索：历史证据入口

[完整历史原文 / Complete original report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b354f203cabb4cfeba72a74e4eefb29dee85776b/archive/mod-targeted-20261023/muxpd-tp8-groupfx/snapshot/docs/performance-tp8-goodput-exploration.md)

顺序单 seed 探索不等于独立确认。原文保留高负载积压、排空失败、间隔不可观测、取消覆盖失败及补充诊断；未建立相对调优原生的持续吞吐收益，有限功能通过不证明数值等价或性能改善。

## 原始数据 / Raw evidence

- [下载已归档原始成员（benchmark collection tar.gz）](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-mod-collections-20261023-v1/muxpd.tar.gz)
- [固定成员清单、路径和 SHA256](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)

可定位的归档成员（并非本地工作区路径）：

- `evidence/2026-10-11-matched-numeric/review.json`
- `evidence/2026-10-11-matched-numeric/review.md`

## 尚未确认的原件

原文 `local-results/tp8-goodput` 的完整 raw 归档及末尾列出的摘要原件尚未确认恢复。上面的 matched-numeric 成员只覆盖独立数值对照，不能替代持续到达六档的原始 SSE、计时和 profiler 材料。历史表格与 SHA 原样保存在完整原文中，本次迁移不把这些缺项标为已找回，也不启动新 TP8 补测。

## 方法与读取

[当前实验协议](goodput-experiments.md)、[实现与生命周期设计](design.md)及[归档读取说明](../EVIDENCE.md)保留。原文中的绝对路径和运行命令是历史记录，不保证旧环境仍可访问。使用 `BENCHMARK_TP8_DOCUMENTS.json` 下载文档原件，使用 `BENCHMARK_COLLECTION.json` 下载数据成员，分别写入新的外部目录；不要执行归档代码。此迁移没有重新运行硬件实验。
