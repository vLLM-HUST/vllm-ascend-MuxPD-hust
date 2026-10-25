# TP8 争用实验：历史证据入口

[完整历史原文 / Complete original report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b354f203cabb4cfeba72a74e4eefb29dee85776b/archive/mod-targeted-20261023/muxpd-tp8-groupfx/snapshot/docs/performance-tp8-contention.md)

固定队列争用实验观察到 decode 停顿改善，同时存在吞吐和 P TTFT 代价；不是持续到达容量或硬时延保证。

v1/v2、初始/确认及排除的 CPU 重叠阶段不可混池。合成 token 差异、启动中止及 profiler/计时边界均保留在原文。

## 原始数据 / Raw evidence

- [下载已归档原始成员（benchmark collection tar.gz）](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-mod-collections-20261023-v1/muxpd.tar.gz)
- [固定成员清单、路径和 SHA256](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)

可定位的归档成员（并非本地工作区路径）：

- `evidence/2026-10-11-tp8-contention/archive-index.json`
- `evidence/2026-10-11-tp8-contention/contention-audit.json.gz`

## 方法与读取

[当前实验协议](goodput-experiments.md)、[实现与生命周期设计](design.md)及[归档读取说明](../EVIDENCE.md)保留。原文中的绝对路径和运行命令是历史记录，不保证旧环境仍可访问。使用 `BENCHMARK_TP8_DOCUMENTS.json` 下载文档原件，使用 `BENCHMARK_COLLECTION.json` 下载数据成员，分别写入新的外部目录；不要执行归档代码。此迁移没有重新运行硬件实验。
