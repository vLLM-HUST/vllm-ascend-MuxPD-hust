# TP8 验证与性能比较：历史证据入口

[完整历史原文 / Complete original report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b354f203cabb4cfeba72a74e4eefb29dee85776b/archive/mod-targeted-20261023/muxpd-tp8-groupfx/snapshot/docs/performance-tp8.md)

六轮比较没有建立稳定吞吐增益；自然语言有限对照不能消除合成 token 的重复不稳定及跨配置差异。原生前后阶段波动、状态复用和生命周期证据分别保留，不把 host 回执解释为设备并发或 KV 内容归零。

<!-- Preserve the existing incoming environment fragment. -->
<a id="本轮环境与模型来源"></a>

模型 metadata/分片大小核对不等于完整权重 SHA 一致；不同 CPU 配额的 TP4/TP8 不可推导纯扩展效率。完整配置、模型来源和方法见上述原文。

## 原始数据 / Raw evidence

- [下载已归档原始成员（benchmark collection tar.gz）](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-mod-collections-20261023-v1/muxpd.tar.gz)
- [固定成员清单、路径和 SHA256](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)

可定位的归档成员（并非本地工作区路径）：

- `evidence/2026-10-10-tp8/benchmark-summary.json`
- `evidence/2026-10-10-tp8/native-state-reuse.json`
- `evidence/2026-10-10-tp8/work256-state-reuse.json`

## 方法与读取

[当前实验协议](goodput-experiments.md)、[实现与生命周期设计](design.md)及[归档读取说明](../EVIDENCE.md)保留。原文中的绝对路径和运行命令是历史记录，不保证旧环境仍可访问。使用 `BENCHMARK_TP8_DOCUMENTS.json` 下载文档原件，使用 `BENCHMARK_COLLECTION.json` 下载数据成员，分别写入新的外部目录；不要执行归档代码。此迁移没有重新运行硬件实验。
