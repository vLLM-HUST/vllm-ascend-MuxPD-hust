# TP4 grouped-prefill FX: historical evidence

[完整历史原文 / Complete original report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b354f203cabb4cfeba72a74e4eefb29dee85776b/archive/mod-targeted-20261023/muxpd-tp8-groupfx/snapshot/docs/validation-tp4-prefill-group-fx.md)

This historical finite functional diagnostic establishes no additional throughput or Goodput benefit. Natural-language agreement and successful host returns do not establish arbitrary-input token equivalence, kernel completion, or physical overlap. This diagnostic itself left mixed-decode differences unresolved; no unique historical cause is assigned to BF16, fusion, communication, or KV corruption.

The later [follow-up investigation and source98 bridge](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ef849ef49577c491fee68317ffc96d52796cd379/reports/muxpd-followup-20261025)
completed a bounded comparison under an explicitly selected collective profile,
and [Issue #11 was closed within that scope](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/11#issuecomment-6405978497).
The original failures, default-profile limitation and archived bytes remain
unchanged. Later lifecycle/performance observations have their own configuration
and source identities; they are not new results of this historical diagnostic.

## 原始数据 / Raw evidence

- [下载已归档原始成员（benchmark collection tar.gz）](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-mod-collections-20261023-v1/muxpd.tar.gz)
- [固定成员清单、路径和 SHA256](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
- [完整 group-FX 原始测试附件（tar.gz）](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-tp4-groupfx-20261012/tp4-groupfx-20261012.tar.gz)
- [Bounded numerical investigation #11: completed; historical limitation retained](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/11)
- [Optional experimental implementation #10](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/pull/10)

可定位的归档成员（并非本地工作区路径）：

- `evidence/2026-10-12-tp4-group-fx/archive-index.json`
- `evidence/2026-10-12-tp4-group-fx/numeric19-independent-audit.json`

## 方法与读取

[当前实验协议](goodput-experiments.md)、[实现与生命周期设计](design.md)及[归档读取说明](../EVIDENCE.md)保留。原文中的绝对路径和运行命令是历史记录，不保证旧环境仍可访问。使用 `BENCHMARK_TP8_DOCUMENTS.json` 下载文档原件，使用 `BENCHMARK_COLLECTION.json` 下载数据成员，分别写入新的外部目录；不要执行归档代码。此迁移没有重新运行硬件实验。
