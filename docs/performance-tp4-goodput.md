# TP4 goodput 历史证据入口

[完整原始报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/a76de7e466e5fabb98b1803a6d8a7037e17cb9a8/archive/mod-targeted-20261023/muxpd-documents/snapshot/docs/performance-tp4-goodput.md)保留所有配置、
数值、失败、TTFT 与 pooled gap 权衡，以及独立串行场景的比较边界。
结论仅适用于原文固定工况，不是当前配置或任意负载的性能承诺。

- [原始测试附件（benchmark tar.gz）](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-tp4-goodput-20261011/tp4-goodput-evidence.tar.gz)
- [逐成员来源与 SHA256](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json)
- [归档说明与校验命令](EVIDENCE_CUSTODY.md)
- [当前实验协议](goodput-experiments.md)

原报告内路径是历史归档相对路径，不是本地文件仍存在的承诺。
使用根目录 `BENCHMARK_DOCUMENTS.json` 下载原文，使用 `BENCHMARK_COLLECTION.json`
恢复原有 collection；两者必须写入不同的新外部目录。不要运行归档脚本。
附件由 canonical-custody CI 按未改写的原始索引逐成员验证。
