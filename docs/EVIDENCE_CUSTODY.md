# Evidence custody

Current downloads come from benchmark:

- [Goodput](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-tp4-goodput-20261011)
- [Grouped-prefill diagnostics](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-tp4-groupfx-20261012)

The original archive indexes retain producer metadata and historical Release
page identities. These fields are provenance, not current download instructions.
The canonical-custody workflow downloads only from benchmark and verifies every
archive member against those unchanged indexes, without extracting or executing
archived code. Original failures and qualification limits are unchanged.

Implementation packages stay in this repository. Git-embedded historical
measurement files remain migration debt tracked by benchmark issue #326; this
Release migration does not claim that all repository data has been moved.

Mixed design/goodput originals now use the pinned `BENCHMARK_DOCUMENTS.json`
snapshot and the same fail-closed downloader; see [custody entry](../EVIDENCE.md).
This bounded migration does not close the remaining organization-wide debt.
