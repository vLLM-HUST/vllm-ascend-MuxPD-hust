# Canonical historical data

This migration batch is stored only in [vllm-hust-benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/2144d58de108cddaff3800fdcbf12dc9d3ab1d18/archive/mod-consumers-20261023/muxpd).
`BENCHMARK_EVIDENCE.json` pins the manifest and its SHA256. The manifest preserves original
repository/commit/path identities, failures and every file hash. This is custody, not a new result.

```bash
python scripts/fetch_mod_consumer_evidence.py --destination /absolute/new/external/evidence
export BENCHMARK_EVIDENCE_ROOT=/absolute/new/external/evidence
```

The destination must be new and outside this checkout. The downloader verifies the archive and
every member before writing, rejects links/unsafe paths, and never falls back to local copies.
Use `$BENCHMARK_EVIDENCE_ROOT/evidence/2026-10-10/` for this batch's original data.
Existing generation commands may still write new local outputs; do not commit them here.

Other historical directories are not declared migrated by this batch. Original failures and
diagnostic/NO_GO scopes remain unchanged. No hardware experiment is required to read this archive.
