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


## Additional historical directories

[Canonical history manifest](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/afd47a8e9964b71844dec75a8e908f279ae98202/archive/mod-history-20261023/muxpd/manifest.json) lists original paths, sizes and SHA256.
This is a separate capsule from BENCHMARK_EVIDENCE.json; do not reuse its destination.
No hardware experiment or new performance acceptance is implied.

```sh
python scripts/fetch_mod_consumer_evidence.py --config BENCHMARK_HISTORY.json \
  --destination /absolute/new/external/history
```

The restored root preserves these original paths:

- `evidence/2026-10-10-tp4-optimization/`
- `docs/performance-tp4.md`

Frozen reports retain original labels and local paths; these do not assert that omitted captures were recovered.


## Remaining collections (formal / handoff / TP4-TP8 / profiler)

[Member manifest](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) preserves the source commit and every original hash.

```sh
python scripts/fetch_mod_consumer_evidence.py --config BENCHMARK_COLLECTION.json \
  --destination /absolute/new/external/collection
export BENCHMARK_COLLECTION_ROOT=/absolute/new/external/collection
```

This root is separate from BENCHMARK_EVIDENCE_ROOT and BENCHMARK_HISTORY_ROOT.
Do not execute archived scripts; use the maintained repository validators.
All paths below are archive-relative; generation commands may still produce ignored local outputs.

- `evidence/`

Current forwarding documents, configuration and imported code remain; exact retained paths are tested in tests/test_collection_evidence_boundary.py.
