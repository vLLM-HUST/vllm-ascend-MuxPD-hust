# TP8 evidence, 2026-10-10

This archive contains the completed eight-device Ascend 910B2 experiment using
Qwen3.5-35B-A3B, TP8+EP and the existing work256 implementation. Every file listed
in [archive-index.json](archive-index.json) is a byte-identical copy of its
original under `local-results/tp8`; the index records its source, size and SHA256.
This README and the index are archive-generated navigation and metadata.
Raw logs retain their original bytes, including carriage returns and whitespace;
whitespace checks apply to authored files and exclude these immutable logs.

Runtime source was unchanged. This experiment did not rerun the earlier TP4
451-pass test suite and did not collect a device profiler. CPU logs, graph
capture messages, client SSE events and scheduler receipts do not by themselves
prove device graph replay or hardware overlap.

## Results and scope

- [Benchmark summary](benchmark-summary.json) includes exactly eight performance
  JSON files: native initial/bracket and work256 stage1/confirmation, each with
  mixed/decode-heavy workloads and three repetitions. No repetitions were
  removed. Each side has six runs, 48 requests and 3072 output tokens per workload.
  Throughput is total output tokens divided by total wall time; P95 uses nearest
  rank over all request values, not the mean of per-run P95 values.
- All-six throughput is native/work256 **177.302/177.314 tokens/s** for mixed
  and **243.153/244.002** for decode-heavy. The differences, +0.0068% and +0.3492%,
  do not establish a stable improvement. Both candidate totals are below the
  native bracket stage. All stages, raw repetitions and text-hash instability
  remain visible in the summary and original inputs.
- [Correctness summary](correctness-summary.md) and its
  [machine-readable audit](correctness-summary.json) separate natural-language
  equality, synthetic transport, repeated-text stability and generated-token
  state-reuse probes. Native eager is a diagnostic control, not a graph
  performance sample. Exact matches on these cases do not guarantee equivalence
  for arbitrary prompts.
- [Lifecycle proof](lifecycle-proof.json) dynamically locates this run's long-P
  probes and every P-first admission. The successful 12000-token P was admitted
  once and completed in 21+19 layers; a separate cancelled P reached 21 layers,
  acknowledged cancellation without further layer progress and cleared its
  cohort on the following step. Four P-first admissions repaid their recorded
  decode debt on the following step. These values were extracted from this run,
  not assumed from TP4.
- The [complete scheduler trace](work256-traced-86528.jsonl) is included, with
  [strict-check results](traced-check.json), [client progress](traced-progress.json)
  and [client cancellation](traced-cancel.json). The full trace is 876277 bytes,
  SHA256 `935052d0ab64e8a53b9021706c6d2abbc53534f0846f2e5109ccd11636967740`.
  Host layer receipts and ownership ACKs do not inspect actual KV tensor contents.
- [Pure-P calibration](traced-calibration.json) precedes the trace-on diagnostic
  and is excluded from performance aggregates. The observed quotas followed
  calibration; they are not a cold-start guarantee.

## Provenance and shutdown

[Environment](environment.json), [initial snapshot](environment-initial.json)
and [model resolution](model-resolution.json) retain hardware, package, source
and model metadata. Collection-time pending fields in those historical snapshots
are not final-result placeholders. Final results are in the completed reports.

Service arguments and successful client receipts are preserved in the
[native initial manifest](native-initial-manifest.json),
[native bracket manifest](native-bracket-manifest.json),
[work256 manifest](work256-manifest.json),
[native eager manifest](native-eager-manifest.json) and
[trace-on manifest](work256-traced-manifest.json).
Launch scripts, server/client logs and all stage shutdown receipts are included.
[The failed model-path startup log](failed-missing-model-server.log) preserves
the initial unsuccessful attempt and contributes no performance samples.

[Final runtime validation](final-runtime-validation.json) records unchanged
runtime-source hashes and no remaining experiment-owned processes. The
[final NPU snapshot](npu-smi-final.txt) reports no running processes on all eight
devices. [The final traced-service shutdown](work256-traced-stop.json) records
process ownership and termination. This proves process absence, not graceful
per-worker allocator cleanup. PID files, temporary orchestration drivers,
`stop_service.py`, release-tool metadata and Issue/PR drafts are excluded.

## Offline verification

From the repository root, verify every archived source artifact:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('evidence/2026-10-10-tp8')
index = json.loads((root / 'archive-index.json').read_text())
for item in index['files']:
    data = (root / item['file']).read_bytes()
    assert len(data) == item['bytes'], item['file']
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['file']
print(f"Verified {len(index['files'])} source artifacts")
PY
```

Recompute all three audits in a temporary copy so the immutable original outputs
are preserved. The copy must stay two directory levels below the repository root
because the correctness/lifecycle audits locate the repository's `scripts/`
helpers there. Use the workspace Python environment with `httpx`; no service or
NPU is accessed. The correctness audit verifies helper hashes against the
experiment's recorded git commit, which must be available locally.

```bash
tp8_replay_dir="$(mktemp -d local-results/tp8-replay-XXXXXX)"
cp -a evidence/2026-10-10-tp8/. "$tp8_replay_dir/"
python "$tp8_replay_dir/summarize_tp8.py"
python "$tp8_replay_dir/audit_lifecycle.py"
python "$tp8_replay_dir/correctness-summary-audit.py"
```

[Offline replay validation](archive-replay-validation.json) records a completed
run of all three audits on an archive copy. Lifecycle output was identical;
benchmark output differed only in its generation timestamp; correctness JSON
differed only in relocated input/script paths and the correctness Markdown was
byte-identical. Those metadata differences were normalized only for semantic
comparison; the archived source files were not rewritten.
