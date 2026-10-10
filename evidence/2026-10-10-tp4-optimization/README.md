# TP4 optimization evidence, 2026-10-10

This archive contains completed receipts through graph-compiled, isolated8 and
the second native graph baseline stage.
The evolving interpretation is in [the performance report](../../docs/performance-tp4.md).
It does not mark the optimization goal complete: the measured graph candidates still
trail the matched native graph throughput in these measurements, and synthetic
output instability remains visible.

[archive-index.json](archive-index.json) lists the original paths, sizes and
SHA256 hashes of all copied artifacts. Copies are byte-identical; source paths
inside receipts preserve where they were originally produced. Runtime manifests
identify the loaded source hashes, so later changes do not retroactively apply
to these measurements.

Included evidence:

- Three-repeat native/P-first eager and native/graph8/graph-adaptive graph
  benchmark receipts, plus effective-launch manifests.
- The graph-compiled candidate combines trace-on, optional full-P compilation
  attempts and the D-only worker fast path. It is slower than both native graph
  baseline stages. Natural-language cases are exact, but synthetic text matches
  are 36/56; transport success is not full output equivalence. Strict progress
  sources verify full-return versus layer-hook receipts, not actual FX hits.
- Both native graph baseline stages (initial and bracket) remain separate.
  A six-run summary uses total tokens/total wall time and nearest-rank request
  percentiles across all observations, retaining stage labels and source hashes.
  No weaker baseline is selected to inflate a candidate's relative result.
- The native graph long-P probe records one decode content event before the P
  first text, versus four in graph8's probe. Different window lengths and SSE
  aggregation prevent treating these counts as a standalone performance score.
- Isolated8 eager benchmarks, fixed natural-language correctness, long-P
  progress, cancellation and compact layer-receipt checks. This frozen variant
  excludes the subsequent D-only fast path; its throughput and TTFT do not beat
  the native eager benchmark here.
- Four-rank isolated/shared device-profile audit with matching nine forward
  equivalents per rank. Python event record/wait counts fall 41.9%/39.9%, while
  measured model-compute overlap does not increase. Stream mapping and exact
  trace/CSV interval cross-checks are retained; this is not a single-variable
  communicator ablation or evidence of hardware resource partitioning.
- Completed natural-language correctness and synthetic streaming reports.
  Graph8 synthetic transport succeeds for all 56 cases per endpoint, while only
  37/56 cross-endpoint decoded texts match; this is not full output equivalence.
- The complete 30-request graph-adaptive state-reuse diagnostic. The two-token,
  max-output-16 case is unstable; the follow-up native diagnostic remains pending.
- Long-prefill progress, cancellation, compact layer-progress checks, independent
  profile audits, and source-comparison/design reports.
- Exact CANN `aclmdlRIExecuteAsync` records: seven per rank in the bounded graph8
  profile, plus four startup capture entries per rank. These are runtime API
  records, not graph kernel duration or per-core occupancy measurements.

The interrupted `graph8-state-reuse.json` is deliberately not copied. The API
was stopped before it completed (41 receipts out of 54 expected); its exclusion
reason is retained in the index. Do not treat partial coverage as a completed
correctness or performance result. Isolated8's completed functional receipts
do not alone prove device-kernel overlap or a communication-speed improvement.

Full profiler exports, kernel CSVs, scheduler JSONL traces, server logs, and
frozen source directories are not included. Compact profile reports retain
source hashes and selected events, and the graph8 proof embeds the relevant
startup log lines. Recomputing all profile metrics requires the full local
source captures.

The audit Python files are exact copies of the scripts used for the local
analysis. Their recorded commands expect the original
`local-results/perf-tp4/` script and input layout. Restore that layout, including
the omitted full traces, before rerunning those commands. Running a copied
script directly from this archive is not a substitute for providing its inputs.
