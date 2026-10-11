# TP4 Goodput confirmation evidence — 2026-10-11

The paired confirmation uses two independent TP4+EP4 services on eight Ascend
910B2 cards, then exchanges their card groups. Three frozen arrival seeds per
orientation give 378 measured requests per role across 720 measured seconds.
The same seed1001 is additionally run serially on group A, with a fresh service
for each role; its 71 measured requests per role are reported separately.

The primary SLO requires a complete response, scheduled TTFT ≤ 1 s and every
observable single-token gap ≤ 100 ms. All four 100/200 ms × 1/2 s thresholds,
failures, dispatch checks, throughput and latency costs are retained.

See the [Chinese performance report](../../docs/performance-tp4-goodput.md) and
`summary.json` for results. Percentiles are reported per window, not averaged.
Shared service startup within each orientation does not provide six independent
startup replicates. Fixed-load output throughput is not sustainable capacity.

## Assets and verification

Large raw evidence is distributed as a
[GitHub release asset](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/releases/tag/evidence-tp4-goodput-20261011).
`archive-index.json` binds every original relative path, size and SHA256, as well
as the compressed asset. The archive includes raw SSE/chunk journals, producer
reports, independent audits, plans, protocols, exact measurement/controller
sources, service manifests and lifecycle receipts, CPU snapshots, natural
checks and all pre-generation readiness failures/amendments. Model weights,
compiled caches and ownership locks are excluded. Source bytes are unchanged.

The verified archive contains **763 files / 1,093,786,647 uncompressed bytes**.
The gzip asset is **94,280,420 bytes**, SHA256
`58e97810b71bfb32c4717c1e590695f6ca6cebb09e1922577e6758c18ce41502`.
All 763 decompressed members passed exact byte verification.

```bash
gh release download evidence-tp4-goodput-20261011 \
  --repo vLLM-HUST/vllm-ascend-MuxPD-hust \
  --pattern 'tp4-goodput-evidence.tar.gz' --dir /tmp/tp4-release
cp evidence/2026-10-11-tp4-goodput/archive-index.json /tmp/tp4-release/
python evidence/2026-10-11-tp4-goodput/archive_tp4_evidence.py \
  --verify-only /tmp/tp4-release
```

Verification streams every decompressed member and checks its exact SHA and
size without writing into the source tree. It does not rerun the raw auditor.
The independent auditor had already passed each recorded measurement before
aggregation; archive verification establishes byte preservation.

## Replay one raw audit

Use Python 3.10+; the auditor uses only the standard library. Extract into a new directory. Recorded paths in
original JSON remain absolute; supply explicit relocated inputs to the auditor:

```bash
mkdir -p /tmp/tp4-replay
tar -xzf /tmp/tp4-release/tp4-goodput-evidence.tar.gz -C /tmp/tp4-replay
cd /tmp/tp4-replay
python local-results/tp8-goodput/audit_goodput.py \
  --report local-results/tp4-goodput/tp4-confirm-first-result/seed-1001/native-goodput.json \
  --journal local-results/tp4-goodput/tp4-confirm-first-result/seed-1001/native-goodput.events.jsonl \
  --plan local-results/tp4-goodput/confirm-plan-seed1001-rate050.json \
  --producer local-results/tp4-goodput/benchmark_goodput_paired.py \
  --sse-parser local-results/tp4-goodput/benchmark_contention.py \
  --output /tmp/tp4-native1001-replay.json
```

Serial audits instead use `scripts/benchmark_goodput.py` and
`scripts/benchmark_contention.py` from the archive. No new requests or hardware
are needed for audit replay. Regenerating measurements requires the pinned
runtime/model and the explicit service configuration in the manifests.

The example audit was rerun successfully after extracting its exact inputs into
a new directory. `portable-replay.json` records that post-archive check; it is
separate from the archived original measurements.

`protocol-seed-wording-erratum.json` records a prose correction: the immutable
protocol load description says seed42, but actual frozen payloads use the plan
seed (pilot101; confirmation1001/1002/1003). Only priming uses42. No plan,
payload, SLO or recorded result was changed.

Local project validation passed: **657 tests passed, 2 skipped**, Ruff, shell
syntax and diff whitespace checks. `local-validation.json` records the checks;
runtime source is unchanged from the measured plugin commit `5342e8cd`.
