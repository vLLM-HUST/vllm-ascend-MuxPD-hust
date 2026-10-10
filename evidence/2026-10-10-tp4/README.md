# Ascend TP4 validation — 2026-10-10

Follow-up to merged [PR #2](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/pull/2).
The serving implementation is the merged `784236edf7bc99ff71d8df7b9b93168164a0f72a` tree.
This experiment uses all four Ascend 910B2 cards (physical 4/5/6/7),
Qwen3.5-35B-A3B BF16, TP4 + EP4, eager V1 execution and synchronous scheduling.
Source pins, packages, model metadata and configuration are in `environment.json`.

All measurements run sequentially on the same four cards. The existing TP2
services are stopped for the experiment. The native control and plugin use
16,384-token context and batch budgets, max 8 sequences, disabled prefix cache,
Mamba `none` and memory utilization 0.85. Native scheduling retains its mixed
P/D and chunked-prefill capability. MuxPD admits full prompts to separate P/D
batches. No source changes to vLLM or vLLM Ascend are made for TP4.

## Serving results

| Configuration | Output tokens/s, pooled | Individual runs (tokens/s) | TTFT p95 (s) | Mean TPOT p95 (s) |
|---|---:|---|---:|---:|
| Native matched control | 32.10 | 31.63 / 32.58 | 0.710 | 0.240 |
| Decoupled overlap, fixed 8 layers | 21.18 | 24.85 / 18.45 | 3.836 | 0.374 |
| Decoupled adaptive fenced, minimum 8 layers | 32.29 | 32.15 / 32.44 | 1.182 | 0.240 |

Adaptive fenced is close to native throughput in this sample (about +0.6%),
with worse TTFT. This is not evidence of a statistically significant speedup.
Fixed overlap is slower and its two runs vary substantially; the receipts show
complete responses and increased decode intervals in the slower run, but do not
establish the cause. All three configurations have zero request goodput at
TTFT <=2 s and mean TPOT <=100 ms. TP2 and TP4 were measured separately;
these short runs do not establish a scaling benefit.

`benchmark-summary.json` preserves the pooling formulas, per-run values,
per-request timing checks, output hashes and input-file SHA256 values. Native
and fixed-overlap synthetic benchmark outputs are stable for only 5/8 and 4/8
request indices, respectively, across their two repeats; adaptive fenced is
stable for 6/8. These synthetic token
prompts are separate from the natural-language correctness cases. Do not treat
any single synthetic native output as an invariant golden result or assume
that the observed differences have a known cause.

The fixed-overlap host trace passes strict full-prompt, 40-layer progression,
cohort ownership and cancellation checks. It records four decode steps before
the 12,000-token prefill completes, and one cancelled prefill retired after
worker acknowledgement. Its client probe independently observes four decode
content events before P's first token. The adaptive long-P probe observes zero
such events: its deadline policy completes P in one iteration in that case.
The adaptive probe therefore does not demonstrate cross-iteration progress.
Its cancellation client also passes, but the adaptive trace has zero deferred
P cancellation acknowledgements: the cancelled request's P is recorded as
completing all 40 layers in one worker iteration. The fixed-overlap experiment
separately verifies deferred cancellation.

All three configurations pass 9/9 API smoke cases. Native, fixed overlap and
adaptive fenced each pass 16/16 natural-language cases, for 48 actual requests.
Both plugin modes exactly match the same TP4 native baseline on all 16 outputs,
and each endpoint is stable between repetitions and isolated/paired execution.
Cases cover counting, arithmetic, distinct identifier extraction and a
2735-token Chinese extraction prompt. These finite checks do not establish
equivalence for arbitrary prompts or sampling settings. The baseline-only
capture mode added to `scripts/compare_outputs.py` makes sequential four-card
comparison possible without keeping two TP4 endpoints resident simultaneously.

The full local regression suite passes 158 tests with one mode-specific skip.
Ruff, shell syntax and diff checks pass. `validation.json` records the checks,
startup configuration excerpts and receipt hashes.

Timed plugin runs include scheduler trace recording. The overlap profiler is
configured but inactive throughout its benchmark; device capture occurs only
after correctness and benchmark clients finish. Startup logs report that NUMA
CPU binding was skipped. A container snapshot records 192 visible CPUs with a
16-CPU-equivalent quota; this is environment evidence, not a diagnosis of the
performance variation. Baseline startup/source information is preserved in
`environment.json`; [the static TP4 audit](tp4-source-audit.md) documents the
communication, head replication and expert-partition paths checked beforehand.

## Device overlap

All four ranks show P/D compute interval intersections in the separate 8-step
profile. The overlap divided by compute-active union is 4.072%, 4.492%, 2.858%
and 4.258% for ranks 0–3. Each rank's 16,086 selected compute records match its
kernel CSV exactly; an independent interval calculation matches the analyzer.
Communication and CPU ranges are excluded. These measurements do not establish
per-core occupancy, hardware resource partitioning, production utilization or
serving speedup. The RECORD-state warning means capture completeness is not
guaranteed. See [the device-profile audit](overlap-profile-summary.md) for exact
denominators, capture boundaries, source hashes, excerpts and reproduction.

## Receipt index

| Files | Purpose |
|---|---|
| `native.json`, `overlap.json`, `adaptive-fenced.json` | Raw benchmark SSE timings, usage and output hashes |
| `benchmark-summary.json` | Verified metrics, per-run variation, formulas and file hashes |
| `*-smoke.json` | Nine API checks per configuration |
| `native-correctness.json`, `native-vs-*.json` | Native capture and two 16-case comparisons |
| `*-progress.json`, `*-cancel.json` | Long-prefill and HTTP cancellation client receipts |
| `*-trace.jsonl.gz`, `*-trace-check.json` | Full scheduler traces and strict 40-layer lifecycle checks |
| `overlap-profile-*` | Four-rank device evidence and independent audit |
| `environment.json`, `cpu-snapshot.json`, `validation.json` | Source/model/runtime pins, constraints and test checks |
| `restored-services.json` | Post-test TP2 health and actual generation checks |

The tested serving implementation remains the merged PR #2 code. This follow-up
adds a configurable TP launcher, baseline-only capture and TP4 evidence; it does
not change scheduling or worker execution. The test sequence was native smoke,
correctness, benchmark; then each plugin's smoke, progress, cancellation,
benchmark and correctness. The overlap-only profiler ran last in its mode.
After measurement, the original TP2 service on port 8000 and adaptive-fenced
MuxPD TP2 service on port 8002 were restored. Both passed health and arithmetic
generation checks; those restoration requests are excluded from TP4 results.

## Reproduction

Run one server at a time; stop its workers before starting the next configuration.
The commands below assume this repository is installed editable in the existing
environment and the four selected devices are free.

```bash
mkdir -p local-results/tp4

# Native matched control.
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_SERVER_PORT=8000 \
  /root/workspace/runtime/serve-qwen35.sh \
  --tensor-parallel-size 4 --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching --mamba-cache-mode none --no-async-scheduling

# Fixed-quota decoupled overlap (after stopping the control).
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 MUXPD_TP_SIZE=4 \
  VLLM_SERVER_PORT=8002 VLLM_USE_V2_MODEL_RUNNER=0 \
  scripts/serve-qwen35.sh \
  --scheduler-cls vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler \
  --worker-cls vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker \
  --batch-admission-policy-config \
  '{"execution":"decoupled","stream_mode":"overlap","layers_per_slice":8,"trace_path":"local-results/tp4/overlap-trace-{pid}.jsonl"}'

# Adaptive fenced uses the same command, with this policy configuration:
# {"execution":"decoupled","stream_mode":"fenced","layers_per_slice":8,
#  "adaptive_layers":true,"trace_path":"local-results/tp4/adaptive-fenced-trace-{pid}.jsonl"}
```

For each running endpoint, use `scripts/smoke_serving.py` and
`scripts/benchmark_serving.py --repeats 2` with the appropriate `--base-url`,
`--label` and `--output`. Benchmarks warm up all four input shapes with 8 output
tokens, then issue 8 staggered requests at 120 ms intervals, lengths
128/2048/512/4096 and 64 output tokens, twice. Throughput is total output tokens
divided by total wall time across both runs. Percentiles are nearest-rank over
the 16 requests. TPOT is a request's mean token latency, not individual-token
tail latency. These short synthetic measurements do not establish statistical
significance or production SLOs.

Capture the TP4 native correctness reference before stopping that server:

```bash
/root/workspace/.venv/bin/python scripts/compare_outputs.py \
  --baseline-url http://127.0.0.1:8000 --baseline-only \
  --output local-results/tp4/native-correctness.json

# Later, against each plugin mode:
/root/workspace/.venv/bin/python scripts/compare_outputs.py \
  --baseline-report local-results/tp4/native-correctness.json \
  --candidate-url http://127.0.0.1:8002 --require-exact \
  --output local-results/tp4/native-vs-overlap.json
```

Each endpoint receives 16 natural-language cases (isolated/paired, twice).
The paired long prompt is sent after decode's first content event while decode
is still active. The baseline is validated and its file SHA256 recorded on reuse.
Correctness checks and profiling are outside timed benchmark runs.

For a separate device capture, append this option to the overlap server launch:

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/workspace/vllm-ascend-MuxPD-hust/local-results/tp4/overlap-profile","ignore_frontend":true,"torch_profiler_with_stack":false,"max_iterations":8}'
```

After the normal serving tests finish, capture and parse offline:

```bash
curl --noproxy '*' -fsS -X POST http://127.0.0.1:8002/start_profile
/root/workspace/.venv/bin/python scripts/exercise_decoupled.py \
  --base-url http://127.0.0.1:8002 \
  --output local-results/tp4/overlap-profile-client.json
curl --noproxy '*' -fsS -X POST http://127.0.0.1:8002/stop_profile
source /root/workspace/runtime/env.sh
/root/workspace/.venv/bin/python -c \
  "from torch_npu.profiler.profiler import analyse; analyse('local-results/tp4/overlap-profile')"
```

Use the runtime's physical stream IDs and each new capture's device labels when
constructing lane maps. The archived independent audit reproduces the existing
capture, whose paths and hashes are recorded in its report.
