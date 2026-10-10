# Ascend integration evidence — 2026-10-10

Hardware: Ascend 910B2. Model: Qwen3.5-35B-A3B BF16, TP2 + EP,
16K context, max 8 sequences, eager, no speculative decoding.

Host source pins:

- vllm-hust: `d0f22d2bda562156e4dbf433ce645e1769b4f804`
- vllm-ascend-hust: `03766ac696fde5ab1980d80ca0b8543d3580c989`
- MuxWise reference: `eeac5148f456428a05823c330e1d40afb0ef2438` (`slo_config`)

These are synthetic integration experiments, not production performance claims.
No CUDA paper speedup or NVIDIA regression parameters are transferred to Ascend.

## Receipts

| File | Experiment |
|---|---|
| `original-smoke.json` | Original runtime configuration; all 9 API cases passed |
| `serial-smoke.json` | MuxPD serial scheduler; all 9 API cases passed |
| `original.json` | Original 2048-token budget, Mamba align, prefix cache enabled |
| `serial-default.json` | MuxPD serial, 16384-token budget, Mamba none, prefix cache disabled |
| `serial-trace.jsonl.gz` | Scheduler admissions/completions for serial smoke and benchmark |

The API cases include greedy arithmetic, three concurrent Chinese SSE requests,
an exact 2305-token input, cancellation, health after cancellation, and subsequent
generation. The trace validates full remaining prefill and homogeneous P/D;
it does not prove kernel concurrency or prove KV reclamation after HTTP cancel.
Native allocator unit/integration tests check lifecycle behavior separately.

Each benchmark warms up input lengths 128, 512, 2048, 4096, then runs 8 staggered
requests with 64 generated tokens and two repetitions. Input shapes cycle through
128, 2048, 512, 4096 at 120 ms arrival intervals. `ignore_eos=true` fixes output
length. Repetitions reuse the same synthetic prompts, so the original cache-enabled
run can benefit from cache reuse; all settings are preserved in the receipts.

The original configuration measured 7.02 and 7.10 output tokens/s; the initial
serial configuration measured 33.10 and 33.32. **This is a configuration comparison,
not an isolated MuxPD speedup:** token budget, Mamba mode and caching differ, and
different pairs of the same NPU model were used. The initial serial experiment
also overlapped part of a separate server's startup on the shared host. A matched
control is required before attributing a gain to scheduling.

At the benchmark's TTFT <= 2 s / mean TPOT <= 0.1 s SLO, both configurations had
zero request goodput. The serial scheduler improved admission latency in this
workload but did not meet the 100 ms decode target. Device overlap remains unproven.

## Reproduce

Use `scripts/smoke_serving.py` and `scripts/benchmark_serving.py` with the commands
in the repository README. For trace validation:

```bash
gzip -dc evidence/2026-10-10/serial-trace.jsonl.gz > /tmp/muxpd-serial-trace.jsonl
/root/workspace/.venv/bin/python scripts/check_trace.py /tmp/muxpd-serial-trace.jsonl
```

Raw verbose server logs are retained locally under `local-results/`; the receipts
contain synthetic inputs/outputs and timing observations, not user traffic.
