# Scheduler and benchmark review, TP=4 execution-overhead candidate

This is a source review and CPU fixture record for branch `perf/tp4-execution-overhead`. It contains no new NPU performance result. The original TP=2 cached/aligned service and the TP=2 adaptive fenced service have different configurations and are not a matched performance comparison.

The review found and fixed one fairness gap in `prefill_first`. The base layered selector can choose P-only before the new hook when a whole prompt cannot fit alongside D in the token budget. Initially that path did not create decode debt. A stream of long prompts with `max_tokens=1` could keep releasing a slot and repeatedly bypass live D. With `prefill_first=True`, the hook now recovers the original D candidates and labels this path `full_budget_prefill_first`; successful P-only completion creates the same debt as `deadline_prefill_first`. The next eligible transaction selects D only. KV allocation failure remains `prefill_kv_fallback` and creates no debt. The default-off and fixed-quota paths retain their prior admission behavior.

Relevant implementation is in `src/vllm_muxpd/multiplex_scheduler.py::_adjust_phase_selection`, `src/vllm_muxpd/decoupled_scheduler.py::_adjust_phase_selection`, `schedule`, and `update_from_output`. The hook runs before `_schedule_layered_selection` invokes native `Scheduler.schedule`. D requests omitted by the allowlist are not scheduled and their `num_computed_tokens` are not advanced. There is no post-scheduling deletion or token-counter rollback.

Lifecycle review found no further blocking defect in the inspected scheduler paths:

- Decode debt is committed only after strict worker receipt validation and native result retirement. An invalid receipt cannot create debt or retire results.
- Debt includes only deferred IDs still in `self.running`; cancelled, preempted, or cancelled-and-reused waiting IDs are excluded. Native external cancellation removes requests from queues before `_free_request_blocks`, which discards their debt.
- Native `_preempt_request` remains responsible for freeing/resetting KV and returning the request to the waiting queue; the adapter then removes its decode debt. `reset_prefix_cache(reset_running_requests=True)` calls this same override, so it also clears debt. A pending P cohort still rejects reset of running requests.
- Debt repayment overrides another full-budget P-only selection. Actual D retirement clears serviced IDs. Cancelling one indebted D preserves the others.
- `PAUSED_NEW` still allows the existing running D requests to repay debt. `PAUSED_ALL` does not run P or D; debt remains until resume or cancellation. Entering `PAUSED_ALL` while P is pending retains the existing fail-closed guard.
- The pending-cohort cancellation protocol still pins KV until worker completion/cancel ACK. All-aborted pending P keeps the engine active for the ACK even when no user request remains. Prefill-first does not introduce a second cohort or bypass those receipts.
- A mandatory D turn can require a new KV block and preempt the newly completed P request. The native fixture explicitly exercises this behavior. Preserving native allocation does not guarantee that every P request will avoid recomputation under memory pressure.

The fairness promise is an admission-turn bound for live schedulable D, not a wall-clock TPOT guarantee. A full P forward can take longer than the configured TPOT SLO; no token chunking is introduced to avoid this. If D is cancelled or preempted, it no longer has a runnable decode obligation. The policy preserves native request order rather than bypassing a large earlier prompt to favor shorter later prompts.

Validation after the fairness fix: `source /root/workspace/runtime/env.sh; python -m pytest -q tests/test_policy.py tests/test_decoupled_scheduler.py tests/test_multiplex_scheduler.py tests/test_scheduler.py tests/test_decoupled_trace.py` produced **134 passed** in 33.01 seconds. These tests use the real native scheduler/KV allocator with synthetic worker receipts and do not load a model on NPU. Ruff checks passed for the changed scheduler and tests. Hardware validation of the combined worker optimizations is separate.

The current `scripts/benchmark_serving.py` is suitable for comparing the client-visible completion throughput and latency of the same finite staggered burst, with these definitions:

- Throughput is completion tokens divided by each full burst wall time, including admission, staggered arrival, prefill, and the completion tail. Pooled throughput must use total tokens divided by summed measured wall times. It is not steady-state decode-only throughput.
- `ttft_s` is the time to the first **nonempty text SSE**. It is not a direct device timestamp or necessarily the first generated token: detokenization can produce an empty text delta.
- `tpot_s` is `(last nonempty text event - first nonempty text event)/(completion_tokens-1)`, a request-level average. Its P95 across requests is not a P95 of individual token latency. The existing maximum content-event gap should also be reported because a one-time decode stall can be hidden by the average.
- The producer requires `[DONE]`, usage, nonempty text, and the requested completion-token count before saving. A 62-event response with 64 completion tokens is not evidence of missing tokens: only nonempty text events are retained. Raw HTTP frames are not archived.
- Nearest-rank P95 for 16 requests is the maximum. Two repeats do not establish statistical significance; retain each repeat and the pooled result. Adaptive estimates persist between repeats, so the repeats are not independent cold starts.
- `arrival_delay_s` is currently the intended delay, not the observed dispatch time. `asyncio.sleep`, prompt construction, and event-loop work can shift actual dispatch and therefore alter P grouping near a scheduling boundary.

The smallest useful additional receipt fields would require no extra server requests and preserve existing metric definitions:

1. Per run: start/end UTC timestamps and a `perf_counter` origin.
2. Per request: actual dispatch and response-end offsets from that run origin, plus dispatch lateness relative to the intended arrival. Pass the same run origin to all request tasks rather than reconstructing absolute timing from the requested sleep delay.
3. Per response: the first SSE `id` (to associate benchmark index with backend cohort IDs), terminal `finish_reason`, explicit `done_received`, and nonempty-text event count.
4. Record the first choice-bearing SSE time including an empty text delta, alongside the existing first-nonempty-text TTFT. Label it as a client event receipt and retain the existing metric for comparisons with prior reports.
5. Persist the producer source hash and generation parameters (`temperature=0`, `seed=42`, `ignore_eos=True`) so reports remain interpretable if the script changes.

For finer token-level diagnostics, this pinned vLLM completion protocol supports `return_token_ids`, but enabling it also includes the entire prompt-token list in the first streamed chunk. That changes response serialization and payload size, especially for long prompts. Use it in a separate diagnostic run or consistently in a new benchmark series, not silently in one side of the present A/B comparison. Merely counting nonempty text events cannot recover true token IDs or per-token timestamps.

No benchmark-script change was made during this review. The new receipt fields are proposals; existing evidence must not be described as containing them.
