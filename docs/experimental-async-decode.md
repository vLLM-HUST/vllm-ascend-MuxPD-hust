# Experimental asynchronous decode queue

This prototype is disabled by default and requires an explicit matching pair:

```text
--scheduler-cls vllm_muxpd.async_scheduler.AsyncDecoupledMuxPDScheduler
--worker-cls vllm_muxpd.async_worker.AsyncDecoupledMuxPDNPUWorker
--async-scheduling
```

The existing MuxPD policy must additionally specify
`"async_decode_pipeline": true` and `"execution": "decoupled"`. All existing
BF16 Qwen3.5, text-only, TP2/TP4 deployment restrictions continue to apply.
Layer quotas are fixed; adaptive quotas, prefill-first, full-prefill compilation
and the decode budget guard are excluded from this prototype. Grouped prefill
retains its existing separate opt-in and backend restrictions.

Only transactions containing decode requests alone can have two batches in
flight. Waiting or skipped prefill requests stop queue filling. Prefill
admission, continuation and cancellation run after all queued decode outputs
retire, with only one such transaction in flight. A zero-token continuation or
cancellation still returns its progress receipt from the worker's execute RPC;
the native queued engine does not issue a separate sampling RPC in that case.

The decode runner uses native asynchronous sampling, its private output-copy
stream and device token feedback. Its output wrapper owns a transaction-local
request order and immutable empty-prefill receipt, so the native output thread
does not read state replaced by a later worker iteration. Prefill sampling
remains synchronous. Mixed steps materialize decode output before returning
the original combined receipt.

Decode output placeholders are included in admission snapshots, but are never
added for pending prefill. Explicit full histories cover both the first
decode following prefill and a preempted decode returning to prefill; shared
CPU token histories may otherwise still contain asynchronous placeholders.
Aborted or preempted decode requests retain their physical KV blocks until the
last queued output has retired. Reusing an ID with an outstanding decode
output is rejected. Keep-pause and resetting running requests require a queue
drain. The existing caller/decode stream dependencies remain in place.

CPU tests exercise queue retirement, placeholders, full-history transitions,
zero-token engine dispatch, cancellation, preemption and block ownership.
These tests do not qualify device execution, task quality or performance.
This is an ordinary explicit-class experiment: the Manager manifest and CLI
deployment profile are unchanged, and pure-decode output wrappers do not yet
emit the worker runtime-effective event required by Manager qualification.
No throughput improvement or resolution of the existing quality issue is
claimed before separate hardware validation.
