# Experimental prefill collective prefix layout

`experimental_prefill_collective_layout` defaults to `"off"`. `"prefix"` is an
explicit experimental communication-layout option. It has CPU lifecycle tests,
but this implementation on current main has **not** passed a new accelerator
baseline bridge, full quality gate, or matched performance evaluation. It must
remain opt-in until those checks are completed. No default manifest, environment,
or ordinary launch recipe is changed.

## Supported scope

The initial profile requires the exact synchronous
`DecoupledMuxPDScheduler` / `DecoupledMuxPDNPUWorker` pair, V1 NPU runners, isolated
P communication, overlap streams, fixed q4, raw complete fresh prefills, and
FULL_DECODE_ONLY graphs with capture sizes `[1,2,4,8]`. It requires BF16 TP4/EP4
on Ascend A2 and the Qwen3.5-35B-A3B architectural profile (including 40 layers,
2048 hidden width, 256 experts and eight selected experts). Architectural
validation is not checkpoint/weight authentication. Other checkpoints still
require their own numerical and quality bridge.

Async/atomic workers, subclasses, multiple concurrent batches, adaptive slicing,
group/full-prefill compilation, chunked/recomputed/resumed P, speculation,
prefix caching, LoRA, quantization, KV/EC transfer, DP/PP/PCP/DCP greater than one,
DBO, MoE sequence parallelism, FlashComm/SP, fine-grained TP, shared-expert
multistream overlap, EPLB and elastic EP startup are rejected. Existing worker
guards continue to check the actual Ascend configuration, including disabled
`enable_npugraph_ex` and `enable_static_kernel` for this experiment.

The option requires `require_hccl_deterministic=true` and literal
`HCCL_DETERMINISTIC=true` **before distributed initialization**; batch invariance
must be disabled. This environment setting does not prove a particular HCCL
algorithm or global determinism.

## Launch configuration

The installed CLI exposes `--experimental-prefill-prefix-layout`. It requires
`--deterministic-collectives`, is mutually exclusive with
`--experimental-prefill-groups`, and rejects TP2. For example, inspect the
recipe without launching any device work:

```sh
vllm-muxpd plan --model /path/to/Qwen3.5-35B-A3B \
  --experimental-prefill-prefix-layout --deterministic-collectives
```

Use the same flags with `configuration` and `serve` when explicitly selecting
this experiment through the Manager. The configuration command does not enable
an extension. The host recipe explicitly disables the two incompatible Ascend
compilation backends. Direct policy configuration accepts only the literal
strings `"off"` and `"prefix"`; bool, `auto`, diagnostic `copy_only`, and `suffix`
are rejected. There is no automatic enablement, fallback, retry, or input-based
selection.

## Tensor and admission contract

The wrapper reads the complete scheduler output before lane slicing. For a new
P cohort, it stores the actual scheduled P count map and the other scheduled
request counts as D. Each D request must schedule exactly one token. Waiting,
deferred, or active-but-unscheduled requests do not contribute to D. This frozen
D-token sum remains owned by the P coroutine across every suspension, regardless
of later decode batch sizes.

For actual contiguous P BF16 TP partial sums `X[P,H]`, the transform allocates a
new zero-filled `[D+P,H]` scratch on the original producer stream, copies X to
rows `[D:D+P]`, calls the existing global TP coordinator's original
`_all_reduce_out_place` exactly once, and copies those result rows back into X.
It returns **the original X object**, preserving storage, stride and aliases.
The existing router and isolated P sibling group select the actual collective.
No communicator, dispatcher, model token, dummy request, KV row, attention
metadata or extra collective is created. EP collectives are not wrapped.
D0, decode and caller lanes call the original function directly with the
original tensor; D0 performs no scratch allocation or extra copy.

The first real P collective verifies owning coroutine/cohort, raw forward
context, packed token IDs, positions, offsets and full fresh prompt lengths.
Single active cohort, one host thread, unchanged three method slots, intact
router ownership, exact shape/dtype/device and no reentrant P collective are
required. Unknown lanes fail. Original P ordering is preserved: native may
order P differently following D reordering, and native D values are not zero.
Consequently this option does **not** establish general native-layout or
numerical equivalence.

## Startup, failure and shutdown

Installation occurs at the successful end of `compile_or_warm_up_model`, after
native warmup/ATB/seed reset returns and `redirect_owner_capture` has exited.
Both runners, graph state and lane/cohort ownership must be idle and valid.
Only then are three slots installed: worker `execute_model`, worker
`_clear_prefill`, and the original global TP coordinator's
`_all_reduce_out_place`. No lazy first-collective installation occurs.

For the pinned core `d0f22d2bda562156e4dbf433ce645e1769b4f804`,
`Executor.initialize_from_config` performs warmup as its final worker RPC and
then aggregates compilation times. `EngineCore` then constructs host scheduler,
structured-output and queue state. Unsupported connector/async/elastic modes
are excluded. The pinned Ascend
`03766ac696fde5ab1980d80ca0b8543d3580c989` worker performs its ATB and seed work
before returning from native warmup. The first real execute also compares the
installation router snapshot and runner-idle state to detect unexpected
intervening collective/model state. These source boundaries need re-auditing
for another native stack; device startup bridge evidence remains required.

Any admission, ownership, collective or copy-back failure poisons the experiment
permanently for that worker. A collective is never retried. Failure retains
cohort ownership and any scratch/result buffers that device work could still
reference. Only an admission that never acquired a worker task/cohort may roll
back immediately. Clear failure or unfinished tasks retain ownership; retrying
safe cleanup does not erase the primary failure.

Cancellation uses the existing abort path: close the P task, synchronize its
stream, retire/ACK, then clear. Terminal shutdown aborts P and synchronizes P/D
and the decode update stream before restoring the layout slots; graph, router,
and groups are destroyed afterward. Terminal pending sampling state is not
reported as a successful scheduler ACK. Synchronization or restoration failure
keeps resources available for cleanup retry. A foreign replacement slot is
never overwritten. Partial installation rolls back only acquired slots still
owned by this controller. Outer method observers are unsupported in this first
version and must not be added to a running layout controller.

## Bounded diagnostics for long serving runs

Active safety state is independent of history. There is no lifetime cohort
limit. Cohort identity is checked against one monotonically increasing watermark;
closed records use a ring of 16 cohorts and cumulative counters. Snapshot
`history_is_complete=false` explicitly reports history truncation; the ring is
not an audit of the entire process lifetime. Errors retain the first failure
and at most eight recent messages. The existing per-cohort 4096 transformed-call
and 512 MiB scratch payload limits remain fail-closed. They are bounds on one
active forward, not a limit on service duration. No device buffers are kept in
the completed history.

Startup logs include profile/module hash, rank, graph state, original/P group
names and ranks, slot ownership, and startup boundary. `snapshot_on_worker`
provides current active metadata, recent calls and cumulative counters through
an explicitly invoked read-only worker RPC. Shutdown reports restoration status,
counters and primary failure. Cleanup success is separate from collection,
quality, process-exit and owned-process cleanup success.

The first version intentionally keeps per-call `new_zeros` and both copies.
Scratch reuse or zeroing only D rows would require a separate implementation
and numerical/performance bridge. Post-warmup scratch memory is additional to
the earlier KV memory profile and may cause an allocation failure; no fallback
is attempted.

## Validation still required

CPU tests cover default-off configuration, unsupported combinations, actual
scheduled D0/D1/multi-D maps, suspended P with changing D batches, alias/single
collective behavior, fresh-P checks, poisoning, clear/cancellation/sync failures,
slot restoration, partial installation and 200 successive cohorts with bounded
history. Lifecycle tests execute the actual worker methods against CPU sentinels.

Frozen external diagnostic results motivated this experiment; they do not
qualify this implementation. Before considering a merge/release, establish a
new main raw/off baseline, main-prefix to frozen-prefix startup/tensor/output
bridge, complete original quality evaluation with unchanged scoring and token
budgets, and matched off/on performance measurements. Report original strict
completeness gates and any native truncations honestly. Neither the CPU tests
nor a small layer/logits bridge replaces those checks.
