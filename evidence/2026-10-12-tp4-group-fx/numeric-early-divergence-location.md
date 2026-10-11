# Existing-data location of the early lane-0 numerical difference

No new requests or runtime changes. The four observations below use the two
completed 19-request reports, already independently audited from raw SSE, and
their retained scheduler traces. Reproduce with
`python3 local-results/tp4-prefill-group-fx/locate_groupfx_numeric_difference.py`.
The companion JSON binds the reports to their existing audit SHA, records every
trace/manifest SHA and relevant trace line/byte hashes, and retains source rows.

Both repeats share generated prefix `[220, 220, 131]`; the fourth generated token
(zero-based index 3) is 223 in raw and 224 in group FX. Later generated sequences
have different autoregressive histories and should not be used as same-input
logit comparisons.

| Variant / request index | Fourth token | Reception before 12K client dispatch | Reception before 12K admission trace |
|---|---:|---:|---:|
| Raw / 6 | 223 | 313.462 ms | 509.373 ms |
| Raw / 11 | 223 | 300.284 ms | 549.332 ms |
| Group FX / 6 | 224 | 345.160 ms | 545.962 ms |
| Group FX / 11 | 224 | 337.471 ms | 601.536 ms |

In all four cases reception also precedes that 12K request's scheduler-entry
timestamp, not merely its later admission-log write. Thus the newly dispatched
12K request cannot cause this already-observed early difference.

The actual early batch is **D1 plus pending P384**, not D4. For both variants and
both repeats, the lane-0 initialization P128 is admitted alone. The other three
P128 initialization requests are admitted together afterwards, totaling 384
tokens. At `computed_before=130`, the lane-0 request's third one-token decode step
produces output index 3 while those other three P requests remain pending. The
P384 advances from 8 to 12 layers in that transaction in all four observations;
the inferred D step is 58 for the first repeat and 197 for the second. The
companion JSON includes the exact P cohort and layer-progress values. This step
mapping follows the synchronous, non-speculative native scheduling contract; it
is not an independently observed device timestamp.

Both producer and scheduler use `time.monotonic()` on this server. The owned
ordinary subprocess launch does not introduce a separate clock namespace; the
recorded Python implementation is `clock_gettime(CLOCK_MONOTONIC)`. Producer
timestamps represent receipt of enough HTTP byte-chunk data to complete an SSE
frame. They are not GPU timestamps. Reception before admission excludes that
later request as a cause; reception after admission would not establish the
opposite causal ordering.

The remaining hypotheses include state produced by lane-0's own initial P128,
and interaction with the three subsequent P128 requests during the early D1
steps. Equal token prefixes do not prove equal KV/GDN state. These observations
do **not** establish fusion error, harmless BF16 roundoff, HCCL misrouting or KV
corruption.

A bounded next diagnostic can first remove the later 12K request, retain the
four original initialization prompts, and record the actual cohort/shape. At the
common prefix, inspect the D sampler's true post-processor greedy input: top-two
IDs/values and the values for IDs 223 and 224, with request ID, prefix, step and
rank. API logprobs are currently rejected by the plugin, so such instrumentation
must be separately registered; it must not resample or alter the original ACK
boundary. A near-zero top-two margin would show argmax sensitivity, not by itself
prove numerical correctness. A D1-only variant would distinguish initialization
state differences from the concurrent P384 path, if the preceding diagnostic
actually reproduces the difference.
