# TP8 correctness diagnostics

This report uses only this TP8 run. All planned reports are complete.

| Service | Natural oracle / receipts | Natural exact vs native | Synthetic transport | Synthetic unstable repeat groups | Synthetic exact vs native | State transport |
| --- | --- | --- | --- | --- | --- | --- |
| native_graph | 16/16 | — | 56/56 | 10/28 | — | 30/30 |
| muxpd_graph_work256 | 16/16 | 16/16 | 56/56 | 11/28 | 33/56 | 30/30 |
| native_graph_bracket | 16/16 | 16/16 | — | — | — | — |
| native_eager | — | — | — | — | — | 30/30 |

State-reuse rows use actual generated token IDs: four serial target samples per case (two before and two after a long P+D request). The six brackets add six background requests, for 30 requests per service.

| Service | Case | First generated token stable across all four | Entire generated sequence stable across all four |
| --- | --- | --- | --- |
| native_graph | original_one_token_max1 | False | False |
| native_graph | original_one_token_max16 | False | False |
| native_graph | hello_one_token_max1 | False | False |
| native_graph | hello_one_token_max16 | False | False |
| native_graph | hello_hi_two_tokens_max1 | True | True |
| native_graph | hello_hi_two_tokens_max16 | True | False |
| muxpd_graph_work256 | original_one_token_max1 | True | True |
| muxpd_graph_work256 | original_one_token_max16 | True | True |
| muxpd_graph_work256 | hello_one_token_max1 | True | True |
| muxpd_graph_work256 | hello_one_token_max16 | True | True |
| muxpd_graph_work256 | hello_hi_two_tokens_max1 | True | True |
| muxpd_graph_work256 | hello_hi_two_tokens_max16 | True | False |
| native_eager | original_one_token_max1 | True | True |
| native_eager | original_one_token_max16 | True | False |
| native_eager | hello_one_token_max1 | True | True |
| native_eager | hello_one_token_max16 | True | True |
| native_eager | hello_hi_two_tokens_max1 | True | True |
| native_eager | hello_hi_two_tokens_max16 | True | False |

In the MuxPD sample, 4/4 one-token-prompt cases keep the entire generated sequence identical across all four before/after samples. This differs from the native graph sample; it is an observation on these finite probes, not proof of a general fix or that the candidate tokens are uniquely correct.

The native eager control keeps the first generated token stable in 4/4 one-token-prompt cases; 3/4 keep the entire output stable. `original_one_token_max16` differs in a later token during its before repeats. Thus first-token instability is observed in the native graph service in this sample, while later sequence divergence also occurs with native eager execution. This separates the observations without identifying a unique underlying cause.

The native TP8 graph control has one-token-prompt first-token instability. For the two-token prompt, the first generated token and max1 output are stable in the completed graph reports, while the max16 output can diverge later in the sequence. Inspect the per-service table above; this report makes no claim that such later divergence is new at TP8. Native instability prevents a plugin-only attribution, but does not prove the root cause or excuse candidate-specific differences.

All 56 synthetic SSE bodies per available endpoint are re-parsed, with saved-body SHA, HTTP 200, response identity, DONE, finish reason and exact usage checked. State responses are validated against their exact prompt IDs and requested output lengths, and strictly serial timestamps are checked. Natural reports retain events and completion flags but no raw SSE body or HTTP-status field; this narrower evidence is stated separately in JSON.

Producer hashes: `compare_outputs.py` 1ef81477269707616c5d0a14becce80df4acf3a671f17484e5716d7fe25fa551 (reconstructed from service commit; older report has no embedded hash); `check_decode_graph_serving.py` 1e6cb3ef7e96fd8d2521e85f2d7f90833a556b7565fb1edc6ae79b06a0ad5a77; `probe_graph_state_reuse.py` 07fc9dedcd3c56db77e5aca3e28eb4f9c944fcb762902bf81fda239b6b3d8da4. The latter two are also embedded in their reports. Every report SHA is checked against its service manifest's successful client receipt, every manifest is pinned to commit `3eb6f2ea0dea5b95109435182d106ff1ce15e8cd`, and copied baseline requests/hashes are checked against the original report. JSON retains those manifest hashes, input-report SHA-256, byte sizes, all mismatch keys and compact state token sequences.

Reproduce offline with the workspace Python: `python local-results/tp8/correctness-summary-audit.py`. This audit never sends requests or imports torch. These diagnostics do not establish actual device graph execution, hardware overlap, overall quality, or performance.
