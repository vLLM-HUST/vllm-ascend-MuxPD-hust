# Independent contention receipt audit

Final selected initial+confirm comparison: **passed**. Passing individual reports does not complete a missing final stage.

Raw SSE bytes, byte-chunk arrival times, tokens, terminal messages, per-request metrics, scoped intact gaps, and all measured aggregate fields were independently recomputed. The benchmark module was not imported. Warmup was checked and excluded from measured summaries; all repeats are retained.

| Report | Audit | D SLO 100/200 ms | P SLO 1/2 s | D max-gap p95 (ms) | Post-gate max: intact / contained (ms) | P TTFT p95 (ms) | Pooled output tok/s | Multi-token events |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| guard-initial-decode-control | PASS | 12/12; 12/12 | 0/0; 0/0 | 36.31 | 27.63 / 27.63 | n/a | 157.91 | 0 |
| guard-initial-p8k | PASS | 12/12; 12/12 | 0/3; 3/3 | 49.11 | 49.15 / 49.15 | 1093.65 | 154.82 | 0 |
| guard-initial-p12k | PASS | 12/12; 12/12 | 0/3; 3/3 | 49.54 | 49.57 / 49.57 | 1167.88 | 152.09 | 0 |
| guard-initial-p12k-burst | PASS | 12/12; 12/12 | 0/9; 9/9 | 60.10 | 60.15 / 60.15 | 1473.59 | 141.64 | 0 |
| guard-v2-confirm-decode-control | PASS | 12/12; 12/12 | 0/0; 0/0 | 68.65 | 28.08 / 28.08 | n/a | 167.76 | 0 |
| guard-v2-confirm-p8k | PASS | 12/12; 12/12 | 3/3; 3/3 | 66.17 | 56.93 / 56.93 | 505.73 | 162.16 | 0 |
| guard-v2-confirm-p12k | PASS | 12/12; 12/12 | 3/3; 3/3 | 65.17 | 63.64 / 63.64 | 523.74 | 162.74 | 0 |
| guard-v2-confirm-p12k-burst | PASS | 12/12; 12/12 | 9/9; 9/9 | 65.96 | 66.14 / 66.14 | 522.68 | 151.79 | 0 |
| guard-v2-initial-decode-control | PASS | 12/12; 12/12 | 0/0; 0/0 | 63.52 | 26.69 / 26.69 | n/a | 170.01 | 0 |
| guard-v2-initial-p8k | PASS | 12/12; 12/12 | 3/3; 3/3 | 65.46 | 65.38 / 65.38 | 488.12 | 163.08 | 0 |
| guard-v2-initial-p12k | PASS | 12/12; 12/12 | 3/3; 3/3 | 67.48 | 67.49 / 67.49 | 506.95 | 161.69 | 0 |
| guard-v2-initial-p12k-burst | PASS | 12/12; 12/12 | 9/9; 9/9 | 68.19 | 68.19 / 68.19 | 514.96 | 146.94 | 0 |
| native16k-confirm-decode-control | PASS | 9/12; 9/12 | 0/0; 0/0 | 365.58 | 27.84 / 27.84 | n/a | 174.35 | 0 |
| native16k-confirm-p8k | PASS | 0/12; 0/12 | 3/3; 3/3 | 391.58 | 391.59 / 391.59 | 409.46 | 166.61 | 0 |
| native16k-confirm-p12k | PASS | 0/12; 0/12 | 3/3; 3/3 | 373.99 | 374.01 / 374.01 | 390.10 | 168.44 | 0 |
| native16k-confirm-p12k-burst | PASS | 0/12; 0/12 | 9/9; 9/9 | 398.02 | 398.08 / 398.08 | 421.62 | 151.82 | 0 |
| native16k-initial-decode-control | PASS | 9/12; 9/12 | 0/0; 0/0 | 371.07 | 25.92 / 25.92 | n/a | 172.50 | 0 |
| native16k-initial-p8k | PASS | 0/12; 0/12 | 3/3; 3/3 | 388.25 | 388.28 / 388.28 | 407.67 | 166.35 | 0 |
| native16k-initial-p12k | PASS | 0/12; 0/12 | 3/3; 3/3 | 385.42 | 385.48 / 385.48 | 407.08 | 167.90 | 0 |
| native16k-initial-p12k-burst | PASS | 0/12; 0/12 | 9/9; 9/9 | 380.81 | 380.85 / 380.85 | 404.23 | 152.80 | 0 |
| native2k-confirm-decode-control | PASS | 9/12; 9/12 | 0/0; 0/0 | 390.01 | 27.75 / 27.75 | n/a | 172.87 | 0 |
| native2k-confirm-p8k | PASS | 0/12; 0/12 | 0/3; 3/3 | 403.08 | 403.09 / 403.09 | 1990.18 | 133.96 | 0 |
| native2k-confirm-p12k | PASS | 0/12; 0/12 | 0/3; 0/3 | 372.30 | 372.32 / 372.32 | 2177.55 | 131.88 | 0 |
| native2k-confirm-p12k-burst | PASS | 0/12; 0/12 | 0/9; 0/9 | 378.62 | 378.62 / 378.62 | 4887.58 | 88.69 | 0 |
| native2k-initial-decode-control | PASS | 9/12; 9/12 | 0/0; 0/0 | 394.41 | 29.59 / 29.59 | n/a | 172.09 | 0 |
| native2k-initial-p8k | PASS | 0/12; 0/12 | 0/3; 3/3 | 395.10 | 395.10 / 395.10 | 1967.99 | 133.59 | 0 |
| native2k-initial-p12k | PASS | 0/12; 0/12 | 0/3; 0/3 | 400.57 | 400.87 / 400.87 | 2352.85 | 127.90 | 0 |
| native2k-initial-p12k-burst | PASS | 0/12; 0/12 | 0/9; 0/9 | 413.55 | 413.55 / 413.55 | 5465.34 | 84.84 | 0 |
| old-work256-confirm-clean-decode-control | PASS | 9/12; 9/12 | 0/0; 0/0 | 377.55 | 379.27 / 26.02 | n/a | 171.42 | 0 |
| old-work256-confirm-clean-p8k | PASS | 0/12; 0/12 | 3/3; 3/3 | 377.23 | 378.39 / 302.26 | 417.68 | 162.37 | 0 |
| old-work256-confirm-clean-p12k | PASS | 0/12; 0/12 | 3/3; 3/3 | 376.18 | 376.65 / 211.58 | 439.84 | 164.23 | 0 |
| old-work256-confirm-clean-p12k-burst | PASS | 0/12; 3/12 | 9/9; 9/9 | 375.24 | 376.90 / 217.23 | 434.22 | 150.58 | 0 |
| old-work256-confirm-decode-control | PASS | 9/12; 9/12 | 0/0; 0/0 | 359.04 | 364.29 / 26.15 | n/a | 173.02 | 0 |
| old-work256-confirm-p8k | PASS | 0/12; 0/12 | 3/3; 3/3 | 366.09 | 378.49 / 302.83 | 414.49 | 164.39 | 0 |
| old-work256-confirm-p12k | PASS | 0/12; 3/12 | 3/3; 3/3 | 379.47 | 382.68 / 215.06 | 427.98 | 163.41 | 0 |
| old-work256-confirm-p12k-burst | PASS | 0/12; 3/12 | 9/9; 9/9 | 351.12 | 352.91 / 215.05 | 425.59 | 151.13 | 0 |
| old-work256-initial-decode-control | PASS | 9/12; 9/12 | 0/0; 0/0 | 372.04 | 375.09 / 28.50 | n/a | 170.10 | 0 |
| old-work256-initial-p8k | PASS | 0/12; 0/12 | 3/3; 3/3 | 377.25 | 377.76 / 305.49 | 420.17 | 161.93 | 0 |
| old-work256-initial-p12k | PASS | 0/12; 0/12 | 3/3; 3/3 | 379.04 | 381.97 / 218.00 | 433.68 | 161.93 | 0 |
| old-work256-initial-p12k-burst | PASS | 0/12; 0/12 | 9/9; 9/9 | 377.17 | 382.46 / 215.34 | 447.95 | 147.56 | 0 |

## Selected final pools

- native16k: native16k-initial, native16k-confirm
- native2k: native2k-initial, native2k-confirm
- old-work256: old-work256-initial, old-work256-confirm-clean
- guard: guard-v2-initial, guard-v2-confirm

| Role | Workload | Final status | Missing own stages |
| --- | --- | --- | --- |
| native16k | decode-control | passed | none |
| native2k | decode-control | passed | none |
| old-work256 | decode-control | passed | none |
| guard | decode-control | passed | none |
| native16k | p8k | passed | none |
| native2k | p8k | passed | none |
| old-work256 | p8k | passed | none |
| guard | p8k | passed | none |
| native16k | p12k | passed | none |
| native2k | p12k | passed | none |
| old-work256 | p12k | passed | none |
| guard | p12k | passed | none |
| native16k | p12k-burst | passed | none |
| native2k | p12k-burst | passed | none |
| old-work256 | p12k-burst | passed | none |
| guard | p12k-burst | passed | none |

| Role / workload | Windows | Pooled output tok/s | Stage tok/s min–max | D request max-gap p95 ms | D mean TPOT p95 ms | P TTFT p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native16k / decode-control | 6 | 173.42 | 172.50–174.35 | 371.38 | 21.77 | n/a |
| native2k / decode-control | 6 | 172.48 | 172.09–172.87 | 394.21 | 21.87 | n/a |
| old-work256 / decode-control | 6 | 170.76 | 170.10–171.42 | 375.99 | 22.18 | n/a |
| guard / decode-control | 6 | 168.87 | 167.76–170.01 | 65.81 | 21.73 | n/a |
| native16k / p8k | 6 | 166.48 | 166.35–166.61 | 391.58 | 22.92 | 411.43 |
| native2k / p8k | 6 | 133.78 | 133.59–133.96 | 402.63 | 28.91 | 1986.66 |
| old-work256 / p8k | 6 | 162.15 | 161.93–162.37 | 377.62 | 23.55 | 419.90 |
| guard / p8k | 6 | 162.62 | 162.16–163.08 | 65.84 | 23.04 | 504.48 |
| native16k / p12k | 6 | 168.17 | 167.90–168.44 | 385.26 | 22.68 | 406.14 |
| native2k / p12k | 6 | 129.86 | 127.90–131.88 | 400.32 | 30.12 | 2351.86 |
| old-work256 / p12k | 6 | 163.07 | 161.93–164.23 | 376.65 | 23.59 | 440.12 |
| guard / p12k | 6 | 162.22 | 161.69–162.74 | 67.47 | 23.03 | 521.85 |
| native16k / p12k-burst | 6 | 152.31 | 151.82–152.80 | 397.86 | 25.52 | 415.52 |
| native2k / p12k-burst | 6 | 86.72 | 84.84–88.69 | 413.55 | 47.33 | 5386.11 |
| old-work256 / p12k-burst | 6 | 149.05 | 147.56–150.58 | 376.45 | 26.34 | 444.95 |
| guard / p12k-burst | 6 | 149.33 | 146.94–151.79 | 68.19 | 25.82 | 522.30 |

| Role / workload | D SLO 100/200 ms | P SLO 1/2 s | Joint windows 100 ms + 1 s | Joint-window goodput req/s |
| --- | --- | --- | ---: | ---: |
| native16k / decode-control | 18/24; 18/24 | 0/0; 0/0 | 0/6 | 0.000 |
| native2k / decode-control | 18/24; 18/24 | 0/0; 0/0 | 0/6 | 0.000 |
| old-work256 / decode-control | 18/24; 18/24 | 0/0; 0/0 | 0/6 | 0.000 |
| guard / decode-control | 24/24; 24/24 | 0/0; 0/0 | 6/6 | 0.660 |
| native16k / p8k | 0/24; 0/24 | 6/6; 6/6 | 0/6 | 0.000 |
| native2k / p8k | 0/24; 0/24 | 0/6; 6/6 | 0/6 | 0.000 |
| old-work256 / p8k | 0/24; 0/24 | 6/6; 6/6 | 0/6 | 0.000 |
| guard / p8k | 24/24; 24/24 | 6/6; 6/6 | 6/6 | 0.788 |
| native16k / p12k | 0/24; 0/24 | 6/6; 6/6 | 0/6 | 0.000 |
| native2k / p12k | 0/24; 0/24 | 0/6; 0/6 | 0/6 | 0.000 |
| old-work256 / p12k | 0/24; 0/24 | 6/6; 6/6 | 0/6 | 0.000 |
| guard / p12k | 24/24; 24/24 | 6/6; 6/6 | 6/6 | 0.786 |
| native16k / p12k-burst | 0/24; 0/24 | 18/18; 18/18 | 0/6 | 0.000 |
| native2k / p12k-burst | 0/24; 0/24 | 0/18; 0/18 | 0/6 | 0.000 |
| old-work256 / p12k-burst | 0/24; 3/24 | 18/18; 18/18 | 0/6 | 0.000 |
| guard / p12k-burst | 24/24; 24/24 | 18/18; 18/18 | 6/6 | 0.997 |

Final rates use total generated tokens or qualified requests divided by the sum of all six measured window durations. Percentiles pool individual requests, not stage percentiles. JSON retains all four D/P SLO combinations and phase-specific goodputs. Missing stages produce no final aggregate; mismatched plans or producers fail validation.

Each request's overall D SLO includes startup gaps. Post-gate and P-wait diagnostics are retained separately in JSON; boundary-crossing gaps keep their full duration and are never clipped. A P's observed wait windows may overlap, so those gaps are not pooled as independent observations.

The active-D list is recorded just before create_task, whereas transport start is recorded in the coroutine. The audit retains the reported list and independently reconstructs the active set at actual transport start, requiring at least one active D and subsequent D token arrivals.

Token-event timestamps are client byte-arrival observations. Multiple SSE frames can share one read timestamp; multi-token events have no fabricated internal token times. These results are not device-overlap evidence, semantic answer correctness, or stationary capacity/SLO guarantees.

Excluded/unfinalized files: 5. Failures: 0. Input, manifest, producer-source, and audit-script SHA256 values are in JSON.

## Retained unselected stages

- guard-initial: Guard v1 retained separately; final guard comparison selects v2.
- old-work256-confirm: CPU regression overlapped serving readiness; entire four-workload stage retained and replaced by clean confirmation. Exclusion was decided from timestamps before inspecting metrics.

## Synthetic generated-token agreement

Descriptive generated-token exactness only, with no pass threshold or semantic oracle. Raw SSE arrays were independently reconstructed and compared before retaining compact hashes/counts. Inputs match workload, phase/index, prompt and complete non-prompt payload. Warmup and measured repeats are separate. Paired repeat exactness matches initial or confirmation slots; missing reference slots are not counted as mismatches. Stage-pair exactness compares initial and confirmation at matching repeat indices within each role. Pairwise exactness compares every observation to all available matched native16k observations. Guard v1/v2 stay separate; unselected stages remain per-stage only. Native graph output variation remains an open issue (#44); these counts assign no cause or establish semantic correctness, even when transport validation passes.

A one-observation warmup group is trivially identical, not a repeated stability measurement. JSON retains observation counts, sequence hashes, native-reference variability and the first paired mismatch for each input.

| Stage | Scope | Workload | Full / first-token stable inputs | Paired native16k full exact | Matches any native16k sequence |
| --- | --- | --- | --- | --- | --- |
| guard-initial | repeats | decode-control | 0/4; 2/4 | 2/12 | 2/12 |
| guard-initial | warmup | decode-control | 4/4; 4/4 | 0/4 | 0/4 |
| guard-initial | repeats | p12k | 2/5; 5/5 | 6/15 | 6/15 |
| guard-initial | warmup | p12k | 5/5; 5/5 | 1/5 | 1/5 |
| guard-initial | repeats | p12k-burst | 3/7; 5/7 | 12/21 | 12/21 |
| guard-initial | warmup | p12k-burst | 7/7; 7/7 | 4/7 | 4/7 |
| guard-initial | repeats | p8k | 2/5; 3/5 | 5/15 | 6/15 |
| guard-initial | warmup | p8k | 5/5; 5/5 | 1/5 | 1/5 |
| guard-v2-confirm | repeats | decode-control | 1/4; 3/4 | 3/12 | 3/12 |
| guard-v2-confirm | warmup | decode-control | 4/4; 4/4 | 1/4 | 1/4 |
| guard-v2-confirm | repeats | p12k | 1/5; 4/5 | 5/15 | 5/15 |
| guard-v2-confirm | warmup | p12k | 5/5; 5/5 | 2/5 | 2/5 |
| guard-v2-confirm | repeats | p12k-burst | 2/7; 6/7 | 10/21 | 10/21 |
| guard-v2-confirm | warmup | p12k-burst | 7/7; 7/7 | 2/7 | 4/7 |
| guard-v2-confirm | repeats | p8k | 1/5; 3/5 | 4/15 | 5/15 |
| guard-v2-confirm | warmup | p8k | 5/5; 5/5 | 1/5 | 1/5 |
| guard-v2-initial | repeats | decode-control | 0/4; 3/4 | 2/12 | 2/12 |
| guard-v2-initial | warmup | decode-control | 4/4; 4/4 | 1/4 | 1/4 |
| guard-v2-initial | repeats | p12k | 2/5; 4/5 | 6/15 | 6/15 |
| guard-v2-initial | warmup | p12k | 5/5; 5/5 | 2/5 | 2/5 |
| guard-v2-initial | repeats | p12k-burst | 3/7; 6/7 | 9/21 | 11/21 |
| guard-v2-initial | warmup | p12k-burst | 7/7; 7/7 | 4/7 | 4/7 |
| guard-v2-initial | repeats | p8k | 2/5; 3/5 | 5/15 | 6/15 |
| guard-v2-initial | warmup | p8k | 5/5; 5/5 | 2/5 | 2/5 |
| native16k-confirm | repeats | decode-control | 1/4; 2/4 | reference | 12/12 |
| native16k-confirm | warmup | decode-control | 4/4; 4/4 | reference | 4/4 |
| native16k-confirm | repeats | p12k | 1/5; 4/5 | reference | 15/15 |
| native16k-confirm | warmup | p12k | 5/5; 5/5 | reference | 5/5 |
| native16k-confirm | repeats | p12k-burst | 3/7; 5/7 | reference | 21/21 |
| native16k-confirm | warmup | p12k-burst | 7/7; 7/7 | reference | 7/7 |
| native16k-confirm | repeats | p8k | 1/5; 4/5 | reference | 15/15 |
| native16k-confirm | warmup | p8k | 5/5; 5/5 | reference | 5/5 |
| native16k-initial | repeats | decode-control | 1/4; 3/4 | reference | 12/12 |
| native16k-initial | warmup | decode-control | 4/4; 4/4 | reference | 4/4 |
| native16k-initial | repeats | p12k | 2/5; 4/5 | reference | 15/15 |
| native16k-initial | warmup | p12k | 5/5; 5/5 | reference | 5/5 |
| native16k-initial | repeats | p12k-burst | 3/7; 6/7 | reference | 21/21 |
| native16k-initial | warmup | p12k-burst | 7/7; 7/7 | reference | 7/7 |
| native16k-initial | repeats | p8k | 1/5; 4/5 | reference | 15/15 |
| native16k-initial | warmup | p8k | 5/5; 5/5 | reference | 5/5 |
| native2k-confirm | repeats | decode-control | 1/4; 3/4 | 3/12 | 3/12 |
| native2k-confirm | warmup | decode-control | 4/4; 4/4 | 1/4 | 1/4 |
| native2k-confirm | repeats | p12k | 1/5; 3/5 | 4/15 | 5/15 |
| native2k-confirm | warmup | p12k | 5/5; 5/5 | 2/5 | 2/5 |
| native2k-confirm | repeats | p12k-burst | 2/7; 7/7 | 9/21 | 10/21 |
| native2k-confirm | warmup | p12k-burst | 7/7; 7/7 | 3/7 | 4/7 |
| native2k-confirm | repeats | p8k | 2/5; 4/5 | 5/15 | 7/15 |
| native2k-confirm | warmup | p8k | 5/5; 5/5 | 2/5 | 2/5 |
| native2k-initial | repeats | decode-control | 1/4; 3/4 | 3/12 | 4/12 |
| native2k-initial | warmup | decode-control | 4/4; 4/4 | 1/4 | 1/4 |
| native2k-initial | repeats | p12k | 1/5; 5/5 | 5/15 | 5/15 |
| native2k-initial | warmup | p12k | 5/5; 5/5 | 2/5 | 2/5 |
| native2k-initial | repeats | p12k-burst | 2/7; 6/7 | 10/21 | 11/21 |
| native2k-initial | warmup | p12k-burst | 7/7; 7/7 | 4/7 | 4/7 |
| native2k-initial | repeats | p8k | 2/5; 3/5 | 5/15 | 6/15 |
| native2k-initial | warmup | p8k | 5/5; 5/5 | 1/5 | 1/5 |
| old-work256-confirm | repeats | decode-control | 0/4; 2/4 | 1/12 | 1/12 |
| old-work256-confirm | warmup | decode-control | 4/4; 4/4 | 1/4 | 1/4 |
| old-work256-confirm | repeats | p12k | 2/5; 3/5 | 5/15 | 6/15 |
| old-work256-confirm | warmup | p12k | 5/5; 5/5 | 2/5 | 2/5 |
| old-work256-confirm | repeats | p12k-burst | 2/7; 6/7 | 9/21 | 10/21 |
| old-work256-confirm | warmup | p12k-burst | 7/7; 7/7 | 2/7 | 4/7 |
| old-work256-confirm | repeats | p8k | 1/5; 3/5 | 4/15 | 5/15 |
| old-work256-confirm | warmup | p8k | 5/5; 5/5 | 1/5 | 1/5 |
| old-work256-confirm-clean | repeats | decode-control | 0/4; 3/4 | 2/12 | 2/12 |
| old-work256-confirm-clean | warmup | decode-control | 4/4; 4/4 | 1/4 | 1/4 |
| old-work256-confirm-clean | repeats | p12k | 0/5; 2/5 | 4/15 | 4/15 |
| old-work256-confirm-clean | warmup | p12k | 5/5; 5/5 | 2/5 | 2/5 |
| old-work256-confirm-clean | repeats | p12k-burst | 4/7; 6/7 | 11/21 | 12/21 |
| old-work256-confirm-clean | warmup | p12k-burst | 7/7; 7/7 | 3/7 | 4/7 |
| old-work256-confirm-clean | repeats | p8k | 1/5; 5/5 | 4/15 | 5/15 |
| old-work256-confirm-clean | warmup | p8k | 5/5; 5/5 | 2/5 | 2/5 |
| old-work256-initial | repeats | decode-control | 0/4; 2/4 | 1/12 | 1/12 |
| old-work256-initial | warmup | decode-control | 4/4; 4/4 | 1/4 | 1/4 |
| old-work256-initial | repeats | p12k | 0/5; 3/5 | 4/15 | 4/15 |
| old-work256-initial | warmup | p12k | 5/5; 5/5 | 1/5 | 1/5 |
| old-work256-initial | repeats | p12k-burst | 2/7; 5/7 | 10/21 | 11/21 |
| old-work256-initial | warmup | p12k-burst | 7/7; 7/7 | 4/7 | 4/7 |
| old-work256-initial | repeats | p8k | 1/5; 4/5 | 4/15 | 6/15 |
| old-work256-initial | warmup | p8k | 5/5; 5/5 | 2/5 | 2/5 |

| Selected role | Scope | Workload | Coverage | Stable full inputs | Initial/confirm full exact | Initial/confirm first-token exact |
| --- | --- | --- | --- | --- | --- | --- |
| native16k | repeats | decode-control | complete | 1/4 | 3/12 | 9/12 |
| native16k | warmup | decode-control | complete | 1/4 | 1/4 | 2/4 |
| native16k | repeats | p12k | complete | 1/5 | 5/15 | 12/15 |
| native16k | warmup | p12k | complete | 2/5 | 2/5 | 3/5 |
| native16k | repeats | p12k-burst | complete | 3/7 | 11/21 | 17/21 |
| native16k | warmup | p12k-burst | complete | 2/7 | 2/7 | 6/7 |
| native16k | repeats | p8k | complete | 1/5 | 4/15 | 10/15 |
| native16k | warmup | p8k | complete | 2/5 | 2/5 | 4/5 |
| native2k | repeats | decode-control | complete | 1/4 | 3/12 | 11/12 |
| native2k | warmup | decode-control | complete | 1/4 | 1/4 | 3/4 |
| native2k | repeats | p12k | complete | 1/5 | 5/15 | 13/15 |
| native2k | warmup | p12k | complete | 2/5 | 2/5 | 4/5 |
| native2k | repeats | p12k-burst | complete | 1/7 | 8/21 | 20/21 |
| native2k | warmup | p12k-burst | complete | 3/7 | 3/7 | 6/7 |
| native2k | repeats | p8k | complete | 2/5 | 6/15 | 14/15 |
| native2k | warmup | p8k | complete | 1/5 | 1/5 | 4/5 |
| old-work256 | repeats | decode-control | complete | 0/4 | 1/12 | 8/12 |
| old-work256 | warmup | decode-control | complete | 1/4 | 1/4 | 2/4 |
| old-work256 | repeats | p12k | complete | 0/5 | 2/15 | 9/15 |
| old-work256 | warmup | p12k | complete | 1/5 | 1/5 | 4/5 |
| old-work256 | repeats | p12k-burst | complete | 2/7 | 10/21 | 18/21 |
| old-work256 | warmup | p12k-burst | complete | 3/7 | 3/7 | 7/7 |
| old-work256 | repeats | p8k | complete | 0/5 | 4/15 | 12/15 |
| old-work256 | warmup | p8k | complete | 2/5 | 2/5 | 3/5 |
| guard | repeats | decode-control | complete | 0/4 | 2/12 | 10/12 |
| guard | warmup | decode-control | complete | 1/4 | 1/4 | 3/4 |
| guard | repeats | p12k | complete | 1/5 | 5/15 | 10/15 |
| guard | warmup | p12k | complete | 2/5 | 2/5 | 5/5 |
| guard | repeats | p12k-burst | complete | 2/7 | 8/21 | 19/21 |
| guard | warmup | p12k-burst | complete | 4/7 | 4/7 | 5/7 |
| guard | repeats | p8k | complete | 1/5 | 5/15 | 12/15 |
| guard | warmup | p8k | complete | 1/5 | 1/5 | 5/5 |
