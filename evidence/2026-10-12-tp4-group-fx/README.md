# TP4 grouped-prefill validation evidence

[Report](../../docs/validation-tp4-prefill-group-fx.md) ·
[Experimental PR #10](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/pull/10) ·
[Numerical follow-up #11](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/11)

These are finite functional diagnostics on two TP4 groups, with no throughput
claim. Both roles use runtime `f09a100ade651936b3efeaf045ec5ce63d0b38fe` and the
same native compiler backend. Grouped P differs from raw P only by the experimental
compiler policy, apart from isolated service/card/cache assignments.

- [Independent raw numeric replay](numeric19-independent-review.md): 19/19
  complete per role, 9/19 exact cross-role sequences; unresolved early D differences.
- [Early-divergence location](numeric-early-divergence-location.md): the observed
  fourth-token difference precedes the later 12K request and occurs during D1/P384.
- [Actual startup artifacts](candidate-startup-artifact-audit.md): ten four-layer
  groups per rank and separate full D backbones, with exact file/source hashes.
- [Four-rank serving returns](candidate-b-rank-returns.json): complete group
  sequences and the 8-layer cancellation prefix, bounded to the ten lifecycle requests.

The complete raw package is attached to
[the evidence release](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/releases/tag/evidence-tp4-groupfx-20261012).
`archive-index.json` records its checksum and every retained member. It includes
all 80 admitted requests (79 complete, one planned cancellation), raw SSE and
journals, immutable plans and registrations, service logs and traces, source
snapshots, generated compiler artifacts, startup/stop receipts, audit tools and
CPU-check receipts. Failed initial tooling checks are preserved.

The archive contains 348 files (138,926,948 uncompressed bytes; 15,707,721 bytes
compressed), SHA256
`a931a7329ad534f03fb7d05e7845b43125c19637b7daf33b146dd9aeebcfe951`.
Every member passed extraction/hash verification. A second directory's
[portable numeric replay](portable-replay-verification.json) read all seven
required artifacts exclusively from that archive and reproduced the original
19/19 transport checks and 9/19 exact-sequence result.

The archive's `workspace/` tree preserves the captured `/root/workspace/` layout.
The original reports retain those absolute paths and their original SHA values.
Portable numeric replay instructions are included in the archive; other original
tools can be replayed in an isolated environment with `workspace/` mounted at
`/root/workspace`. This mount instruction is for offline evidence inspection,
not authorization to start the archived services.

The independent raw audit validates finite transport and observation history.
It does not establish numerical equivalence, compiler semantics, actual opaque
HCCL routing, all KV-pressure cases, kernel overlap, or performance. Outbound
request headers were not captured as wire bytes; their construction is supported
by the frozen producer, while body/response-ID agreement is checked from raw SSE.
