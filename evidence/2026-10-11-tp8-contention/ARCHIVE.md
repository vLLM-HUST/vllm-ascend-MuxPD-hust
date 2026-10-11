# Replaying the TP8 contention archive

`archive-index.json` records source, stored and decompressed SHA-256 and byte
counts. Unchanged artifacts have identical source/decompressed SHA. Gzip uses
an empty filename, mtime=0 and level 6; the Python/zlib versions are in the index.
The index is not a cryptographic signature; pin the Git commit when citing it.

All measured stages, including guard v1 and the excluded CPU-overlap old stage,
are retained. Selection/exclusion decisions belong to the plans and final audit;
publication is not evidence that every included measurement is a final candidate.
Client warmups, failures and raw SSE are not removed. All 16 rank trace exports
and kernel CSVs are included with their original relative profiler directories.
No profiler database/binary is needed by the archived offline classifier.

`environment-initial.json` omits the original `processes` field, which contained
unrelated system-management commands and connection-token arguments. The local
original is retained. The index explicitly identifies this transformation and
both hashes: the public snapshot is not presented as its byte-identical original.
Administrative PID/control files, PR drafts, caches and temporary files are
excluded. Other selected raw data is byte-exact; inferred token fields such as
`token_ids` and `max_tokens` are never treated as credentials or removed.

The three external CPU logs referenced by source-validation.json are copied
from the parent local-results directory into validation/. Their exact origin
mapping is indexed. The early decode-budget-v2-tests.log is retained alongside
the later regression/full-suite passes; a failed test invocation is not removed
because a subsequent corrected fixture passes. Consult source-validation.json
and the logs for the command, failure and later validation scope.

From the repository root, after timing/CPU tests have finished:

```sh
python evidence/2026-10-11-tp8-contention/archive_evidence.py verify
python evidence/2026-10-11-tp8-contention/archive_evidence.py unpack \
  --output local-results/tp8-contention-replay
python local-results/tp8-contention-replay/profile-hotspots-audit.py \
  --output local-results/tp8-contention-replay/replayed-profile-hotspots.json
```

The unpack destination must not exist. Unpack verifies stored and decompressed
hashes before atomically publishing the restored tree. Use a directory immediately
inside this repository's `local-results` so the profile auditor's repo-root import
of `scripts/analyze_npu_trace.py` resolves. Its historical report source paths can
change after replay; compare metrics and source hashes, not absolute path strings.
Check the indexed classifier SHA against the checked-out script before replay.
Auditing/profiling can be CPU intensive and must not overlap inference timing.
No archived launcher is executed by verification or unpacking.
