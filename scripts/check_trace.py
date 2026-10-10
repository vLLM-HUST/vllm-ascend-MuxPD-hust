"""Check full-prefill and phase invariants from a scheduler JSONL receipt."""

import argparse
import collections
import json
from pathlib import Path


def check(paths, require_long_prefill=2048):
    phases = collections.Counter()
    max_prefill = 0
    completed = set()
    scheduled = set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            key = (row["pid"], row["step"])
            if row["event"] == "complete":
                completed.add(key)
                continue
            if row["event"] != "schedule" or not row["scheduled_tokens"]:
                continue
            scheduled.add(key)
            phases[row["phase"]] += 1
            if row["scheduled_tokens"] != row["expected_tokens"]:
                raise AssertionError(f"Partial prompt in step {key}")
            kinds = set(row["request_phases"].values())
            if row["phase"] != "multiplex" and len(kinds) != 1:
                raise AssertionError(f"Mixed model batch in step {key}")
            for rid, count in row["scheduled_tokens"].items():
                if row["request_phases"][rid] == "decode":
                    if count != 1:
                        raise AssertionError(f"Non-uniform decode in step {key}")
                else:
                    max_prefill = max(max_prefill, count)
    if not scheduled or scheduled - completed:
        raise AssertionError("Missing completed scheduler transactions")
    if max_prefill <= require_long_prefill:
        raise AssertionError(f"No complete prefill > {require_long_prefill} tokens")
    return {
        "passed": True,
        "transactions": len(scheduled),
        "phases": dict(phases),
        "max_complete_prefill_tokens": max_prefill,
        "scope": "scheduler counts; device overlap requires a profiler timeline",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--require-long-prefill", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(check(args.paths, args.require_long_prefill), indent=2))
