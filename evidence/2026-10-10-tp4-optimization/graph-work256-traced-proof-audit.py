"""Extract the completed work256 lifecycle diagnostic; stdlib, no device access."""

import hashlib
import json
from pathlib import Path

BASE = Path("local-results/perf-tp4")
TRACE = BASE / "graph-work256-traced-817463.jsonl"


def source(path):
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main():
    raw = TRACE.read_text().splitlines()
    rows = [json.loads(line) for line in raw]
    by_step = {}
    for number, row in enumerate(rows, 1):
        key = row["pid"], row["step"]
        events = by_step.setdefault(key, {})
        assert row["event"] not in events
        events[row["event"]] = (number, row)
    assert len(by_step) == 466
    assert all(
        set(events) == {"schedule", "decoupled_schedule", "decoupled_complete"}
        for events in by_step.values()
    )
    pid = 817463
    selected = set()

    def get(step, event):
        return by_step[pid, step][event][1]

    def keep(steps):
        line_numbers = []
        for step in steps:
            for number, _ in by_step[pid, step].values():
                selected.add(number)
                line_numbers.append(number)
        return sorted(line_numbers)

    complete_ids = []
    cancellation_ids = []
    for (row_pid, step), events in by_step.items():
        schedule = events["decoupled_schedule"][1]
        work = (schedule.get("quota") or {}).get("work_quota") or {}
        if (
            schedule["new_prefill_ids"]
            and schedule["decode_ids"]
            and work.get("cohort_full_prefill_tokens") == 12000
        ):
            assert row_pid == pid
            receipt = get(step, "decoupled_complete")
            next_schedule = get(step + 1, "decoupled_schedule")
            next_receipt = get(step + 1, "decoupled_complete")
            prefill_ids = schedule["new_prefill_ids"]
            assert len(prefill_ids) == 1
            prefill_id = prefill_ids[0]
            native = get(step, "schedule")
            assert native["scheduled_tokens"][prefill_id] == 12000
            assert native["expected_tokens"][prefill_id] == 12000
            assert native["computed_before"][prefill_id] == 0
            assert schedule["layers_per_slice"] == receipt["layers_completed"] == 21
            assert schedule["layers_completed_before"] == 0
            assert not receipt["completed_prefill_ids"]
            assert next_schedule["cohort_id"] == schedule["cohort_id"]
            assert next_schedule["layers_completed_before"] == 21
            assert next_schedule["layers_per_slice"] == 19
            assert not next_schedule["new_prefill_ids"]
            assert prefill_id not in get(step + 1, "schedule")["scheduled_tokens"]
            assert receipt["layers_total"] == next_receipt["layers_total"] == 40
            assert (
                receipt["progress_source"]
                == next_receipt["progress_source"]
                == "layer_hooks"
            )
            record = {
                "cohort_id": schedule["cohort_id"],
                "request_id": prefill_id,
                "steps": [step, step + 1],
                "full_prompt_tokens": 12000,
                "scheduled_layer_quotas": [21, 19],
                "source_lines": keep([step, step + 1]),
            }
            if next_receipt["cancelled_prefill_ids"]:
                assert next_schedule["cancelled_prefill_ids"] == [prefill_id]
                assert next_receipt["cancelled_prefill_ids"] == [prefill_id]
                assert not next_receipt["completed_prefill_ids"]
                assert next_receipt["layers_completed"] == 21
                assert get(step + 2, "decoupled_schedule")["cohort_id"] is None
                assert get(step + 2, "decoupled_complete")["progress_source"] == "none"
                record.update(
                    {
                        "completed_layers": [21, 21],
                        "actual_layer_deltas": [21, 0],
                        "cancel_ack": True,
                        "next_step_cohort_cleared": True,
                        "source_lines": keep([step, step + 1, step + 2]),
                    }
                )
                cancellation_ids.append(record)
            else:
                assert next_receipt["completed_prefill_ids"] == [prefill_id]
                assert next_receipt["layers_completed"] == 40
                assert next_schedule["finish_prefill"]
                record.update(
                    {"completed_layers": [21, 40], "actual_layer_deltas": [21, 19]}
                )
                complete_ids.append(record)
    assert len(complete_ids) == len(cancellation_ids) == 1

    debt_pairs = []
    for (row_pid, step), events in by_step.items():
        schedule = events["decoupled_schedule"][1]
        if schedule["admission"]["reason"] != "token_budget_prefill_first":
            continue
        assert row_pid == pid
        deferred = set(schedule["admission"]["deferred_decode_ids"])
        assert deferred and not schedule["decode_ids"]
        assert not deferred.intersection(get(step, "schedule")["scheduled_tokens"])
        assert set(get(step, "decoupled_complete")["decode_debt_ids"]) == deferred
        repay = get(step + 1, "decoupled_schedule")
        assert repay["admission"]["reason"] == "repay_decode"
        assert deferred <= set(repay["decode_ids"])
        assert not get(step + 1, "decoupled_complete")["decode_debt_ids"]
        counters = {}
        for req_id in sorted(deferred):
            previous = get(step - 1, "schedule")
            next_native = get(step + 1, "schedule")
            before = previous["computed_before"][req_id]
            increment = previous["scheduled_tokens"][req_id]
            after = next_native["computed_before"][req_id]
            assert increment > 0 and next_native["scheduled_tokens"][req_id] == 1
            assert after == before + increment
            counters[req_id] = {
                "previous_computed_before": before,
                "previous_scheduled_tokens": increment,
                "computed_before_repayment": after,
                "extra_advance_during_prefill_only_step": 0,
            }
        debt_pairs.append(
            {
                "prefill_first_step": step,
                "repay_decode_step": step + 1,
                "deferred_ids": sorted(deferred),
                "computed_token_check": counters,
                "source_lines": keep([step - 1, step, step + 1]),
            }
        )
    assert len(debt_pairs) == 4

    names = [
        "graph-work256-traced-manifest.json",
        "graph-work256-traced-calibration.json",
        "graph-work256-traced-progress.json",
        "graph-work256-traced-cancel.json",
        "graph-work256-traced-check.json",
        "native-graph-vs-graph-work256-traced.json",
        "validation-final.json",
    ]
    report = {
        "schema_version": 1,
        "passed": True,
        "scope": "Host scheduler/worker progress and lifecycle protocol; not kernel overlap, KV contents, or performance evidence.",
        "runtime_commit": "4caeeba",
        "trace": source(TRACE),
        "trace_lines": len(raw),
        "inputs": [source(BASE / name) for name in names],
        "audit_script": source(Path(__file__)),
        "reproduce": "python local-results/perf-tp4/graph-work256-traced-proof-audit.py",
        "diagnostic_preparation": "Separate trace-on process, same frozen runtime source as timed work256. One 12000-token P request plus an 8-token-output warmup calibrated pure-P latency before these probes. Calibration and all diagnostics are excluded from performance aggregates.",
        "selection": "All fresh 12000-token cohorts admitted alongside decode, and every token_budget_prefill_first/next-step repay_decode pair. Original lines are UTF-8 text without the terminal newline; numbers are 1-based.",
        "completed_long_prefill": complete_ids,
        "cancelled_long_prefill": cancellation_ids,
        "prefill_first_decode_debt_pairs": debt_pairs,
        "checker_summary": json.loads(
            (BASE / "graph-work256-traced-check.json").read_text()
        ),
        "original_trace_lines": [
            {"line_number": n, "raw_line": raw[n - 1]} for n in sorted(selected)
        ],
    }
    output = BASE / "graph-work256-traced-proof.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"{output}: passed; 1 completed/1 cancelled 12K cohort, 4 debt pairs, {len(selected)} original lines"
    )


if __name__ == "__main__":
    main()
