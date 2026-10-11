#!/usr/bin/env python3
"""Offline decomposition of the frozen guard-v1 diagnostic trace; no model calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(data):
    return hashlib.sha256(data).hexdigest()


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "nearest_rank_p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max_ms": ordered[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path(__file__).with_name(
        "guard-diagnostic-trace-189423.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    transactions = defaultdict(dict)
    for line, text in enumerate(raw.splitlines(), 1):
        row = json.loads(text)
        event = row["event"]
        if event in transactions[row["step"]]:
            raise ValueError(f"duplicate event at line {line}")
        transactions[row["step"]][event] = (line, row)
    required = {"schedule", "decoupled_schedule", "decoupled_complete"}
    unpaired = sorted(step for step, rows in transactions.items() if set(rows) != required)
    solo = defaultdict(list)
    residual = defaultdict(list)
    cohorts = {}
    for step, rows in sorted(transactions.items()):
        if set(rows) != required:
            continue
        schedule_line, schedule = rows["decoupled_schedule"]
        complete_line, complete = rows["decoupled_complete"]
        guard = schedule.get("quota", {}).get("decode_budget", {})
        timing = complete.get("prefill_slice_timing")
        sample = complete.get("decode_budget_sample") or {}
        base = {
            "step": step,
            "schedule_line": schedule_line,
            "complete_line": complete_line,
            "host_step_ms": complete["duration_ms"],
        }
        if sample.get("phase") == "decode":
            shape = tuple(sample["shape"])
            solo[shape].append(dict(base, source="completed_solo_decode_host_round_trip"))
        cohort_id = schedule.get("cohort_id")
        if cohort_id is None:
            continue
        cohort = cohorts.setdefault(cohort_id, {
            "cohort_id": cohort_id,
            "prefill_shape_tokens": guard.get("prefill_shape_tokens"),
            "pending_prefill_ids": schedule["pending_prefill_ids"],
            "steps": [],
        })
        before = schedule["layers_completed_before"]
        after = complete["layers_completed"]
        total = complete["layers_total"]
        position = "first_and_final" if before == 0 and after == total else (
            "first" if before == 0 else "final" if after == total else "continuation")
        record = dict(base,
            slice_position=position,
            layers_before=before,
            layers_after=after,
            layer_delta=after-before,
            quota=schedule["layers_per_slice"],
            guard_mode=guard.get("mode"),
            cold_reasons=guard.get("cold_reasons", []),
            estimated_solo_decode_ms=guard.get("estimated_solo_decode_ms"),
            estimated_raw_layer_ms=guard.get("estimated_raw_layer_ms"),
            selected_decode_count=len(schedule["decode_ids"]),
            active_decode_count=len(guard.get("active_decode_ids", [])),
            progress_source=complete["progress_source"],
            completed_prefill_ids=complete["completed_prefill_ids"],
            cancelled_prefill_ids=complete["cancelled_prefill_ids"],
            observed_guard_sample=sample,
            timing=timing,
        )
        cohort["steps"].append(record)
        eligible = (
            timing is not None
            and timing["device_complete"]
            and position == "continuation"
            and after > before
            and complete["progress_source"] == "layer_hooks"
            and timing["mode"] == "raw_layers"
            and schedule["decode_ids"]
            and not schedule["pause_prefill"]
            and not schedule["cancelled_prefill_ids"]
            and not complete["cancelled_prefill_ids"]
            and sample.get("phase") == "raw_prefill_layer"
        )
        if eligible:
            value = complete["duration_ms"] - timing["device_elapsed_ms"]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid residual step {step}: {value}")
            shape = tuple(guard["decode_shape"])
            residual[shape].append(dict(base,
                source="hypothetical_fenced_host_step_minus_driver_prefill_device_span",
                cohort_id=cohort_id,
                prefill_device_elapsed_ms=timing["device_elapsed_ms"],
                prefill_host_submit_ms=timing["host_submit_ms"],
                residual_ms=value,
                layer_delta=after-before,
            ))

    def summarize_buckets(buckets, key):
        return [{
            "decode_shape": list(shape),
            "distribution": distribution([row[key] for row in samples]),
            "maximum_sample": max(samples, key=lambda row: row[key]),
            "all_samples": samples,
        } for shape, samples in sorted(buckets.items())]

    for cohort in cohorts.values():
        steps = cohort["steps"]
        cohort["step_count"] = len(steps)
        cohort["quota_counts"] = dict(Counter(str(row["quota"]) for row in steps))
        cohort["mode_counts"] = dict(Counter(row["guard_mode"] for row in steps))
        cohort["cold_reason_counts"] = dict(Counter(
            reason for row in steps for reason in row["cold_reasons"]))
        cohort["raw_timing_by_position"] = {}
        for position in ("first", "continuation", "final", "first_and_final"):
            samples = [row for row in steps if row["slice_position"] == position
                       and row["timing"] and row["timing"]["device_complete"]]
            cohort["raw_timing_by_position"][position] = {
                "count": len(samples),
                "host_submit": distribution([row["timing"]["host_submit_ms"]
                                             for row in samples]),
                "device_elapsed": distribution([row["timing"]["device_elapsed_ms"]
                                                for row in samples]),
                "max_host_device_span_sample": max(samples, key=lambda row: max(
                    row["timing"]["host_submit_ms"], row["timing"]["device_elapsed_ms"]
                )) if samples else None,
            }
    result = {
        "schema_version": 1,
        "producer": {"file": Path(__file__).name,
                     "sha256": digest(Path(__file__).read_bytes())},
        "trace": {"file": args.trace.name, "bytes": len(raw), "sha256": digest(raw)},
        "paired_steps": len(transactions)-len(unpaired),
        "excluded_unpaired_steps": unpaired,
        "measurement_contract": {
            "diagnostic_only": True,
            "runtime_changed": False,
            "p95": "sorted samples[ceil(0.95*n)-1]",
            "residual_formula": "completed host step duration_ms - driver-rank P device_elapsed_ms",
            "residual_eligibility": "completed nonfirst/nonfinal raw P slice with selected D; no pause/cancel; valid raw-prefill learning receipt",
            "limitations": [
                "Residual is an offline candidate estimator, not used by guard v1.",
                "Driver-rank device event is not an all-rank maximum or pure kernel time.",
                "Residual includes D, RPC, host work, and cross-rank tail; it is a heuristic, not a hard bound.",
                "Full-step wall intervals and device elapsed intervals use different clocks; subtraction is a duration heuristic.",
                "No counterfactual latency or throughput claim is made from changing quotas offline.",
                "This diagnostic stage order differs from timed stage order; it does not prove the timed cold-state history.",
            ],
        },
        "solo_decode": summarize_buckets(solo, "host_step_ms"),
        "candidate_continuation_residual": summarize_buckets(residual, "residual_ms"),
        "cohorts": list(cohorts.values()),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output.read_bytes()),
                      "paired_steps": result["paired_steps"], "cohorts": len(cohorts),
                      "solo_samples": sum(map(len, solo.values())),
                      "residual_samples": sum(map(len, residual.values())),
                      "unpaired_steps": unpaired}, indent=2))


if __name__ == "__main__":
    main()
