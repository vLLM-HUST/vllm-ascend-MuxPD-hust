#!/usr/bin/env python3
"""Audit a completed v2 diagnostic trace without loading torch or contacting APIs.

This reports actual layer progress and estimator arithmetic, not a performance
counterfactual. Run only after the diagnostic driver and trace have finished.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def fingerprint(path):
    raw = path.read_bytes()
    return {
        "file": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "count": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "nearest_rank_p95": values[math.ceil(0.95 * len(values)) - 1],
        "max": values[-1],
    }


def near(actual, expected, label):
    if (
        not isinstance(actual, (int, float))
        or isinstance(actual, bool)
        or not (math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-6))
    ):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def read_trace(path):
    rows = []
    transactions = defaultdict(dict)
    for line, text in enumerate(path.read_bytes().splitlines(), 1):
        row = json.loads(text)
        key = (row["pid"], row["step"])
        if row["event"] in transactions[key]:
            raise AssertionError(f"duplicate event at line {line}")
        transactions[key][row["event"]] = (line, row)
        rows.append((line, row))
    expected = {"schedule", "decoupled_schedule", "decoupled_complete"}
    incomplete = [key for key, rows in transactions.items() if set(rows) != expected]
    if incomplete:
        raise AssertionError(f"unfinished trace transactions: {incomplete}")
    return rows, transactions


def position(before, after, total):
    if before == 0:
        return "first_and_final" if after == total else "first"
    return "final" if after == total else "continuation"


def summarize_cohort(cohort):
    steps = cohort["steps"]
    cohort["step_count"] = len(steps)
    cohort["quota_counts"] = dict(Counter(str(row["quota"]) for row in steps))
    cohort["mode_counts"] = dict(Counter(row["guard_mode"] for row in steps))
    cohort["cold_reason_counts"] = dict(
        Counter(reason for row in steps for reason in row["cold_reasons"])
    )
    cohort["first_measured_step"] = next(
        (row["step"] for row in steps if row["guard_mode"] == "measured_slice_budget"),
        None,
    )
    cohort["first_residual_selected_step"] = next(
        (
            row["step"]
            for row in steps
            if row.get("decode_estimate_source") == "fenced_continuation_residual"
        ),
        None,
    )
    cohort["first_actual_multilayer_step"] = next(
        (
            row["step"]
            for row in steps
            if row["layers_before"] > 0 and row["layer_delta"] > 1
        ),
        None,
    )
    cohort["completed"] = any(row["completed_prefill_ids"] for row in steps)
    cohort["cancelled"] = any(row["cancelled_prefill_ids"] for row in steps)
    cohort["sum_host_step_ms"] = sum(row["host_step_ms"] for row in steps)
    cohort["completed_timing_by_position"] = {}
    for kind in ("first", "continuation", "final", "first_and_final"):
        samples = [
            row
            for row in steps
            if row["slice_position"] == kind
            and row["timing"]
            and row["timing"]["device_complete"]
        ]
        cohort["completed_timing_by_position"][kind] = {
            "host_submit_ms": distribution(
                [row["timing"]["host_submit_ms"] for row in samples]
            ),
            "device_elapsed_ms": distribution(
                [row["timing"]["device_elapsed_ms"] for row in samples]
            ),
        }
    return cohort


def compare_shapes(v1, v2):
    groups = defaultdict(lambda: {"v1_cohorts": [], "v2_cohorts": []})
    for name, source in (("v1", v1["cohorts"]), ("v2", v2)):
        for cohort in source:
            steps = cohort["steps"]
            if not any(row["active_decode_count"] for row in steps):
                continue
            shape = tuple(cohort["prefill_shape_tokens"])
            groups[shape][f"{name}_cohorts"].append(
                {
                    "cohort_id": cohort["cohort_id"],
                    "step_count": len(steps),
                    "quota_counts": cohort["quota_counts"],
                    "mode_counts": cohort["mode_counts"],
                    "cold_reason_counts": cohort["cold_reason_counts"],
                    "completed": any(row["completed_prefill_ids"] for row in steps),
                    "cancelled": any(row["cancelled_prefill_ids"] for row in steps),
                    "sum_host_step_ms": sum(row["host_step_ms"] for row in steps),
                }
            )
    return [
        dict(prefill_shape_tokens=list(shape), **group)
        for shape, group in sorted(groups.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--trace", type=Path)
    parser.add_argument(
        "--v1-audit", type=Path, default=base / "guard-v1-estimator-audit.json"
    )
    parser.add_argument(
        "--checker",
        type=Path,
        default=base.parents[1] / "scripts/check_decoupled_trace.py",
    )
    parser.add_argument(
        "--output", type=Path, default=base / "guard-v2-estimator-audit.json"
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=1.2,
        help="Must match the diagnostic runtime configuration",
    )
    parser.add_argument("--require-quota-increase", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.safety_factor) or args.safety_factor < 1:
        parser.error("safety factor must be finite and >=1")
    if args.trace is None:
        matches = sorted(base.glob("guard-v2-diagnostic-trace-*.jsonl"))
        if len(matches) != 1:
            parser.error(
                f"specify --trace; expected one candidate, found {len(matches)}"
            )
        args.trace = matches[0]
    spec = importlib.util.spec_from_file_location("guard_trace_checker", args.checker)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    protocol = checker.check(
        [args.trace], min_decode_steps=1, require_cancellation=True
    )
    rows, transactions = read_trace(args.trace)
    v1 = json.loads(args.v1_audit.read_text())
    residual_history = {}
    cohorts = {}
    samples = []
    rejected = []
    increases = []
    cold_start_transitions = []
    selected_count = 0
    source_counts = Counter()
    for line, row in rows:
        key = (row["pid"], row["step"])
        if row["event"] == "decoupled_schedule":
            guard = (row.get("quota") or {}).get("decode_budget")
            if guard is None:
                continue
            if guard.get("version") != 2:
                raise AssertionError(f"line {line}: expected guard version 2")
            # This compact audit intentionally does not recreate LRU lookup
            # order. The diagnostic must not evict residual entries; if it does,
            # a full LRU replay is needed before checking lifetime maxima.
            cache = guard["estimate_cache"]
            if cache["evictions"]["decode_residual"]:
                raise AssertionError("Residual eviction requires a full LRU replay")
            shape = guard["decode_shape"]
            history_key = (row["pid"], tuple(shape)) if shape else None
            history = residual_history.get(history_key)
            expected_count = len(history) if history else 0
            if guard["residual_decode_samples"] != expected_count:
                raise AssertionError(
                    f"line {line}: residual count mismatches prior receipts"
                )
            if history:
                maximum = max(sample["residual_ms"] for sample in history)
                near(
                    guard["estimated_decode_residual_ms"],
                    maximum * args.safety_factor,
                    f"line {line}: residual max",
                )
            source = guard["decode_estimate_source"]
            source_counts[str(source)] += 1
            if source == "fenced_continuation_residual":
                if not history or row["layers_completed_before"] <= 0:
                    raise AssertionError(
                        f"line {line}: residual lacks prior continuation"
                    )
                selected_count += 1
                if guard["solo_decode_samples"] == 0:
                    cold_start_transitions.append(
                        {
                            "step": row["step"],
                            "schedule_line": line,
                            "residual_samples": expected_count,
                        }
                    )
        if row["event"] != "decoupled_complete":
            continue
        schedule_line, scheduled = transactions[key]["decoupled_schedule"]
        guard = (scheduled.get("quota") or {}).get("decode_budget")
        if guard is None:
            continue
        timing = row.get("prefill_slice_timing")
        observed = row.get("decode_budget_sample") or {}
        sample = observed.get("decode_residual_sample")
        before, after, total = (
            scheduled["layers_completed_before"],
            row["layers_completed"],
            row["layers_total"],
        )
        sample_eligible = bool(
            timing
            and timing["device_complete"]
            and 0 < before < after < total
            and row["progress_source"] == "layer_hooks"
            and timing["mode"] == "raw_layers"
            and scheduled["decode_ids"]
            and not scheduled["pause_prefill"]
            and not scheduled["cancelled_prefill_ids"]
            and not row["cancelled_prefill_ids"]
            and observed.get("phase") == "raw_prefill_layer"
        )
        if (sample is not None) != sample_eligible:
            raise AssertionError(f"line {line}: residual presence/eligibility mismatch")
        if sample is not None:
            # The shared checker verifies source, selected-D shape, arithmetic,
            # positive acceptance, and cancellation/layer boundaries as well.
            checker._check_decode_residual_sample(row, scheduled)
            evidence = dict(
                sample,
                step=row["step"],
                pid=row["pid"],
                cohort_id=row["cohort_id"],
                schedule_line=schedule_line,
                complete_line=line,
                layers_before=before,
                layers_after=after,
                layer_delta=after - before,
            )
            if sample["accepted"]:
                samples.append(evidence)
                history_key = (row["pid"], tuple(sample["decode_shape"]))
                residual_history.setdefault(history_key, []).append(evidence)
            else:
                rejected.append(evidence)
        cohort_key = (row["pid"], row["cohort_id"])
        cohort = cohorts.setdefault(
            cohort_key,
            {
                "pid": row["pid"],
                "cohort_id": row["cohort_id"],
                "prefill_shape_tokens": guard["prefill_shape_tokens"],
                "steps": [],
            },
        )
        record = {
            "step": row["step"],
            "schedule_line": schedule_line,
            "complete_line": line,
            "layers_before": before,
            "layers_after": after,
            "layer_delta": after - before,
            "quota": scheduled["layers_per_slice"],
            "slice_position": position(before, after, total),
            "guard_mode": guard["mode"],
            "cold_reasons": guard["cold_reasons"],
            "decode_estimate_source": guard["decode_estimate_source"],
            "estimated_solo_decode_ms": guard["estimated_solo_decode_ms"],
            "estimated_decode_cost_ms": guard["estimated_decode_cost_ms"],
            "estimated_raw_layer_ms": guard["estimated_raw_layer_ms"],
            "selected_decode_count": len(scheduled["decode_ids"]),
            "active_decode_count": len(guard["active_decode_ids"]),
            "completed_prefill_ids": row["completed_prefill_ids"],
            "cancelled_prefill_ids": row["cancelled_prefill_ids"],
            "host_step_ms": row["duration_ms"],
            "timing": timing,
            "progress_source": row["progress_source"],
        }
        if (
            before > 0
            and guard["decode_estimate_source"] == "fenced_continuation_residual"
            and guard["calibrated"]
            and not guard["control_step"]
        ):
            solo = guard["estimated_solo_decode_ms"]
            if solo is None:
                solo_quota = 1  # The existing cold fallback, not invented timing.
            else:
                slack = (
                    guard["decode_budget_ms"]
                    - max(guard["decode_age_ms"].values(), default=0)
                    - solo
                )
                solo_quota = min(
                    total - before,
                    guard["desired_layers_per_slice"],
                    max(1, math.floor(max(0, slack) / guard["estimated_raw_layer_ms"])),
                )
            record["same_step_solo_fallback_arithmetic_quota"] = solo_quota
            if after - before > 1 and scheduled["layers_per_slice"] > solo_quota:
                increases.append(
                    {
                        "step": row["step"],
                        "pid": row["pid"],
                        "schedule_line": schedule_line,
                        "complete_line": line,
                        "cohort_id": row["cohort_id"],
                        "actual_layers_advanced": after - before,
                        "same_step_solo_fallback_arithmetic_quota": solo_quota,
                        "schedule": scheduled,
                        "receipt": row,
                    }
                )
        cohort["steps"].append(record)
    summarized = [summarize_cohort(cohort) for cohort in cohorts.values()]
    result = {
        "schema_version": 1,
        "producer": fingerprint(Path(__file__)),
        "checker": fingerprint(args.checker),
        "trace": fingerprint(args.trace),
        "v1_audit": fingerprint(args.v1_audit),
        "v1_trace": v1["trace"],
        "safety_factor_from_command_line": args.safety_factor,
        "protocol_check": protocol,
        "paired_steps": len(transactions),
        "residual_history_verified_without_evictions": True,
        "residual_samples": len(samples),
        "rejected_residual_samples": rejected,
        "residual_sample_distribution_ms": distribution(
            [s["residual_ms"] for s in samples]
        ),
        "all_accepted_residual_samples": samples,
        "decode_estimate_source_counts": dict(source_counts),
        "residual_selected_steps": selected_count,
        "residual_selected_without_solo_samples": cold_start_transitions,
        "actual_multilayer_steps_exceeding_same_step_solo_fallback": len(increases),
        "quota_increase_evidence": increases,
        "cohorts": summarized,
        "v1_v2_shape_comparison": compare_shapes(v1, summarized),
        "measurement_contract": {
            "diagnostic_only": True,
            "no_runtime_mutations_or_api_calls": True,
            "scope": (
                "visible lifecycle/progress, producer learning validity, "
                "estimator arithmetic"
            ),
            "limitations": [
                (
                    "Residual is non-P host wall cost, not pure D "
                    "or an all-rank worst-case bound."
                ),
                (
                    "Late host abort eligibility relies on "
                    "the recorded producer learning rejection."
                ),
                (
                    "Replay covers the residual cache only "
                    "when no residual entries were evicted."
                ),
                "P and solo-D cache histories are not independently reconstructed.",
                (
                    "Quota arithmetic using recorded solo D "
                    "is not a counterfactual latency result."
                ),
                (
                    "v1/v2 diagnostic stage timing and cache histories can differ; "
                    "cohort summaries are not matched performance trials."
                ),
                (
                    "A quota advantage may be absent; complete lifecycle checks "
                    "do not imply improvement."
                ),
            ],
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": fingerprint(args.output),
                "residual_samples": len(samples),
                "residual_selected_steps": selected_count,
                "actual_quota_increase_steps": len(increases),
                "paired_steps": len(transactions),
            },
            indent=2,
        )
    )
    if args.require_quota_increase and not increases:
        raise SystemExit(
            "No actual multilayer advancement exceeding solo fallback was observed"
        )


if __name__ == "__main__":
    main()
