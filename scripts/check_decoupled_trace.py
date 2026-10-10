"""Verify persistent-prefill admission, decode progress and cancellation ACKs."""

import argparse
import collections
import json
import math
from pathlib import Path

LAYER_SCHEDULE_FIELDS = {"layers_per_slice", "layers_completed_before", "layers_total"}
LAYER_COMPLETE_FIELDS = {"layers_completed", "layers_total"}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise AssertionError(f"Invalid {name}: expected integer >= {minimum}")
    return value


def _check_layer_schedule(row, cohort, progress, model_layers, expected_layers):
    if not LAYER_SCHEDULE_FIELDS <= row.keys():
        raise AssertionError("Incomplete layer scheduling metadata")
    quota = _integer(row["layers_per_slice"], "layer quota", 1)
    before = _integer(row["layers_completed_before"], "previous layer count")
    total = _integer(row["layers_total"], "total layer count")
    pending = set(row["pending_prefill_ids"])
    if not pending:
        if total or before or row["cohort_id"] is not None:
            raise AssertionError("Layer state without a prefill cohort")
        return 0
    if total < 1 or before >= total:
        raise AssertionError("Invalid pending prefill layer range")
    if expected_layers is not None and total != expected_layers:
        raise AssertionError("Layer count does not match the expected model")
    previous_total = model_layers.setdefault(row["pid"], total)
    if total != previous_total:
        raise AssertionError("Model layer count changed within one worker")
    if before != progress.get(cohort, 0):
        raise AssertionError("Layer progress changed without a matching receipt")
    paused, finish = row["pause_prefill"], row["finish_prefill"]
    if type(paused) is not bool or type(finish) is not bool:
        raise AssertionError("Invalid prefill control flags")
    full_cancel = set(row["cancelled_prefill_ids"]) == pending
    info = row.get("quota")
    if info is not None:
        if not isinstance(info, dict):
            raise AssertionError("Invalid quota metadata")
        reason = info.get("reason")
        if reason in ("deadline_quota", "deadline_finish"):
            rounds = _integer(info.get("allowed_decode_rounds"), "allowed rounds", 1)
            if info.get("enabled") is not True or info.get("calibrated") is not True:
                raise AssertionError(
                    "Deadline quota used without calibrated adaptation"
                )
            remaining = total - before
            if not math.ceil(remaining / rounds) <= quota <= remaining:
                raise AssertionError("Layer quota violates the allowed decode rounds")
            if reason == "deadline_finish" and (rounds != 1 or not finish):
                raise AssertionError("Deadline finish did not request full completion")
    return (
        before
        if paused or full_cancel
        else total
        if finish
        else min(total, before + quota)
    )


def _check_layer_completion(row, scheduled, expected, pending, completed, cancelled):
    if not LAYER_COMPLETE_FIELDS <= row.keys():
        raise AssertionError("Missing layer progress receipt")
    progress = _integer(row["layers_completed"], "completed layer count")
    total = _integer(row["layers_total"], "receipt total layer count")
    if total != scheduled["layers_total"] or progress != expected:
        raise AssertionError("Layer receipt does not match the scheduled quota/control")
    if completed and progress != total:
        raise AssertionError("Prefill retired before the final layer")
    if pending and progress == total and completed | cancelled != pending:
        raise AssertionError("Final layer lacks complete cohort retirement")
    if pending and set(scheduled["cancelled_prefill_ids"]) == pending:
        if completed | cancelled != pending:
            raise AssertionError("Full cancellation lacks a cohort ACK")


def check(paths, require_cancellation=False, min_decode_steps=2, expected_layers=None):
    _integer(min_decode_steps, "minimum decode steps")
    if expected_layers is not None:
        _integer(expected_layers, "expected model layers", 1)
    cohorts = {}
    steps = {}
    frames = collections.Counter()
    retirements = collections.Counter()
    max_prompt = 0
    layer_protocol = {}
    layer_progress = {}
    model_layers = {}
    expected_progress = {}
    checked = collections.Counter()
    quota_reasons = collections.Counter()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            event = row["event"]
            key = (row["pid"], row["step"])
            if event == "schedule":
                if row["scheduled_tokens"] != row["expected_tokens"]:
                    raise AssertionError(f"Token chunking at {key}")
                for rid, kind in row["request_phases"].items():
                    if kind == "prefill":
                        max_prompt = max(max_prompt, row["scheduled_tokens"][rid])
                continue
            if event == "decoupled_schedule":
                if key in steps:
                    raise AssertionError("Duplicate step")
                steps[key] = row
                cohort = (row["pid"], row["cohort_id"])
                pending = set(row["pending_prefill_ids"])
                new = set(row["new_prefill_ids"])
                if new:
                    if cohort in cohorts or new != pending:
                        raise AssertionError(
                            "Prefill admitted twice or pin set mismatch"
                        )
                    if any(k[0] == row["pid"] and v for k, v in cohorts.items()):
                        raise AssertionError(
                            "Second prefill cohort admitted before ACK"
                        )
                    cohorts[cohort] = pending.copy()
                elif pending != cohorts.get(cohort, set()):
                    raise AssertionError(
                        "Pending prefill ownership changed without ACK"
                    )
                if pending & set(row["decode_ids"]):
                    raise AssertionError("Unfinished prefill entered decode")
                if not set(row["cancelled_prefill_ids"]) <= pending:
                    raise AssertionError("Unknown prefill cancelled")
                has_layers = bool(LAYER_SCHEDULE_FIELDS & row.keys()) or "quota" in row
                previous_protocol = layer_protocol.setdefault(row["pid"], has_layers)
                if has_layers != previous_protocol:
                    raise AssertionError(
                        "Layer receipt protocol changed within one worker"
                    )
                if has_layers:
                    expected_progress[key] = _check_layer_schedule(
                        row, cohort, layer_progress, model_layers, expected_layers
                    )
                    if row.get("quota"):
                        quota_reasons[row["quota"].get("reason", "unknown")] += 1
                elif expected_layers is not None:
                    raise AssertionError(
                        "Legacy trace cannot verify expected model layers"
                    )
                continue
            if event != "decoupled_complete":
                continue
            if key not in steps:
                raise AssertionError("Unknown completion")
            scheduled = steps.pop(key)
            if row["cohort_id"] != scheduled["cohort_id"]:
                raise AssertionError("Reordered cohort")
            cohort = (row["pid"], row["cohort_id"])
            completed = set(row["completed_prefill_ids"])
            cancelled = set(row["cancelled_prefill_ids"])
            pending = cohorts.get(cohort, set())
            if completed & cancelled or not (completed | cancelled) <= pending:
                raise AssertionError("Duplicate or unknown retirement")
            if set(row["decode_ids"]) != set(scheduled["decode_ids"]):
                raise AssertionError("Decode receipt differs from the scheduled IDs")
            if not cancelled <= set(scheduled["cancelled_prefill_ids"]):
                raise AssertionError("Worker acknowledged an unrequested cancellation")
            has_layers = bool(LAYER_COMPLETE_FIELDS & row.keys())
            if has_layers != layer_protocol[row["pid"]]:
                raise AssertionError("Layer receipt protocol differs from its schedule")
            if has_layers:
                _check_layer_completion(
                    row,
                    scheduled,
                    expected_progress.pop(key),
                    pending,
                    completed,
                    cancelled,
                )
                layer_progress[cohort] = row["layers_completed"]
                checked["layer_progress_steps"] += 1
            else:
                checked["legacy_steps"] += 1
            if pending and row["decode_ids"] and not completed and cancelled != pending:
                frames[cohort] += 1
            pending.difference_update(completed | cancelled)
            retirements["completed"] += len(completed)
            retirements["cancelled"] += len(cancelled)
    if steps or any(cohorts.values()):
        raise AssertionError("Trace ends with unretired worker steps/prefill ownership")
    if not cohorts:
        raise AssertionError("No prefill cohort evidence")
    if max(frames.values(), default=0) < min_decode_steps:
        raise AssertionError("No evidence of repeated decode during pending prefill")
    if require_cancellation and not retirements["cancelled"]:
        raise AssertionError("No prefill cancellation ACK in trace")
    return {
        "passed": True,
        "cohorts": len(cohorts),
        "max_complete_prompt_tokens": max_prompt,
        "max_decode_steps_before_prefill_completion": max(frames.values(), default=0),
        "prefill_retirements": dict(retirements),
        "receipt_checks": dict(checked),
        "model_layers_by_pid": model_layers,
        "quota_reasons": dict(quota_reasons),
        "scope": "host lifecycle/progress receipts; kernel overlap needs profiling",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--require-cancellation", action="store_true")
    parser.add_argument(
        "--expected-layers",
        type=int,
        help="Verify the model layer count; requires a trace with layer receipts",
    )
    parser.add_argument(
        "--min-decode-steps",
        type=int,
        default=2,
        help="Required repeated decode progress; use 0 for deadline-driven full P",
    )
    args = parser.parse_args()
    if args.min_decode_steps < 0:
        parser.error("min-decode-steps must be nonnegative")
    print(
        json.dumps(
            check(
                args.paths,
                args.require_cancellation,
                args.min_decode_steps,
                args.expected_layers,
            ),
            indent=2,
        )
    )
