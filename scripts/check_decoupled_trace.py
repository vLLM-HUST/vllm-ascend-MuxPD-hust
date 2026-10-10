"""Verify persistent-prefill admission, decode progress and cancellation ACKs."""

import argparse
import collections
import json
import math
from pathlib import Path

LAYER_SCHEDULE_FIELDS = {"layers_per_slice", "layers_completed_before", "layers_total"}
LAYER_COMPLETE_FIELDS = {"layers_completed", "layers_total"}
WORK_QUOTA_FIELDS = {
    "source",
    "prefill_layer_token_budget",
    "cohort_full_prefill_tokens",
    "configured_layers_per_slice",
    "base_layers_per_slice",
}


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


def _check_progress_source(row, scheduled, pending, completed, cancelled):
    source = row["progress_source"]
    if type(source) is not str or source not in (
        "none",
        "layer_hooks",
        "full_forward_return",
    ):
        raise AssertionError("Invalid prefill progress source")
    if (source == "none") != (not pending):
        raise AssertionError("Prefill progress source does not match cohort ownership")
    if type(scheduled.get("compile_full_prefill")) is not bool:
        raise AssertionError("Missing compiled-prefill scheduling configuration")
    if source == "full_forward_return" and not (
        scheduled["compile_full_prefill"]
        and set(scheduled["new_prefill_ids"]) == pending
        and scheduled["layers_completed_before"] == 0
        and not scheduled["pause_prefill"]
        and not scheduled["cancelled_prefill_ids"]
        and row["layers_completed"] == row["layers_total"]
        and completed == pending
        and not cancelled
    ):
        # Exact expected progress has already been checked against the control
        # flags and quota. This source certifies a full forward return, never
        # resumed execution whose Python layer hooks may have been bypassed.
        raise AssertionError("Full-forward progress is not a fresh complete prefill")


def _check_work_quota(row, cohort, cohort_tokens, configurations, scheduled_prefills):
    info = row["quota"]
    work = info["work_quota"]
    if not row["pending_prefill_ids"]:
        if work is not None:
            raise AssertionError("Work quota without a prefill cohort")
        return
    if not isinstance(work, dict) or not WORK_QUOTA_FIELDS <= work.keys():
        raise AssertionError("Missing work quota metadata")
    budget = _integer(work["prefill_layer_token_budget"], "layer-token budget")
    tokens = _integer(work["cohort_full_prefill_tokens"], "cohort prefill tokens", 1)
    floor = _integer(work["configured_layers_per_slice"], "configured layer floor", 1)
    base = _integer(work["base_layers_per_slice"], "base layer quota", 1)
    expected_source = (
        "prefill_layer_token_budget" if budget else "configured_layers_per_slice"
    )
    if work["source"] != expected_source:
        raise AssertionError("Work quota source does not match its budget")
    config = configurations.setdefault(row["pid"], (budget, floor))
    if config != (budget, floor):
        raise AssertionError("Work quota configuration changed within one worker")
    new = set(row["new_prefill_ids"])
    if new:
        if set(scheduled_prefills) != new:
            raise AssertionError(
                "Work quota lacks original full-prefill token evidence"
            )
        cohort_tokens[cohort] = sum(
            _integer(tokens, "scheduled prefill tokens", 1)
            for tokens in scheduled_prefills.values()
        )
    if cohort_tokens.get(cohort) != tokens:
        raise AssertionError("Work quota differs from original cohort token count")
    total = row["layers_total"]
    expected_base = max(floor, min(total, budget // tokens)) if budget else floor
    if base != expected_base:
        raise AssertionError("Work quota does not match the layer-token formula")
    remaining = total - row["layers_completed_before"]
    expected = min(remaining, base) if budget else floor
    if info.get("reason") in ("deadline_quota", "deadline_finish"):
        rounds = _integer(info["allowed_decode_rounds"], "allowed rounds", 1)
        expected = min(remaining, max(base, math.ceil(remaining / rounds)))
    if row["layers_per_slice"] != expected:
        raise AssertionError("Scheduled quota differs from work/deadline calculation")


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
    source_protocol = {}
    progress_sources = collections.Counter()
    scheduled_prefills = {}
    work_protocol = {}
    cohort_work_tokens = {}
    work_configurations = {}
    work_quota_sources = collections.Counter()
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
                scheduled_prefills[key] = {
                    rid: tokens
                    for rid, tokens in row["scheduled_tokens"].items()
                    if row["request_phases"][rid] == "prefill"
                }
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
                info = row.get("quota")
                has_work = isinstance(info, dict) and "work_quota" in info
                previous_work = work_protocol.setdefault(row["pid"], has_work)
                if has_work != previous_work:
                    raise AssertionError(
                        "Work quota protocol changed within one worker"
                    )
                if has_work:
                    if not has_layers:
                        raise AssertionError("Work quota requires exact layer receipts")
                    _check_work_quota(
                        row,
                        cohort,
                        cohort_work_tokens,
                        work_configurations,
                        scheduled_prefills.get(key, {}),
                    )
                    if info["work_quota"] is not None:
                        work_quota_sources[info["work_quota"]["source"]] += 1
                scheduled_prefills.pop(key, None)
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
            has_source = "progress_source" in row
            previous_source = source_protocol.setdefault(row["pid"], has_source)
            if has_source != previous_source:
                raise AssertionError(
                    "Progress source protocol changed within one worker"
                )
            if "compile_full_prefill" in scheduled and not has_source:
                raise AssertionError("Missing prefill progress source receipt")
            if has_source:
                if not has_layers:
                    raise AssertionError(
                        "Progress source requires exact layer receipts"
                    )
                _check_progress_source(row, scheduled, pending, completed, cancelled)
                progress_sources[row["progress_source"]] += 1
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
        "progress_sources": dict(progress_sources),
        "work_quota_sources": dict(work_quota_sources),
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
