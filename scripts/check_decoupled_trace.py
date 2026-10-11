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
    short_limit = _integer(
        scheduled.get("compile_short_prefill_tokens", 0), "short-P compile threshold"
    )
    work = (scheduled.get("quota") or {}).get("work_quota") or {}
    short_eligible = bool(
        short_limit
        and 0
        < _integer(work.get("cohort_full_prefill_tokens", 0), "P tokens")
        <= short_limit
        and scheduled.get("quota", {}).get("reason")
        == "compiled_short_prefill_full_quota"
    )
    if source == "full_forward_return" and not (
        (scheduled["compile_full_prefill"] or short_eligible)
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


def _check_work_quota(
    row, cohort, cohort_tokens, configurations, scheduled_prefills, guard_shapes
):
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
    guard = info.get("decode_budget")
    reason = (
        guard.get("desired_reason") if isinstance(guard, dict) else info.get("reason")
    )
    if reason in ("deadline_quota", "deadline_finish"):
        if info.get("enabled") is not True or info.get("calibrated") is not True:
            raise AssertionError("Guard used an uncalibrated deadline quota")
        rounds = _integer(info["allowed_decode_rounds"], "allowed rounds", 1)
        expected = min(remaining, max(base, math.ceil(remaining / rounds)))
    if reason == "compiled_short_prefill_full_quota":
        short_limit = _integer(
            row.get("compile_short_prefill_tokens"), "short-P compile threshold", 1
        )
        if not (
            0 < tokens <= short_limit
            and new == set(row["pending_prefill_ids"])
            and row["layers_completed_before"] == 0
            and not row["pause_prefill"]
            and not row["cancelled_prefill_ids"]
            and row["finish_prefill"]
            and not row["compile_full_prefill"]
            and guard is None
        ):
            raise AssertionError("Short P full quota requires a fresh eligible cohort")
        expected = total
    if guard is not None:
        expected = _check_decode_budget(
            row, guard, expected, cohort, guard_shapes, scheduled_prefills
        )
    elif info.get("reason") == "decode_budget_quota":
        raise AssertionError("Decode budget quota lacks guard evidence")
    if row["layers_per_slice"] != expected:
        raise AssertionError("Scheduled quota differs from work/deadline calculation")


def _finite(value, name, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        raise AssertionError(f"Invalid {name}: expected finite number")
    return value


def _close(actual, expected, name):
    if not math.isclose(_finite(actual, name, None), expected, abs_tol=1e-6):
        raise AssertionError(f"Incorrect decode budget {name}")


def _check_decode_budget(row, guard, desired, cohort, shapes, scheduled_prefills):
    """Audit a lowered quota separately from the unchanged work/deadline desire.

    The trace verifies reported observations and arithmetic, not a hard latency
    bound. Active D includes requests excluded from this admission's token map.
    """
    if (
        not isinstance(guard, dict)
        or type(guard.get("version")) is not int
        or (guard["version"] not in (1, 2))
    ):
        raise AssertionError("Invalid decode budget protocol")
    if row["quota"]["reason"] != "decode_budget_quota":
        raise AssertionError("Guard evidence without a decode budget quota")
    if _integer(guard.get("desired_layers_per_slice"), "desired quota", 1) != desired:
        raise AssertionError("Guard changed the work/deadline desired quota")
    active = guard.get("active_decode_ids")
    contexts = guard.get("active_decode_context_tokens")
    selected = guard.get("selected_decode_ids")
    ages = guard.get("decode_age_ms")
    if (
        not isinstance(active, list)
        or any(not isinstance(rid, str) for rid in active)
        or len(active) != len(set(active))
        or not isinstance(contexts, dict)
        or set(active) != set(contexts)
        or selected != sorted(row["decode_ids"])
        or not set(selected) <= set(active)
        or set(active) & set(row["pending_prefill_ids"])
        or not isinstance(ages, dict)
        or not set(ages) <= set(active)
    ):
        raise AssertionError("Invalid active/selected decode budget evidence")
    lengths = [_integer(n, "active decode context", 1) for n in contexts.values()]
    shape = (
        [len(lengths), (sum(lengths) - 1).bit_length(), (max(lengths) - 1).bit_length()]
        if lengths
        else None
    )
    if guard.get("decode_shape") != shape:
        raise AssertionError("Decode budget context shape mismatch")
    p_shape = guard.get("prefill_shape_tokens")
    if not isinstance(p_shape, list) or not p_shape:
        raise AssertionError("Missing raw prefill shape")
    for tokens in p_shape:
        _integer(tokens, "raw prefill shape tokens", 1)
    if row["new_prefill_ids"]:
        shapes[cohort] = sorted(scheduled_prefills.values())
    if p_shape != shapes.get(cohort):
        raise AssertionError("Guard prefill shape changed from original allocation")
    for age in ages.values():
        _finite(age, "decode age")
    p_count = _integer(guard.get("prefill_samples"), "raw prefill sample count")
    first_count = _integer(guard.get("first_layer_samples"), "first layer sample count")
    d_count = _integer(guard.get("decode_samples"), "selected decode cost sample count")
    p_ms = guard.get("estimated_raw_layer_ms")
    first_ms = guard.get("estimated_first_layer_ms")
    d_ms = guard.get("estimated_solo_decode_ms")
    first = row["layers_completed_before"] == 0
    if guard["version"] == 2:
        solo_count = _integer(guard.get("solo_decode_samples"), "solo D sample count")
        residual_count = _integer(
            guard.get("residual_decode_samples"), "residual D sample count"
        )
        residual_ms = guard.get("estimated_decode_residual_ms")
        for value, count in ((d_ms, solo_count), (residual_ms, residual_count)):
            if count:
                if _finite(value, "decode observation estimate") <= 0:
                    raise AssertionError("Decode observation must be positive")
            elif value is not None:
                raise AssertionError("Decode observation lacks matching samples")
        use_residual = not first and residual_count > 0
        source = (
            "fenced_continuation_residual"
            if use_residual
            else "solo_decode_round_trip"
            if solo_count
            else None
        )
        if guard.get("decode_estimate_source") != source or d_count != (
            residual_count if use_residual else solo_count
        ):
            raise AssertionError("Invalid continuation/solo decode estimate source")
        expected_d = residual_ms if use_residual else d_ms
        d_ms = guard.get("estimated_decode_cost_ms")
        if expected_d is None:
            if d_ms is not None:
                raise AssertionError("Decode cost lacks a matching observation")
        else:
            _close(d_ms, expected_d, "selected non-prefill cost")
    for value, count in ((p_ms, p_count), (first_ms, first_count), (d_ms, d_count)):
        if count:
            if _finite(value, "estimated guard latency") <= 0:
                raise AssertionError("Guard latency must be positive")
        elif value is not None:
            raise AssertionError("Guard estimate lacks a matching sample")
    if guard.get("first_slice") is not first:
        raise AssertionError("Guard first-slice mode differs from layer progress")
    calibrated = bool(
        p_count and d_count and (not first or first_count) and set(ages) == set(active)
    )
    if guard.get("calibrated") is not calibrated:
        raise AssertionError("Guard calibration does not match its shape samples")
    budget = _finite(guard.get("decode_budget_ms"), "decode budget")
    slack = budget - max(ages.values(), default=0) - d_ms if d_ms is not None else None
    if slack is None:
        if guard.get("decode_slack_ms") is not None:
            raise AssertionError("Unknown D latency cannot give measured slack")
    else:
        _close(guard.get("decode_slack_ms"), slack, "decode slack")
    remaining = row["layers_total"] - row["layers_completed_before"]
    if not active:
        mode, cap, quota = "no_live_decode", remaining, remaining
    elif not calibrated:
        mode, cap, quota = "cold_start", 1, 1
    else:
        mode = "measured_slice_budget"
        cap = min(
            remaining,
            max(1, 1 + math.floor((slack - first_ms) / p_ms))
            if first
            else max(1, math.floor(max(0, slack) / p_ms)),
        )
        quota = min(remaining, desired, cap)
    if (
        guard.get("mode") != mode
        or _integer(guard.get("cap_layers"), "guard cap", 1) != cap
        or _integer(guard.get("selected_layers_per_slice"), "guard quota", 1) != quota
    ):
        raise AssertionError("Guard quota differs from the measured/cold-start cap")
    control = bool(
        row["pause_prefill"]
        or set(row["cancelled_prefill_ids"]) == set(row["pending_prefill_ids"])
    )
    if guard.get("control_step") is not control or row["finish_prefill"] != (
        quota == remaining and not control
    ):
        raise AssertionError("Guard finish flag bypasses the decode cap")
    exceeded = bool(active and calibrated and slack < (first_ms if first else p_ms))
    if guard.get("minimum_progress_exceeds_budget") is not exceeded:
        raise AssertionError("Guard failed to report an infeasible minimum layer")
    age = _finite(guard.get("cohort_age_ms"), "cohort age")
    ttft = _finite(guard.get("ttft_budget_ms"), "TTFT budget")
    projected = (
        (first_ms + (remaining - 1) * p_ms if first else remaining * p_ms)
        + math.ceil(remaining / quota) * d_ms
        if calibrated
        else None
    )
    if projected is None:
        if guard.get("estimated_remaining_ttft_ms") is not None:
            raise AssertionError("Unknown slice costs cannot give a TTFT estimate")
    else:
        _close(guard.get("estimated_remaining_ttft_ms"), projected, "remaining TTFT")
    feasible = (
        age + projected <= ttft
        if projected is not None
        else False
        if age >= ttft
        else None
    )
    if guard.get("ttft_feasible") is not feasible:
        raise AssertionError("Guard TTFT feasibility mismatch")
    return quota


def _check_decode_residual_sample(row, scheduled):
    sample = (row.get("decode_budget_sample") or {}).get("decode_residual_sample")
    if sample is None:
        return
    timing = row.get("prefill_slice_timing")
    if not (
        isinstance(sample, dict)
        and isinstance(timing, dict)
        and timing.get("device_complete") is True
        and timing.get("first_slice") is False
        and timing.get("mode") == "raw_layers"
        and row.get("progress_source") == "layer_hooks"
        and 0
        < scheduled["layers_completed_before"]
        < row["layers_completed"]
        < row["layers_total"]
        and scheduled["decode_ids"]
        and not scheduled["pause_prefill"]
        and not scheduled["cancelled_prefill_ids"]
        and not row["cancelled_prefill_ids"]
        and row["decode_budget_sample"].get("phase") == "raw_prefill_layer"
    ):
        raise AssertionError("Residual sample lacks completed raw continuation")
    contexts = scheduled["quota"]["decode_budget"]["active_decode_context_tokens"]
    lengths = [contexts[rid] for rid in scheduled["decode_ids"]]
    shape = [
        len(lengths),
        (sum(lengths) - 1).bit_length(),
        (max(lengths) - 1).bit_length(),
    ]
    if sample.get("decode_shape") != shape:
        raise AssertionError("Residual sample differs from selected decode shape")
    host = _finite(row["duration_ms"], "residual host step")
    device = _finite(timing.get("device_elapsed_ms"), "residual device span")
    _close(sample.get("host_step_ms"), host, "residual host step")
    _close(sample.get("prefill_device_elapsed_ms"), device, "residual device span")
    _close(
        sample.get("prefill_host_submit_ms"),
        _finite(timing.get("host_submit_ms"), "residual host submit"),
        "residual host submit",
    )
    _close(sample.get("residual_ms"), host - device, "fenced non-P residual")
    accepted = host > device
    reason = (
        "completed_fenced_continuation"
        if accepted
        else "nonpositive_or_nonfinite_residual"
    )
    if sample.get("accepted") is not accepted or sample.get("reason") != reason:
        raise AssertionError("Invalid residual acceptance decision")


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
    guard_protocol = {}
    guard_shapes = {}
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
                has_guard = isinstance(info, dict) and "decode_budget" in info
                if guard_protocol.setdefault(row["pid"], has_guard) != has_guard:
                    raise AssertionError(
                        "Decode budget protocol changed within one worker"
                    )
                if has_guard and not pending and info["decode_budget"] is not None:
                    raise AssertionError("Decode budget P quota without a cohort")
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
                        guard_shapes,
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
            _check_decode_residual_sample(row, scheduled)
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
