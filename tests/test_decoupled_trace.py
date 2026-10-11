"""Offline protocol checks against small synthetic JSONL receipts."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_decoupled_trace.py"
SPEC = importlib.util.spec_from_file_location("muxpd_cohort_trace", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def trace(*, legacy=False, full=False, cancel=False):
    rows = [
        {
            "event": "schedule",
            "pid": 7,
            "step": 1,
            "scheduled_tokens": {"p": 64, "d": 1},
            "expected_tokens": {"p": 64, "d": 1},
            "request_phases": {"p": "prefill", "d": "decode"},
        }
    ]
    before = 0
    count = 1 if full else 3
    for step in range(1, count + 1):
        last = step == count
        quota = 4 if full else 2 if last else 1
        after = before if cancel and last else min(4, before + quota)
        rows += [
            {
                "event": "decoupled_schedule",
                "pid": 7,
                "step": step,
                "cohort_id": 1,
                "pending_prefill_ids": ["p"],
                "new_prefill_ids": ["p"] if step == 1 else [],
                "cancelled_prefill_ids": ["p"] if last and cancel else [],
                "decode_ids": ["d"],
                "finish_prefill": last,
                "pause_prefill": False,
                "layers_per_slice": quota,
                "layers_completed_before": before,
                "layers_total": 4,
                "quota": {
                    "enabled": full,
                    "calibrated": full,
                    "reason": "deadline_finish" if full else "fixed_quota",
                    "allowed_decode_rounds": 1,
                },
            },
            {
                "event": "decoupled_complete",
                "pid": 7,
                "step": step,
                "cohort_id": 1,
                "decode_ids": ["d"],
                "completed_prefill_ids": ["p"] if last and not cancel else [],
                "cancelled_prefill_ids": ["p"] if last and cancel else [],
                "layers_completed": after,
                "layers_total": 4,
            },
        ]
        before = after
    if legacy:
        for row in rows:
            for key in list(row):
                if key.startswith("layers_") or key == "quota":
                    del row[key]
    return rows


def check(tmp_path, rows, **kwargs):
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return checker.check([path], **kwargs)


def test_exact_progress_and_legacy_compatibility(tmp_path):
    current = check(tmp_path, trace(), expected_layers=4)
    assert current["receipt_checks"] == {"layer_progress_steps": 3}
    assert current["model_layers_by_pid"] == {7: 4}
    legacy = check(tmp_path, trace(legacy=True))
    assert legacy["receipt_checks"] == {"legacy_steps": 3}
    with pytest.raises(AssertionError, match="Legacy"):
        check(tmp_path, trace(legacy=True), expected_layers=4)


def test_deadline_finish_requires_explicit_zero_decode_progress_threshold(tmp_path):
    with pytest.raises(AssertionError, match="repeated decode"):
        check(tmp_path, trace(full=True))
    result = check(tmp_path, trace(full=True), min_decode_steps=0, expected_layers=4)
    assert result["quota_reasons"] == {"deadline_finish": 1}


def test_cancellation_ack_does_not_advance_layers(tmp_path):
    result = check(tmp_path, trace(cancel=True), require_cancellation=True)
    assert result["prefill_retirements"]["cancelled"] == 1
    rows = trace(cancel=True)
    rows[-1]["layers_completed"] += 1
    with pytest.raises(AssertionError, match="quota/control"):
        check(tmp_path, rows)


@pytest.mark.parametrize(
    "fault",
    [
        "overquota",
        "regression",
        "changed_total",
        "bool",
        "missing",
        "early_retirement",
        "missing_retirement",
        "different_decode",
        "protocol_downgrade",
    ],
)
def test_corrupted_progress_receipts_fail_closed(tmp_path, fault):
    rows = trace()
    if fault == "overquota":
        rows[2]["layers_completed"] = 2
    elif fault == "regression":
        rows[3]["layers_completed_before"] = 0
    elif fault == "changed_total":
        rows[3]["layers_total"] = 5
    elif fault == "bool":
        rows[2]["layers_completed"] = True
    elif fault == "missing":
        del rows[2]["layers_total"]
    elif fault == "early_retirement":
        rows[2]["completed_prefill_ids"] = ["p"]
    elif fault == "missing_retirement":
        rows[-1]["completed_prefill_ids"] = []
    elif fault == "different_decode":
        rows[2]["decode_ids"] = []
    else:
        for key in list(rows[3]):
            if key.startswith("layers_") or key == "quota":
                del rows[3][key]
    with pytest.raises(AssertionError):
        check(tmp_path, rows)


def test_trace_cannot_claim_progress_without_prefill_evidence(tmp_path):
    with pytest.raises(AssertionError, match="No prefill"):
        check(tmp_path, [], min_decode_steps=0)


def test_deadline_quota_and_model_count_are_checked(tmp_path):
    with pytest.raises(AssertionError, match="expected model"):
        check(tmp_path, trace(), expected_layers=40)
    rows = trace(full=True)
    rows[1]["layers_per_slice"] = 2
    with pytest.raises(AssertionError, match="allowed decode rounds"):
        check(tmp_path, rows, min_decode_steps=0)


def source_trace(*, full=False, cancel=False):
    rows = trace(full=full, cancel=cancel)
    for row in rows:
        if row["event"] == "decoupled_schedule":
            row["compile_full_prefill"] = full
        elif row["event"] == "decoupled_complete":
            row["progress_source"] = "full_forward_return" if full else "layer_hooks"
    return rows


def test_full_forward_source_and_layer_hooks_are_distinct_evidence(tmp_path):
    full = check(tmp_path, source_trace(full=True), min_decode_steps=0)
    assert full["progress_sources"] == {"full_forward_return": 1}
    hooks = check(tmp_path, source_trace())
    assert hooks["progress_sources"] == {"layer_hooks": 3}
    cancelled = check(tmp_path, source_trace(cancel=True), require_cancellation=True)
    assert cancelled["progress_sources"] == {"layer_hooks": 3}


@pytest.mark.parametrize("fault", ["unknown", "none", "disabled", "missing_config"])
def test_invalid_full_forward_source_fails_closed(tmp_path, fault):
    rows = source_trace(full=True)
    if fault in ("unknown", "none"):
        rows[-1]["progress_source"] = fault
    elif fault == "disabled":
        rows[1]["compile_full_prefill"] = False
    else:
        del rows[1]["compile_full_prefill"]
    with pytest.raises(AssertionError):
        check(tmp_path, rows, min_decode_steps=0)


@pytest.mark.parametrize("step", [0, 1, 2])
def test_partial_or_old_prefill_cannot_claim_full_forward(tmp_path, step):
    rows = source_trace()
    for row in rows:
        if row["event"] == "decoupled_schedule":
            row["compile_full_prefill"] = True
    # The last receipt is a valid complete forward, but it belongs to an old
    # cohort. The first two are valid quota receipts, but incomplete forwards.
    rows[2 + step * 2]["progress_source"] = "full_forward_return"
    with pytest.raises(AssertionError, match="fresh complete"):
        check(tmp_path, rows)


def test_resuming_zero_progress_cohort_cannot_claim_full_forward(tmp_path):
    rows = source_trace()
    rows[1]["pause_prefill"] = True
    rows[2]["layers_completed"] = 0
    rows[3]["layers_completed_before"] = 0
    rows[3]["finish_prefill"] = True
    rows[3]["compile_full_prefill"] = True
    rows[4]["layers_completed"] = 4
    rows[4]["completed_prefill_ids"] = ["p"]
    rows[4]["progress_source"] = "full_forward_return"
    with pytest.raises(AssertionError, match="fresh complete"):
        check(tmp_path, rows[:5], min_decode_steps=0)


@pytest.mark.parametrize("step", [0, 1, 2])
def test_progress_source_cannot_be_missing_or_downgraded(tmp_path, step):
    rows = source_trace()
    del rows[2 + step * 2]["progress_source"]
    with pytest.raises(AssertionError, match="source"):
        check(tmp_path, rows)


@pytest.mark.parametrize("source", ["none", "layer_hooks", "full_forward_return"])
def test_no_cohort_source_requires_none(tmp_path, source):
    rows = source_trace(full=True)
    schedule = rows[1].copy()
    schedule.update(
        step=2,
        cohort_id=None,
        pending_prefill_ids=[],
        new_prefill_ids=[],
        layers_completed_before=0,
        layers_total=0,
    )
    complete = rows[2].copy()
    complete.update(
        step=2,
        cohort_id=None,
        completed_prefill_ids=[],
        layers_completed=0,
        layers_total=0,
        progress_source=source,
    )
    rows += [schedule, complete]
    if source == "none":
        report = check(tmp_path, rows, min_decode_steps=0)
        assert report["progress_sources"] == {"full_forward_return": 1, "none": 1}
    else:
        with pytest.raises(AssertionError, match="cohort ownership"):
            check(tmp_path, rows, min_decode_steps=0)


def work_trace():
    template = source_trace()
    rows = template[:1]
    for step in range(1, 5):
        schedule = json.loads(json.dumps(template[1]))
        complete = json.loads(json.dumps(template[2]))
        schedule.update(
            step=step,
            new_prefill_ids=["p"] if step == 1 else [],
            layers_per_slice=1,
            layers_completed_before=step - 1,
            finish_prefill=step == 4,
        )
        schedule["quota"]["reason"] = "token_budget_quota"
        schedule["quota"]["work_quota"] = {
            "source": "prefill_layer_token_budget",
            "prefill_layer_token_budget": 64,
            "cohort_full_prefill_tokens": 64,
            "configured_layers_per_slice": 1,
            "base_layers_per_slice": 1,
        }
        complete.update(
            step=step,
            layers_completed=step,
            completed_prefill_ids=["p"] if step == 4 else [],
        )
        rows += [schedule, complete]
    return rows


def test_work_quota_verifies_original_cohort_work_and_keeps_legacy_separate(tmp_path):
    result = check(tmp_path, work_trace(), expected_layers=4)
    assert result["work_quota_sources"] == {"prefill_layer_token_budget": 4}
    legacy = check(tmp_path, trace())
    assert legacy["work_quota_sources"] == {}


def short_fx_trace():
    rows = trace(full=True)
    scheduled, completed = rows[1:]
    scheduled.update(compile_full_prefill=False, piecewise_short_prefill_tokens=64)
    scheduled["quota"].update(
        reason="piecewise_short_prefill_full_quota",
        work_quota={
            "source": "configured_layers_per_slice",
            "prefill_layer_token_budget": 0,
            "cohort_full_prefill_tokens": 64,
            "configured_layers_per_slice": 1,
            "base_layers_per_slice": 1,
        },
    )
    completed["progress_source"] = "full_forward_return"
    return rows


def test_short_fx_trace_audits_full_quota_and_distinct_opt_in(tmp_path):
    check(tmp_path, short_fx_trace(), min_decode_steps=0)


@pytest.mark.parametrize("fault", ["threshold", "quota", "cancel", "fresh", "source"])
def test_short_fx_trace_rejects_ineligible_full_return(tmp_path, fault):
    rows = short_fx_trace()
    scheduled = rows[1]
    if fault == "threshold":
        scheduled["piecewise_short_prefill_tokens"] = 63
    elif fault == "quota":
        scheduled["layers_per_slice"] = 1
    elif fault == "cancel":
        scheduled["cancelled_prefill_ids"] = ["p"]
    elif fault == "fresh":
        scheduled["new_prefill_ids"] = []
    else:
        scheduled["quota"]["reason"] = "fixed_quota"
    with pytest.raises(AssertionError):
        check(tmp_path, rows)


@pytest.mark.parametrize(
    "fault",
    [
        "ceil",
        "token_count",
        "shrink",
        "quota",
        "source",
        "config_change",
        "protocol_downgrade",
        "missing_original_tokens",
    ],
)
def test_corrupted_work_quota_proof_is_rejected(tmp_path, fault):
    rows = work_trace()
    work = rows[1]["quota"]["work_quota"]
    if fault == "ceil":
        work["prefill_layer_token_budget"] = 127
        work["base_layers_per_slice"] = 2
    elif fault == "token_count":
        work["cohort_full_prefill_tokens"] = 32
    elif fault == "shrink":
        rows[3]["quota"]["work_quota"]["cohort_full_prefill_tokens"] = 32
    elif fault == "quota":
        rows[1]["layers_per_slice"] = 2
    elif fault == "source":
        work["source"] = "configured_layers_per_slice"
    elif fault == "config_change":
        rows[3]["quota"]["work_quota"]["prefill_layer_token_budget"] = 128
    elif fault == "protocol_downgrade":
        del rows[3]["quota"]["work_quota"]
    else:
        rows = rows[1:]
    with pytest.raises(AssertionError):
        check(tmp_path, rows)


def test_deadline_can_raise_work_quota_and_current_receipt_stays_exact(tmp_path):
    rows = source_trace(full=True)
    rows[1]["quota"]["work_quota"] = {
        "source": "prefill_layer_token_budget",
        "prefill_layer_token_budget": 128,
        "cohort_full_prefill_tokens": 64,
        "configured_layers_per_slice": 1,
        "base_layers_per_slice": 2,
    }
    report = check(tmp_path, rows, min_decode_steps=0)
    assert report["work_quota_sources"] == {"prefill_layer_token_budget": 1}
    assert report["progress_sources"] == {"full_forward_return": 1}


def guard_trace(*, measured=False, selected=True):
    rows = work_trace()
    if not selected:
        rows[0]["scheduled_tokens"].pop("d")
        rows[0]["expected_tokens"].pop("d")
        rows[0]["request_phases"].pop("d")
    for row in rows:
        if row["event"] == "decoupled_complete":
            if row["step"] == 1 and not selected:
                row["decode_ids"] = []
            continue
        if row["event"] != "decoupled_schedule":
            continue
        before = row["layers_completed_before"]
        remaining = row["layers_total"] - before
        row["quota"]["enabled"] = True
        row["quota"]["work_quota"].update(
            prefill_layer_token_budget=256,
            configured_layers_per_slice=4,
            base_layers_per_slice=4,
        )
        row["quota"]["reason"] = "decode_budget_quota"
        if row["step"] == 1 and not selected:
            row["decode_ids"] = []
        row["quota"]["decode_budget"] = {
            "version": 1,
            "desired_reason": "token_budget_quota",
            "desired_layers_per_slice": remaining,
            "active_decode_ids": ["d"],
            "active_decode_context_tokens": {"d": 65},
            "selected_decode_ids": row["decode_ids"],
            "decode_shape": [1, 7, 7],
            "decode_age_ms": {"d": 0},
            "prefill_shape_tokens": [64],
            "first_slice": before == 0,
            "first_layer_samples": 1 if measured else 0,
            "prefill_samples": 1 if measured else 0,
            "decode_samples": 1 if measured else 0,
            "estimated_first_layer_ms": 40 if measured else None,
            "estimated_raw_layer_ms": 40 if measured else None,
            "estimated_solo_decode_ms": 60 if measured else None,
            "calibrated": measured,
            "decode_budget_ms": 100,
            "decode_slack_ms": 40 if measured else None,
            "mode": "measured_slice_budget" if measured else "cold_start",
            "cap_layers": 1,
            "selected_layers_per_slice": 1,
            "control_step": False,
            "minimum_progress_exceeds_budget": False,
            "cohort_age_ms": 0,
            "ttft_budget_ms": 1000,
            "estimated_remaining_ttft_ms": remaining * 100 if measured else None,
            "ttft_feasible": True if measured else None,
        }
    return rows


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_guard_trace_accepts_downward_cap_with_live_but_unselected_decode(
    tmp_path, measured, selected
):
    report = check(tmp_path, guard_trace(measured=measured, selected=selected))
    assert report["quota_reasons"] == {"decode_budget_quota": 4}


@pytest.mark.parametrize(
    "fault",
    [
        "upward_cap",
        "fake_drain",
        "missing_active",
        "shape_change",
        "ttft",
        "unearned_estimate",
        "desired_floor",
        "protocol_downgrade",
        "first_slice",
        "sample_count",
        "minimum_progress",
        "first_layer_cost",
        "slack",
    ],
)
def test_guard_trace_rejects_quota_and_evidence_corruption(tmp_path, fault):
    rows = guard_trace(measured=True)
    guard = rows[1]["quota"]["decode_budget"]
    if fault == "upward_cap":
        guard["cap_layers"] = 2
    elif fault == "fake_drain":
        rows[1]["finish_prefill"] = True
    elif fault == "missing_active":
        guard["active_decode_ids"] = []
    elif fault == "shape_change":
        rows[3]["quota"]["decode_budget"]["prefill_shape_tokens"] = [32, 32]
    elif fault == "ttft":
        guard["ttft_feasible"] = False
    elif fault == "unearned_estimate":
        guard["prefill_samples"] = 0
    elif fault == "desired_floor":
        guard["desired_layers_per_slice"] = 1
    elif fault == "protocol_downgrade":
        del rows[3]["quota"]["decode_budget"]
    elif fault == "first_slice":
        guard["first_slice"] = False
    elif fault == "sample_count":
        guard["decode_samples"] = True
    elif fault == "minimum_progress":
        guard["minimum_progress_exceeds_budget"] = True
    elif fault == "first_layer_cost":
        guard["estimated_first_layer_ms"] = 10
    else:
        guard["decode_slack_ms"] = 100
    with pytest.raises(AssertionError):
        check(tmp_path, rows)


def residual_trace():
    rows = guard_trace(measured=True)
    for row in rows:
        if row["event"] != "decoupled_schedule":
            continue
        guard = row["quota"]["decode_budget"]
        guard.update(
            version=2,
            solo_decode_samples=1,
            residual_decode_samples=1,
            estimated_decode_cost_ms=60,
            estimated_decode_residual_ms=60,
            decode_estimate_source=(
                "solo_decode_round_trip"
                if row["layers_completed_before"] == 0
                else "fenced_continuation_residual"
            ),
        )
    complete = next(
        row for row in rows if row["event"] == "decoupled_complete" and row["step"] == 2
    )
    complete.update(
        duration_ms=70,
        prefill_slice_timing={
            "device_complete": True,
            "first_slice": False,
            "mode": "raw_layers",
            "device_elapsed_ms": 10,
            "host_submit_ms": 20,
        },
        decode_budget_sample={
            "phase": "raw_prefill_layer",
            "decode_residual_sample": {
                "accepted": True,
                "reason": "completed_fenced_continuation",
                "decode_shape": [1, 7, 7],
                "host_step_ms": 70,
                "prefill_device_elapsed_ms": 10,
                "prefill_host_submit_ms": 20,
                "residual_ms": 60,
            },
        },
    )
    return rows


def test_guard_v2_checks_residual_source_and_observation(tmp_path):
    assert check(tmp_path, residual_trace())["passed"]


@pytest.mark.parametrize(
    "fault",
    [
        "fresh_residual",
        "source",
        "count",
        "cost",
        "host_subtraction",
        "shape",
        "accept_negative",
        "first",
        "final",
        "incomplete",
    ],
)
def test_guard_v2_rejects_residual_corruption(tmp_path, fault):
    rows = residual_trace()
    scheduled = rows[3]
    complete = rows[4]
    guard = scheduled["quota"]["decode_budget"]
    sample = complete["decode_budget_sample"]["decode_residual_sample"]
    if fault == "fresh_residual":
        rows[1]["quota"]["decode_budget"]["decode_estimate_source"] = (
            "fenced_continuation_residual"
        )
    elif fault == "source":
        guard["decode_estimate_source"] = "solo_decode_round_trip"
    elif fault == "count":
        guard["residual_decode_samples"] = 0
    elif fault == "cost":
        guard["estimated_decode_cost_ms"] = 30
    elif fault == "host_subtraction":
        sample["residual_ms"] = 50
    elif fault == "shape":
        sample["decode_shape"] = [2, 8, 7]
    elif fault == "accept_negative":
        complete["duration_ms"] = sample["host_step_ms"] = 5
        sample["residual_ms"] = -5
    elif fault in ("first", "final"):
        # Test sample eligibility independently of unrelated layer arithmetic.
        scheduled["layers_completed_before"] = 0 if fault == "first" else 3
        complete["layers_completed"] = 1 if fault == "first" else 4
        with pytest.raises(AssertionError, match="continuation"):
            checker._check_decode_residual_sample(complete, scheduled)
        return
    else:
        complete["prefill_slice_timing"]["device_complete"] = False
    with pytest.raises(AssertionError):
        check(tmp_path, rows)


@pytest.mark.parametrize("host_ms", [5, 10])
def test_guard_v2_preserves_rejected_nonpositive_residual(tmp_path, host_ms):
    rows = residual_trace()
    complete = rows[4]
    sample = complete["decode_budget_sample"]["decode_residual_sample"]
    complete["duration_ms"] = sample["host_step_ms"] = host_ms
    sample.update(
        accepted=False,
        reason="nonpositive_or_nonfinite_residual",
        residual_ms=host_ms - 10,
    )
    assert check(tmp_path, rows)["passed"]
