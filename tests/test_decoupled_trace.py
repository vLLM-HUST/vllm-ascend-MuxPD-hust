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
