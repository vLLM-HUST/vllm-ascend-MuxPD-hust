"""No network or accelerator required for the serial graph-state probe."""

import copy

import httpx
import pytest

from scripts import probe_graph_state_reuse as probe


def args():
    return probe.parse_args([
        "--base-url", "http://test", "--output", "unused.json",
        "--repeats", "2", "--long-prompt-tokens", "5",
    ])


def response(payload, first_token=10):
    length = payload["max_tokens"]
    return {
        "id": "cmpl-synthetic",
        "choices": [{
            "index": 0,
            "token_ids": [first_token] + [11] * (length - 1),
            "prompt_token_ids": payload["prompt"],
            "text": "",  # Special or incomplete byte tokens can produce no text.
            "finish_reason": "length",
        }],
        "usage": {
            "prompt_tokens": len(payload["prompt"]),
            "completion_tokens": length,
            "total_tokens": len(payload["prompt"]) + length,
        },
    }


def test_plan_brackets_each_case_and_preserves_original_one_token_payload():
    plan = list(probe.request_plan(args()))
    assert len(plan) == 30
    assert plan[0]["payload"]["prompt"] == [100]
    assert plan[0]["payload"]["max_tokens"] == 1
    for start in range(0, len(plan), 5):
        bracket = plan[start:start + 5]
        assert [r["stage"] for r in bracket] == [
            "before", "before", "background", "after", "after",
        ]
        assert bracket[2]["payload"]["prompt"] == [9419, 12675, 9419, 12675, 9419]
        assert len({r["case"] for r in bracket}) == 1
        assert all(r["payload"]["return_token_ids"] for r in bracket)
        assert not any(r["payload"]["stream"] for r in bracket)
        assert bracket[0]["payload"] == bracket[4]["payload"]


def test_serial_probe_reports_token_differences_and_all_receipts():
    plan = list(probe.request_plan(args()))
    sent = []
    checkpoints = []

    def handle(request):
        import json

        payload = json.loads(request.content)
        item = plan[len(sent)]
        assert payload == item["payload"]
        sent.append(payload)
        token = 12 if item["stage"] == "after" else 10
        return httpx.Response(200, json=response(payload, token))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        report = probe.run_probe(
            args(), client, lambda r: checkpoints.append(copy.deepcopy(r))
        )
    assert len(sent) == len(report["requests"]) == 30
    assert report["complete"] and report["transport_passed"]
    assert not checkpoints[0]["complete"] and checkpoints[-1]["complete"]
    assert all(row["before"]["stable"] for row in report["summary"])
    assert all(row["after"]["stable"] for row in report["summary"])
    assert all(not row["first_before_matches_first_after"]
               for row in report["summary"])
    assert all(row["first_before_vs_first_after_difference"] == {
        "token_index": 0, "left_token_id": 10, "right_token_id": 12,
    } for row in report["summary"])
    assert all(row["usage"] == row["response"]["usage"]
               for row in report["requests"])
    assert all(a["finished_monotonic_s"] <= b["started_monotonic_s"]
               for a, b in zip(report["requests"], report["requests"][1:]))


@pytest.mark.parametrize("failure", ["timeout", "http", "missing_ids"])
def test_failed_request_stops_probe_before_any_following_request(failure):
    sent = []

    def handle(request):
        import json

        payload = json.loads(request.content)
        sent.append(payload)
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        data = response(payload)
        if failure == "missing_ids":
            del data["choices"][0]["token_ids"]
        return httpx.Response(500 if failure == "http" else 200, json=data)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        report = probe.run_probe(args(), client, lambda _: None)
    assert len(sent) == 1
    assert not report["complete"] and not report["transport_passed"]
    assert report["requests"][0]["error"]
    assert all(row["before"]["stable"] is None for row in report["summary"])


@pytest.mark.parametrize("field", ["usage", "prompt_ids", "token_type", "finish"])
def test_invalid_response_is_not_counted_as_stable(field):
    payload = next(probe.request_plan(args()))["payload"]
    data = response(payload)
    if field == "usage":
        data["usage"]["completion_tokens"] = True
    elif field == "prompt_ids":
        data["choices"][0]["prompt_token_ids"] = [9419]
    elif field == "token_type":
        data["choices"][0]["token_ids"] = [True]
    else:
        data["choices"][0]["finish_reason"] = "stop"
    with pytest.raises(ValueError):
        probe.validate_response(data, payload)


def test_first_difference_handles_length_and_repeat_instability():
    assert probe.first_difference([1], [1]) is None
    assert probe.first_difference([1], [1, 2]) == {
        "token_index": 1, "left_token_id": None, "right_token_id": 2,
    }
    rows = [
        {"transport_passed": True, "token_ids": [1, 2]},
        {"transport_passed": True, "token_ids": [1, 3]},
    ]
    summary = probe.repeat_summary(rows, 2)
    assert summary["stable"] is False
    assert summary["distinct_token_sequences"] == 2
    assert summary["first_difference_from_repeat_zero"]["token_index"] == 1


@pytest.mark.parametrize("flag,value", [
    ("--repeats", "1"), ("--timeout", "nan"), ("--timeout", "0"),
    ("--long-prompt-tokens", "1"), ("--long-max-tokens", "0"),
])
def test_invalid_arguments_fail_before_network(flag, value):
    with pytest.raises(SystemExit):
        probe.parse_args([
            "--base-url", "http://test", "--output", "unused", flag, value,
        ])
