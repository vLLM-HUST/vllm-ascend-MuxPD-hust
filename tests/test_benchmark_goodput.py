"""CPU-only open-loop, failure retention, and non-interchangeable SLO metrics."""

import asyncio
import base64
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts import benchmark_goodput as bench


def parameters(**changes):
    return {
        **bench.GENERATOR_DEFAULTS,
        "seeds": [1],
        "warmup": 0.02,
        "duration": 0.1,
        "request_rate": 100,
        "short_input_tokens": 3,
        "long_input_tokens": 5,
        "short_output_tokens": 2,
        "long_output_tokens": 2,
        **changes,
    }


def seal(plan):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = bench.digest(body)
    return plan


def fixed_plan(arrivals=(0.0, 0.005, 0.01), **changes):
    p = parameters(**({"warmup": 0, "duration": 0.025} | changes))
    items = []
    for i, arrival in enumerate(arrivals):
        payload = {
            "model": p["model"],
            "prompt": [100, 101, 102],
            "max_tokens": 2,
            "temperature": 0,
            "seed": 1,
            "ignore_eos": True,
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        }
        items.append(
            {
                "id": f"seed1-request{i}",
                "class": "short",
                "phase": "warmup" if arrival < p["warmup"] else "measurement",
                "arrival_offset_s": arrival,
                "payload": payload,
                "payload_sha256": bench.digest(payload),
                "prompt_sha256": bench.digest(payload["prompt"]),
            }
        )
    return bench.validate_plan(
        seal(
            {
                "schema": bench.PLAN_SCHEMA,
                "parameters": p,
                "runs": [{"seed": 1, "requests": items}],
            }
        )
    )


def options(**changes):
    return SimpleNamespace(
        **(
            {
                "base_url": "http://fake",
                "drain_timeout": 0.2,
                "request_timeout": 1,
                "checkpoint_interval": 0.005,
                "max_send_lag": 0.5,
                "label": "fixture",
            }
            | changes
        )
    )


def frames(payload):
    values = [
        {
            "id": "cmpl",
            "choices": [
                {
                    "index": 0,
                    "text": "a",
                    "token_ids": [110],
                    "prompt_token_ids": payload["prompt"],
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "cmpl",
            "choices": [
                {
                    "index": 0,
                    "text": "b",
                    "token_ids": [111],
                    "finish_reason": "length",
                }
            ],
        },
        {
            "id": "cmpl",
            "choices": [],
            "usage": {
                "prompt_tokens": len(payload["prompt"]),
                "completion_tokens": 2,
                "total_tokens": len(payload["prompt"]) + 2,
            },
        },
        "[DONE]",
    ]
    return values


def wire(value):
    return (
        "data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n"
    ).encode()


class Stream(httpx.AsyncByteStream):
    def __init__(self, body, delay, state):
        self.body, self.delay, self.state = body, delay, state

    async def __aiter__(self):
        yield self.body[0]
        await asyncio.sleep(self.delay)
        for chunk in self.body[1:]:
            yield chunk

    async def aclose(self):
        self.state["active"] -= 1


def run_fake(
    tmp_path, plan=None, args=None, fault=None, delay=0.03, progress_error=False
):
    plan, args = plan or fixed_plan(), args or options()
    state = {"active": 0, "max_active": 0, "requests": 0}
    snapshots, progress = [], []
    journal = bench.Journal(tmp_path / "events.jsonl", flush_interval=0.005)

    def checkpoint(run, point):
        if progress_error:
            raise OSError("Synthetic checkpoint failure")
        progress.append(copy.deepcopy(point))

    async def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        state["active"] += 1
        state["requests"] += 1
        state["max_active"] = max(state["active"], state["max_active"])
        payload = json.loads(request.content)
        values = frames(payload)
        if fault:
            fault(values)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Stream([wire(v) for v in values], delay, state),
        )

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await bench.run_benchmark(
                args,
                plan,
                client,
                journal,
                lambda value: snapshots.append(copy.deepcopy(value)),
                checkpoint,
            )

    try:
        result = asyncio.run(execute())
    finally:
        journal.close()
    lines = [json.loads(line) for line in journal.path.read_text().splitlines()]
    return result, state, snapshots, progress, lines


def test_plan_is_seeded_immutable_and_arrivals_independent_of_shape():
    first = bench.build_plan(parameters(seeds=[1, 2]))
    assert first == bench.build_plan(parameters(seeds=[1, 2]))
    changed = bench.build_plan(parameters(seeds=[1, 2], long_input_tokens=17))
    for old_run, new_run in zip(first["runs"], changed["runs"], strict=True):
        assert [(r["arrival_offset_s"], r["class"]) for r in old_run["requests"]] == [
            (r["arrival_offset_s"], r["class"]) for r in new_run["requests"]
        ]
    altered = copy.deepcopy(first)
    altered["runs"][0]["requests"][0]["arrival_offset_s"] += 0.001
    with pytest.raises(ValueError, match="Plan SHA"):
        bench.validate_plan(altered)


@pytest.mark.parametrize("mutation", ["arrival", "prompt", "usage", "phase", "bool"])
def test_rehashed_but_invalid_plan_is_rejected(mutation):
    plan = fixed_plan()
    row = plan["runs"][0]["requests"][1]
    if mutation == "arrival":
        row["arrival_offset_s"] = -1
    elif mutation == "prompt":
        row["payload"]["prompt"][0] = True
        row["prompt_sha256"] = bench.digest(row["payload"]["prompt"])
        row["payload_sha256"] = bench.digest(row["payload"])
    elif mutation == "usage":
        row["payload"]["stream_options"]["include_usage"] = False
        row["payload_sha256"] = bench.digest(row["payload"])
    elif mutation == "phase":
        row["phase"] = "warmup"
    else:
        plan["parameters"]["request_rate"] = True
    with pytest.raises(ValueError):
        bench.validate_plan(seal(plan))


@pytest.mark.parametrize(
    "changes",
    [
        {"request_rate": 0},
        {"duration": float("nan")},
        {"warmup": -1},
        {"long_fraction": 2},
        {"seeds": [1, 1]},
        {"short_output_tokens": 1},
    ],
)
def test_invalid_parameters_fail_before_any_generation(changes):
    with pytest.raises(ValueError):
        bench.build_plan(parameters(**changes))


def test_plan_only_needs_no_endpoint_and_replay_rejects_generation_overrides(tmp_path):
    destination = tmp_path / "plan.json"
    assert (
        bench.main(
            [
                "--plan-only",
                "--output",
                str(destination),
                "--duration",
                ".1",
                "--warmup",
                "0",
                "--request-rate",
                "100",
                "--seeds",
                "1",
                "--short-input-tokens",
                "3",
                "--long-input-tokens",
                "5",
            ]
        )
        == 0
    )
    plan = bench.validate_plan(json.loads(destination.read_text()))
    assert plan["parameters"]["duration"] == 0.1
    with pytest.raises(SystemExit):
        bench.parse_args(
            [
                "--plan",
                str(destination),
                "--output",
                "run.json",
                "--base-url",
                "http://fake",
                "--request-rate",
                "1",
            ]
        )
    with pytest.raises(ValueError, match="already exists"):
        bench.main(["--plan-only", "--output", str(destination)])


def test_overlapping_requests_are_not_semaphore_gated_and_raw_is_preserved(tmp_path):
    result, state, snapshots, progress, journal = run_fake(tmp_path)
    assert result["complete"]
    assert state["max_active"] == 3
    rows = result["runs"][0]["requests"]
    assert max(r["started_monotonic_s"] for r in rows) < min(
        r["finished_monotonic_s"] for r in rows
    )
    assert snapshots[0]["complete"] is False and snapshots[-1]["complete"] is True
    assert progress
    for row in rows:
        body = b"".join(
            base64.b64decode(e["body_base64"])
            for e in journal
            if e["kind"] == "http_body" and e["request_id"] == row["request_id"]
        )
        assert body == base64.b64decode(row["raw_sse_base64"])
        assert row["usage"]["completion_tokens"] == 2
    summary = result["runs"][0]["summary"]
    assert summary["arrival_window_s"] == 0.025
    assert summary["joint_slos"][0]["single_token_request_goodput_per_s"] == 120


@pytest.mark.parametrize("fault", ["prompt", "usage", "done"])
def test_protocol_errors_keep_bytes_and_do_not_hide_other_arrivals(tmp_path, fault):
    def break_stream(values):
        if fault == "prompt":
            values[0]["choices"][0]["prompt_token_ids"] = [999]
        elif fault == "usage":
            values[-2]["usage"]["completion_tokens"] = 1
        else:
            values.pop()

    result, state, _, _, journal = run_fake(tmp_path, fault=break_stream)
    assert not result["complete"]
    assert state["requests"] == 3
    rows = result["runs"][0]["requests"]
    assert all(r["status"] == "failed" and r["raw_sse_bytes"] > 0 for r in rows)
    assert sum(e["kind"] == "request_finished" for e in journal) == 3
    summary = result["runs"][0]["summary"]
    assert summary["measured_incomplete_or_failed_requests"] == 3
    assert all(
        s["single_token_request_goodput_per_s"] == 0
        and not s["paper_like_pooled_gap_p99_passed"]
        for s in summary["joint_slos"]
    )


def test_drain_deadline_preserves_partial_sse_and_fixed_denominator(tmp_path):
    result, _, _, _, journal = run_fake(
        tmp_path, args=options(drain_timeout=0.01), delay=1
    )
    run = result["runs"][0]
    assert not result["complete"] and run["drain_deadline_exceeded"]
    assert all(r["status"] == "cancelled_at_deadline" for r in run["requests"])
    assert all(
        r["token_ids"] == [110] and r["raw_sse_bytes"] > 0 for r in run["requests"]
    )
    assert run["summary"]["arrival_window_s"] == 0.025
    assert journal[-1]["kind"] == "benchmark_finished"
    assert not journal[-1]["complete"]


def test_monitor_failure_retains_current_run_and_prevents_success(tmp_path):
    result, _, snapshots, _, journal = run_fake(tmp_path, progress_error=True)
    assert not result["complete"]
    assert len(result["runs"]) == 1
    run = result["runs"][0]
    assert not run["complete"] and "checkpoint failure" in run["monitor_error"]
    assert all(
        r["transport_passed"] and r["raw_sse_bytes"] > 0 for r in run["requests"]
    )
    assert snapshots[-1]["runs"][0]["monitor_error"] == run["monitor_error"]
    assert all(
        not s["eligible_for_capacity_comparison"] for s in run["summary"]["joint_slos"]
    )
    assert any(e["kind"] == "run_finished" and not e["complete"] for e in journal)


def test_external_cancel_before_first_dispatch_keeps_all_planned_requests(tmp_path):
    plan = fixed_plan(arrivals=(1.0, 1.5), duration=2)
    journal = bench.Journal(tmp_path / "cancel.jsonl")

    async def execute():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: pytest.fail("No request should have been dispatched")
            )
        ) as client:
            task = asyncio.create_task(
                bench.run_once(
                    client,
                    options(),
                    plan["runs"][0],
                    plan["parameters"],
                    journal,
                    lambda run, point: None,
                )
            )
            await asyncio.sleep(0.005)
            task.cancel()
            return await task

    try:
        run = asyncio.run(execute())
    finally:
        journal.close()
    assert not run["complete"]
    assert "CancelledError" in run["controller_error"]
    assert all(r["status"] == "planned" for r in run["requests"])
    assert run["summary"]["planned_measured_requests"] == 2
    assert run["summary"]["joint_slos"][0]["single_token_request_attainment"] == 0


def test_consumer_exception_is_saved_and_failed_seed_stops_later_seeds(tmp_path):
    plan = fixed_plan()
    second = copy.deepcopy(plan["runs"][0])
    second["seed"] = 2
    for i, row in enumerate(second["requests"]):
        row["id"] = f"seed2-request{i}"
        row["payload"]["seed"] = 2
        row["payload_sha256"] = bench.digest(row["payload"])
    plan["parameters"]["seeds"].append(2)
    plan["runs"].append(second)
    bench.validate_plan(seal(plan))
    journal = bench.Journal(tmp_path / "exception.jsonl")
    saved = []

    async def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        raise httpx.ReadError("Synthetic transport failure")

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await bench.run_benchmark(
                options(),
                plan,
                client,
                journal,
                lambda report: saved.append(copy.deepcopy(report)),
                lambda run, point: None,
            )

    try:
        result = asyncio.run(execute())
    finally:
        journal.close()
    assert len(result["runs"]) == 1
    assert "stopped_early_reason" in result and not result["complete"]
    rows = result["runs"][0]["requests"]
    assert all("ReadError" in r["error"] and r["raw_sse_bytes"] == 0 for r in rows)
    assert (
        saved[-1]["runs"][0]["summary"]["measured_incomplete_or_failed_requests"] == 3
    )


def metric_row(
    times,
    *,
    scheduled=0,
    started=None,
    phase="measurement",
    kind="short",
    counts=None,
    complete=True,
):
    counts = counts or [1] * len(times)
    return {
        "phase": phase,
        "class": kind,
        "scheduled_monotonic_s": scheduled,
        "started_monotonic_s": scheduled if started is None else started,
        "finished_monotonic_s": times[-1] + 0.001,
        "transport_passed": complete,
        "events": [
            {"received_monotonic_s": t, "token_count": n}
            for t, n in zip(times, counts, strict=True)
        ],
        "token_ids": [100] * sum(counts),
    }


def metric_run(rows, start=0, end=10):
    return {
        "complete": True,
        "requests": rows,
        "origin_monotonic_s": 0,
        "measurement_start_monotonic_s": start,
        "arrival_end_monotonic_s": end,
        "backlog_samples": [],
    }


def test_pooled_p99_does_not_mean_every_request_meets_max_gap():
    times = [0.1]
    for i in range(255):
        times.append(times[-1] + (0.39 if i == 100 else 0.01))
    summary = bench.summarize_run(
        metric_run([metric_row(times)]), {"duration": 10}, 0.05
    )
    strict = summary["joint_slos"][0]
    assert summary["pooled_observed_event_gap_s"]["p99"] < 0.011
    assert strict["paper_like_pooled_gap_p99_passed"]
    assert strict["pooled_gap_and_ttft_p99_passed"]
    assert strict["single_token_qualified_requests"] == 0


def test_both_slos_apply_to_both_request_classes_and_use_planned_arrival():
    # Short request TTFT fails; long request decode gap fails. Role-only scoring
    # would have counted both. Late client dispatch cannot erase the first miss.
    rows = [metric_row([2.1, 2.11], started=2.05), metric_row([0.1, 0.4], kind="long")]
    summary = bench.summarize_run(metric_run(rows), {"duration": 10}, 0.05)
    assert rows[0]["metrics"]["started_ttft_s"] == pytest.approx(0.05)
    assert rows[0]["metrics"]["scheduled_ttft_s"] == 2.1
    assert not summary["client_dispatch_valid"]
    assert all(s["single_token_qualified_requests"] == 0 for s in summary["joint_slos"])


def test_multitoken_chunks_cannot_pass_single_token_slo():
    row = metric_row([0.1, 0.12], counts=[1, 2])
    summary = bench.summarize_run(metric_run([row]), {"duration": 10}, 0.05)
    for slo in summary["joint_slos"]:
        assert slo["event_gap_qualified_requests"] == 1
        assert slo["single_token_qualified_requests"] == 0
        assert slo["single_token_unobservable_requests"] == 1
        assert not slo["paper_like_pooled_gap_p99_passed"]


def test_fixed_window_throughput_and_arrival_cohort_goodput_have_distinct_edges():
    rows = [
        metric_row([0.6, 1.1, 1.5], scheduled=0.5, phase="warmup"),
        metric_row([1.8, 1.85, 2.01], scheduled=1.7),
    ]
    run = metric_run(rows, start=1, end=2)
    summary = bench.summarize_run(run, {"duration": 1}, 0.05)
    assert summary["fixed_window_output"]["validated_tokens"] == 4
    assert summary["completed_cohort_output_tokens"] == 3
    assert summary["joint_slos"][2]["single_token_request_goodput_per_s"] == 1
    assert summary["backlog"]["at_measurement_start"]["outstanding"] == 1
    assert summary["backlog"]["at_arrival_end"]["outstanding"] == 1


def test_failed_requests_still_count_in_denominator_and_disable_capacity_check():
    good = metric_row([0.1, 0.11])
    failed = metric_row([0.1], complete=False)
    summary = bench.summarize_run(metric_run([good, failed]), {"duration": 10}, 0.05)
    slo = summary["joint_slos"][0]
    assert slo["single_token_request_goodput_per_s"] == 0.1
    assert slo["single_token_request_attainment"] == 0.5
    assert not slo["eligible_for_capacity_comparison"]
    assert not slo["paper_like_pooled_gap_p99_passed"]


@pytest.mark.parametrize("failure", ["complete", "controller_error", "monitor_error"])
def test_control_failure_cannot_qualify_successful_transport_for_capacity(failure):
    run = metric_run([metric_row([0.1, 0.11])])
    run[failure] = False if failure == "complete" else "Synthetic control failure"
    summary = bench.summarize_run(run, {"duration": 10}, 0.05)
    assert summary["all_transport_passed_including_warmup"]
    assert all(s["paper_like_pooled_gap_p99_passed"] for s in summary["joint_slos"])
    assert all(not s["eligible_for_capacity_comparison"] for s in summary["joint_slos"])
