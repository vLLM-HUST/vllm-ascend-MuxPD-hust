"""Strict SSE and timed arrival checks, using a CPU-only in-memory server."""

import asyncio
import base64
import copy
import json

import httpx
import pytest

from scripts import benchmark_contention as bench


def args(*extra):
    return bench.parse_args([
        "--base-url", "http://test", "--output", "unused.json",
        "--decode-requests", "2", "--decode-input-tokens", "3",
        "--decode-output-tokens", "4", "--prefill-input-tokens", "7",
        "--prefill-output-tokens", "2", "--repeats", "2",
        "--prefill-interval", "0", *extra,
    ])


def frames(payload, groups=None):
    groups = groups or [[100 + i] for i in range(payload["max_tokens"])]
    result = []
    for index, tokens in enumerate(groups):
        choice = {
            "index": 0, "text": "", "token_ids": tokens,
            "finish_reason": "length" if index == len(groups) - 1 else None,
        }
        if index == 0:
            choice["prompt_token_ids"] = payload["prompt"]
        result.append({"id": "cmpl-test", "choices": [choice]})
    result.append({
        "id": "cmpl-test", "choices": [],
        "usage": {
            "prompt_tokens": len(payload["prompt"]),
            "completion_tokens": payload["max_tokens"],
            "total_tokens": len(payload["prompt"]) + payload["max_tokens"],
        },
    })
    result.append("[DONE]")
    return result


def wire(frame, newline="\n"):
    data = frame if isinstance(frame, str) else json.dumps(frame)
    return ("data: " + data + newline + newline).encode()


def row_for(payload):
    return {"payload": payload, "started_monotonic_s": 10.0,
            "transport_passed": True}


def test_sse_split_bytes_prompt_ids_empty_text_and_exact_raw_body():
    payload = bench.request_plan(args())[0]["payload"]
    row = row_for(payload)
    signals = []
    receipt = bench.SSEReceipt(row, lambda: signals.append(True))
    body = b": heartbeat\r\n\r\n" + b"".join(
        wire(frame, "\r\n") for frame in frames(payload)
    )
    # Arbitrarily split JSON and CRLF separators over incoming HTTP bytes.
    for index in range(0, len(body), 7):
        receipt.feed(body[index:index + 7], 10.0 + index / 1000)
    receipt.finalize()
    receipt.preserve_raw()
    assert signals == [True]
    assert row["token_ids"] == [100, 101, 102, 103]
    assert row["text"] == ""
    assert base64.b64decode(row["raw_sse_base64"]) == body
    assert row["raw_sse_bytes"] == len(body)
    assert row["events"][-1]["kind"] == "done"


@pytest.mark.parametrize("fault", [
    "prompt", "missing_prompt", "token_bool", "missing_ids", "usage",
    "missing_done", "extra_done", "truncated", "response_id", "finish",
    "missing_finish", "after_finish", "too_many_tokens", "error",
])
def test_invalid_or_incomplete_stream_is_rejected(fault):
    payload = bench.request_plan(args())[0]["payload"]
    source = frames(payload)
    first = source[0]["choices"][0]
    if fault == "prompt":
        first["prompt_token_ids"] = [99]
    elif fault == "missing_prompt":
        del first["prompt_token_ids"]
    elif fault == "token_bool":
        first["token_ids"] = [True]
    elif fault == "missing_ids":
        del first["token_ids"]
    elif fault == "usage":
        source[-2]["usage"]["completion_tokens"] = True
    elif fault == "missing_done":
        source.pop()
    elif fault == "extra_done":
        source.append("[DONE]")
    elif fault == "response_id":
        source[1]["id"] = "different"
    elif fault == "finish":
        source[-3]["choices"][0]["finish_reason"] = "stop"
    elif fault == "missing_finish":
        source[-3]["choices"][0]["finish_reason"] = None
    elif fault == "after_finish":
        source.insert(-2, copy.deepcopy(source[0]))
    elif fault == "too_many_tokens":
        first["token_ids"] *= 6
    elif fault == "error":
        source[0] = {"error": {"message": "synthetic failure"}}
    body = b"".join(wire(frame) for frame in source)
    if fault == "truncated":
        body = body[:-1]
    receipt = bench.SSEReceipt(row_for(payload), lambda: None)
    with pytest.raises(ValueError):
        receipt.feed(body, 11.0)
        receipt.finalize()
    receipt.preserve_raw()
    assert base64.b64decode(receipt.row["raw_sse_base64"]) == body


def test_multi_token_chunks_do_not_invent_token_times_or_per_token_slo():
    payload = bench.request_plan(args())[0]["payload"]
    row = row_for(payload)
    receipt = bench.SSEReceipt(row, lambda: None)
    source = frames(payload, [[100], [101, 102], [103]])
    for frame, timestamp in zip(source, [10.1, 10.25, 10.29, 10.3, 10.3]):
        receipt.feed(wire(frame), timestamp)
    receipt.finalize()
    metrics = bench.request_metrics(row)
    assert metrics["token_event_gaps_s"] == pytest.approx([0.15, 0.04])
    assert metrics["single_token_events"] == 2
    assert metrics["multi_token_events"] == 1
    assert metrics["event_gap_slo"] == {"100": False, "200": True}
    assert metrics["single_token_observation_slo"] == {"100": None, "200": None}
    assert metrics["mean_tpot_s"] == pytest.approx(0.19 / 3)
    # The mean hides the 150 ms interruption, so max-gap SLO must stay false.
    assert metrics["mean_tpot_s"] < 0.1


def test_single_coalesced_chunk_has_no_measurable_gap_or_mean_tpot():
    payload = bench.request_plan(args())[0]["payload"]
    row = row_for(payload)
    receipt = bench.SSEReceipt(row, lambda: None)
    for frame in frames(payload, [[100, 101, 102, 103]]):
        receipt.feed(wire(frame), 11.0)
    receipt.finalize()
    metrics = bench.request_metrics(row)
    assert metrics["mean_tpot_s"] is None
    assert metrics["event_gap_slo"] == {"100": None, "200": None}


class DelayedStream(httpx.AsyncByteStream):
    def __init__(self, source, delay=0.003, initial_delay=0, truncate=False):
        self.source = source
        self.delay = delay
        self.initial_delay = initial_delay
        self.truncate = truncate

    async def __aiter__(self):
        await asyncio.sleep(self.initial_delay)
        for frame in self.source[:-1] if self.truncate else self.source:
            yield wire(frame)
            await asyncio.sleep(self.delay)


def run_mock(options, handle):
    checkpoints = []

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await bench.run_benchmark(
                options, client, lambda r: checkpoints.append(copy.deepcopy(r)),
            )

    return asyncio.run(execute()), checkpoints


def test_arrivals_wait_for_all_decode_first_tokens_and_preserve_every_repeat():
    options = args("--prefill-requests", "3")
    sent = []

    def handle(request):
        payload = json.loads(request.content)
        sent.append(payload)
        is_decode = len(payload["prompt"]) == options.decode_input_tokens
        delay = 0.012 if is_decode else 0.001
        # Stagger the two D first tokens; the P gate must wait for the latter.
        initial = 0.006 if is_decode and len(sent) % 2 else 0
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=DelayedStream(frames(payload), delay, initial))

    report, checkpoints = run_mock(options, handle)
    assert report["complete"]
    assert len(report["warmup"]) == 1 and len(report["repeats"]) == 2
    assert len(sent) == 15
    for run in report["warmup"] + report["repeats"]:
        ds = [r for r in run["requests"] if r["phase"] == "decode"]
        ps = [r for r in run["requests"] if r["phase"] == "prefill"]
        assert len(ds) == 2 and len(ps) == 3
        assert run["decode_gate_monotonic_s"] >= max(
            r["first_token_monotonic_s"] for r in ds)
        assert all(r["started_monotonic_s"] >= run["decode_gate_monotonic_s"]
                   for r in ps)
        assert all(len(r["active_decode_ids_at_dispatch"]) == 2 for r in ps)
        assert all(any(d["events_after_prefill_dispatch"]
                       for d in p["decode_continuation"]) for p in ps)
        assert run["summary"]["decode_completed"] == 2
        assert run["summary"]["prefill_completed"] == 3
        assert all(slo["all_requests_meet_slo"]
                   for slo in run["summary"]["joint_slo"])
    assert not checkpoints[0]["complete"] and checkpoints[-1]["complete"]
    assert report["summary"]["duration_s"]["count"] == 2
    # Request shapes/prompt hashes repeat exactly; warmup is separate evidence.
    assert [r["payload_sha256"] for r in report["repeats"][0]["requests"]] == [
        r["payload_sha256"] for r in report["warmup"][0]["requests"]]


def test_no_prefill_control_accepts_completed_decodes_and_has_no_fake_prefill():
    options = args("--prefill-requests", "0")

    def handle(request):
        payload = json.loads(request.content)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=b"".join(wire(f) for f in frames(payload)))

    report, _ = run_mock(options, handle)
    assert report["complete"]
    assert all(not run["contention_required"] for run in report["repeats"])
    assert report["summary"]["prefill_ttft_s"]["count"] == 0


def test_late_prefill_is_not_sent_or_counted_as_contention():
    options = args("--prefill-start-delay", "0.03")
    sent = []

    def handle(request):
        payload = json.loads(request.content)
        sent.append(payload)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=DelayedStream(frames(payload), 0.001))

    report, checkpoints = run_mock(options, handle)
    assert not report["complete"] and not report["repeats"]
    assert len(sent) == 2
    assert "No ongoing decode" in report["error"]
    assert all(slo["slo_request_goodput_per_s"] is None
               for slo in report["warmup"][0]["summary"]["joint_slo"])
    assert checkpoints[-1]["error"]


@pytest.mark.parametrize("fault", ["http", "truncated", "timeout"])
def test_failed_request_saves_partial_evidence_and_stops_later_repeats(fault):
    options = args("--timeout", "0.02" if fault == "timeout" else "2")

    def handle(request):
        payload = json.loads(request.content)
        if fault == "http":
            return httpx.Response(503, content=b"server unavailable")
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            stream=DelayedStream(frames(payload),
                                 initial_delay=0.1 if fault == "timeout" else 0,
                                 truncate=fault == "truncated"),
        )

    report, checkpoints = run_mock(options, handle)
    assert not report["complete"] and not report["repeats"]
    assert checkpoints[-1]["error"]
    rows = report["warmup"][0]["requests"]
    assert any(not row["transport_passed"] and row["error"] for row in rows)
    assert all("raw_sse_base64" in row for row in rows)
    if fault == "http":
        assert base64.b64decode(rows[0]["raw_sse_base64"]) == b"server unavailable"


def test_plan_reproducible_independent_request_lengths_and_phase_controls():
    first = bench.request_plan(args())
    assert first == bench.request_plan(args())
    assert first != bench.request_plan(args("--seed", "43"))
    assert len(first[0]["payload"]["prompt"]) == 3
    assert len(first[-1]["payload"]["prompt"]) == 7
    assert first[0]["payload"]["max_tokens"] == 4
    assert first[-1]["payload"]["max_tokens"] == 2
    assert all(item["payload"]["return_token_ids"] for item in first)
    assert len(bench.request_plan(args("--prefill-requests", "0"))) == 2


def test_all_tokens_received_but_connection_open_is_not_ongoing_decode():
    options = args()
    plan = bench.request_plan(options)

    def handle(request):
        payload = json.loads(request.content)
        # Each prompt's ID audit still has to match its own request.
        class Held(httpx.AsyncByteStream):
            async def __aiter__(self):
                for frame in frames(payload)[:-2]:
                    yield wire(frame)
                await asyncio.sleep(1)

        assert len(payload["prompt"]) == len(plan[0]["payload"]["prompt"])
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=Held())

    report, _ = run_mock(options, handle)
    assert not report["complete"]
    assert "No ongoing decode" in report["error"]
    assert len(report["warmup"][0]["requests"]) == 2


def test_failed_cli_returns_nonzero_and_writes_atomic_partial_evidence(
    tmp_path, monkeypatch,
):
    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _: httpx.Response(503, content=b"failed"))
    monkeypatch.setattr(
        bench.httpx, "AsyncClient",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )
    output = tmp_path / "partial.json"
    assert bench.main([
        "--base-url", "http://test", "--output", str(output),
    ]) == 1
    saved = json.loads(output.read_text())
    assert not saved["complete"] and saved["error"]
    assert saved["warmup"][0]["requests"][0]["http_status"] == 503
    assert not output.with_suffix(".json.tmp").exists()


def test_percentiles_are_linear_not_nearest_rank():
    assert bench.percentile([0, 1], 95) == pytest.approx(0.95)
    assert bench.percentile([], 99) is None
    assert bench.percentile([3], 50) == 3


def test_scoped_gaps_preserve_crossings_and_do_not_clip_stalls():
    def row(rid, timestamps):
        return {"request_id": rid, "events": [
            {"received_monotonic_s": stamp, "token_count": 1}
            for stamp in timestamps
        ]}

    selected = bench.gap_scope([
        row("crossing", [0.0, 0.2, 0.4, 0.6]),
        row("both", [0.0, 0.6]),
        row("before", [-1.0, 0.0]),
    ], 0.1, 0.5)
    groups = selected["intervals"]
    assert groups["fully_contained"][0]["gap_s"] == pytest.approx(0.2)
    assert groups["crossing_start"][0]["start_monotonic_s"] == 0.0
    assert groups["crossing_end"][0]["end_monotonic_s"] == 0.6
    assert groups["crossing_both"][0]["gap_s"] == pytest.approx(0.6)
    assert selected["all_intersecting_intact_gap_s"]["count"] == 4
    assert selected["all_intersecting_intact_gap_s"]["max"] == pytest.approx(0.6)


@pytest.mark.parametrize("extra", [
    ["--warmup-repeats", "0"], ["--decode-output-tokens", "1"],
    ["--prefill-requests", "-1"], ["--timeout", "nan"],
    ["--prefill-interval", "inf"], ["--prefill-start-delay", "-1"],
])
def test_invalid_cli_values_are_rejected(extra):
    with pytest.raises(SystemExit):
        args(*extra)
