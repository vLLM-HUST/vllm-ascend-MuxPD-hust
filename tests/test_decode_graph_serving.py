"""Strict HTTP/SSE receipts using fake streams only; no torch or device imports."""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/check_decode_graph_serving.py"
SPEC = importlib.util.spec_from_file_location("muxpd_graph_serving", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)
CASE = checker.cases()[0]


def wire(*, usage=True, done=True, finish="length", prompt_tokens=1):
    frames = [
        {
            "id": "cmpl-test",
            "choices": [{"index": 0, "text": "你好", "finish_reason": None}],
        },
        {
            "id": "cmpl-test",
            "choices": [{"index": 0, "text": "", "finish_reason": finish}],
        },
    ]
    if usage:
        frames.append(
            {
                "id": "cmpl-test",
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 16,
                    "total_tokens": prompt_tokens + 16,
                },
            }
        )
    body = "".join(
        "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n" for frame in frames
    )
    return body + ("data: [DONE]\n\n" if done else "")


def parse(body):
    parser = checker.CompletionSSE()
    parser.feed(body, 1.0)
    parser.finish(CASE)
    return parser


@pytest.mark.parametrize(
    "body,match",
    [
        (wire(usage=False), "require DONE, finish and usage"),
        (wire(done=False), "require DONE, finish and usage"),
        (wire(finish=None), "require DONE, finish and usage"),
        (wire(finish="stop"), "reason=length"),
        (wire(prompt_tokens=2), "prompt_tokens"),
        (wire().rstrip("\n"), "Partial SSE"),
        (wire() + "data: [DONE]\n\n", "after \\[DONE\\]"),
        ("data: {broken}\n\n", "Expecting property name"),
    ],
)
def test_invalid_transports_fail_closed(body, match):
    with pytest.raises(ValueError, match=match):
        parse(body)


def test_valid_usage_length_hash_and_exact_one_token_prompt():
    parser = parse(wire())
    assert parser.text == "你好"
    assert (
        parser.receipt()["text_sha256"] == hashlib.sha256("你好".encode()).hexdigest()
    )
    assert parser.usage["prompt_tokens"] == 1
    assert parser.first_choice_s == parser.first_text_s == 1.0
    assert parser.finish_reason == "length" and parser.done
    with pytest.raises(ValueError, match="completion_tokens"):
        parse(wire().replace('"completion_tokens": 16', '"completion_tokens": 15'))


class Stream(httpx.AsyncByteStream):
    def __init__(self, fail=False):
        self.fail = fail

    async def __aiter__(self):
        body = wire().encode()
        # Split inside a UTF-8 code point to exercise incremental decoding.
        offset = body.index("你".encode()) + 1
        yield body[:offset]
        if self.fail:
            raise httpx.RemoteProtocolError("incomplete chunked response")
        yield body[offset:]


def request(*, fail=False):
    async def run():
        def handler(request):
            payload = json.loads(request.content)
            assert payload["prompt"] == [100]
            assert payload["add_special_tokens"] is False
            assert payload["ignore_eos"] is True
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, stream=Stream(fail)
            )

        gate = asyncio.Event()
        gate.set()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await checker.stream_case(
                client, NS(model="test"), "http://fake", CASE, 0, gate, [0.0]
            )

    return asyncio.run(run())


def test_collector_preserves_raw_sse_and_actual_receipt_times():
    row = request()
    assert row["transport_passed"] and "error" not in row
    assert row["sse_body"] == wire()
    assert (
        row["started_monotonic_s"]
        <= row["first_choice_monotonic_s"]
        <= row["ended_monotonic_s"]
    )
    assert row["http_status"] == 200
    assert len(row["events"]) == 3


def test_partial_http_stream_is_retained_as_failure():
    row = request(fail=True)
    assert not row["transport_passed"]
    assert "RemoteProtocolError" in row["error"]
    assert not row["done_received"]
    assert row["sse_body"]


def test_baseline_reuse_revalidates_raw_sse_and_complete_coverage(tmp_path):
    row = request()
    report = {
        "schema_version": 1,
        "complete": True,
        "parameters": {"model": "test", "repeats": 1},
        "cases": [CASE],
        "generation": checker.GENERATION,
        "endpoints": [{"label": "baseline", "requests": [row]}],
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(report))
    endpoint = checker.load_baseline(path, NS(model="test", repeats=1), [CASE])
    assert (
        endpoint["reused_report_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    report["endpoints"][0]["requests"][0]["text"] = "tampered"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="differs from raw SSE"):
        checker.load_baseline(path, NS(model="test", repeats=1), [CASE])
    report["endpoints"][0]["requests"] = []
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="coverage"):
        checker.load_baseline(path, NS(model="test", repeats=1), [CASE])


def test_case_groups_and_summary_do_not_claim_device_graphs():
    cases = checker.cases()
    groups = {
        name: [case for case in cases if case["group"] == name]
        for name in dict.fromkeys(case["group"] for case in cases)
    }
    assert [len(rows) for rows in groups.values()] == [1, 1, 2, 3, 5, 8, 8]
    assert len({row["max_tokens"] for row in groups["shrinking8"]}) > 1
    row = request()
    summary = checker.summarize(
        [
            {"label": "baseline", "requests": [row]},
            {"label": "candidate", "requests": [row]},
        ]
    )
    assert summary["cross_endpoint_text_matches"][0]["exact_text_match"]
    assert summary["device_graph_execution_proven"] is False
    assert summary["actual_decode_batch_sizes_proven"] is False
