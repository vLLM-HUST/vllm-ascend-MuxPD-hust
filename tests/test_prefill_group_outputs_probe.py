"""CPU-only HTTP streams: no model, service, NPU, or network access."""

import asyncio
import base64
import copy
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts import probe_prefill_group_outputs as probe


def frame(value):
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ("data: " + data + "\n\n").encode()


def choice(payload, tokens, *, first=False, finish=False, text="\u2029"):
    value = {
        "index": 0,
        "text": text,
        "token_ids": tokens,
        "finish_reason": "length" if finish else None,
    }
    if first:
        value["prompt_token_ids"] = payload["prompt"]
    return {"id": "cmpl-" + payload["request_id"], "choices": [value]}


def ending(payload):
    prompt, outputs = len(payload["prompt"]), payload["max_tokens"]
    return [
        frame(
            {
                "id": "cmpl-" + payload["request_id"],
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": outputs,
                    "total_tokens": prompt + outputs,
                },
            }
        ),
        frame("[DONE]"),
    ]


class Stream(httpx.AsyncByteStream):
    def __init__(self, iterator):
        self.iterator = iterator
        self.closed = False

    async def __aiter__(self):
        async for chunk in self.iterator:
            yield chunk

    async def aclose(self):
        self.closed = True
        await self.iterator.aclose()


class Server:
    """Hold D alive until P completes, without relying on scheduling sleeps."""

    def __init__(
        self,
        *,
        broken=None,
        break_kind="tokens",
        blocked=False,
        finish_decode_early=False,
    ):
        self.seen = []
        self.streams = []
        self.broken, self.break_kind = broken, break_kind
        self.blocked, self.finish_decode_early = blocked, finish_decode_early
        self.prefill_started = [asyncio.Event(), asyncio.Event()]
        self.prefill_done = [asyncio.Event(), asyncio.Event()]
        self.second_tokens = [[asyncio.Event() for _ in range(4)] for _ in range(2)]

    def handle(self, request):
        payload = json.loads(request.content)
        index = int(payload["request_id"].rsplit("-", 1)[1])
        self.seen.append(index)
        assert request.headers["x-request-id"] == payload["request_id"]
        assert not any(key in payload for key in ("logprobs", "prompt_logprobs"))
        stream = Stream(self.generate(index, payload))
        self.streams.append(stream)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream
        )

    async def generate(self, index, payload):
        n = payload["max_tokens"]
        # Deliberately vary two repeats at token 4: numerical differences must
        # not stop the remaining fixed requests or be classified as transport.
        tokens = list(range(1000, 1000 + n))
        if index in (1, 17):
            tokens[4] += 20
        if index == self.broken:
            value = choice(payload, tokens, first=True, finish=True)
            if self.break_kind == "tokens":
                value["choices"][0]["token_ids"][0] = True
            elif self.break_kind == "prompt":
                value["choices"][0]["prompt_token_ids"] = [8]
            yield frame(value)
            tails = ending(payload)
            if self.break_kind == "usage":
                usage = json.loads(tails[0][6:].strip())
                usage["usage"]["completion_tokens"] -= 1
                tails[0] = frame(usage)
            for chunk in tails[:1] if self.break_kind == "done" else tails:
                yield chunk
            return
        if n == 128 and not self.finish_decode_early:
            repeat, lane = (0, index - 6) if index < 11 else (1, index - 11)
            if self.blocked:
                await asyncio.Event().wait()
            yield frame(choice(payload, tokens[:1], first=True))
            await self.prefill_started[repeat].wait()
            yield frame(choice(payload, tokens[1:2]))
            self.second_tokens[repeat][lane].set()
            await self.prefill_done[repeat].wait()
            yield frame(choice(payload, tokens[2:], finish=True))
        else:
            repeat = {10: 0, 15: 1}.get(index)
            if repeat is not None:
                self.prefill_started[repeat].set()
                await asyncio.gather(
                    *(event.wait() for event in self.second_tokens[repeat])
                )
            # Split a Unicode-containing UTF-8 frame between bytes, exercising
            # the real strict parser without treating U+2029 as an SSE newline.
            chunk = frame(choice(payload, tokens, first=True, finish=True))
            yield chunk[:31]
            yield chunk[31:]
        for chunk in ending(payload):
            yield chunk
        if index in (10, 15):
            self.prefill_done[{10: 0, 15: 1}[index]].set()


def report(plan):
    return {
        "schema": probe.REPORT_SCHEMA,
        "plan": plan,
        "label": "fake",
        "run_id": "cpu",
        "requests": [],
        "phases": [],
        "complete": False,
    }


async def collect(tmp_path, server, *, gate_timeout=1, timeout=2, cancel=False):
    plan = probe.build_plan()
    result = report(plan)
    journal = probe.Journal(tmp_path / "events.jsonl")
    output = tmp_path / "report.json"
    args = SimpleNamespace(
        base_url="http://never-network.invalid",
        timeout=timeout,
        gate_timeout=gate_timeout,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(server.handle)
        ) as client:
            task = asyncio.create_task(
                probe.run_probe(
                    plan,
                    args,
                    client,
                    result,
                    journal,
                    lambda: probe.checkpoint(output, result),
                )
            )
            if cancel:
                # Wait for the deliberately blocked first mixed D cohort.
                while len(server.seen) < 10:
                    await asyncio.sleep(0)
                task.cancel()
            await task
    finally:
        journal.close()
    assert json.loads(output.read_text()) == result
    assert all(stream.closed for stream in server.streams)
    return result


def test_plan_exact_count_history_and_fixed_sampling():
    plan = probe.build_plan()
    probe.validate_plan(plan)
    assert len(plan["requests"]) == plan["generation_requests"] == 19
    assert [phase["indices"] for phase in plan["phases"]] == [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16],
        [17],
        [18],
    ]
    assert all(
        plan["requests"][i]["payload"] == plan["requests"][0]["payload"]
        for i in (1, 2, 16, 17, 18)
    )
    assert all(
        plan["requests"][i]["payload"] == plan["requests"][3]["payload"]
        for i in (4, 5, 10, 15)
    )
    for row in plan["requests"]:
        payload = row["payload"]
        assert payload["temperature"] == 0 and payload["seed"] == 42
        assert payload["ignore_eos"] and payload["return_token_ids"]
        assert probe.digest(payload) == row["payload_sha256"]
        assert "request_id" not in payload and "logprobs" not in payload


@pytest.mark.parametrize("mutation", ["count", "payload", "logprobs", "history"])
def test_plan_changes_rejected(mutation):
    plan = probe.build_plan()
    if mutation == "count":
        plan["generation_requests"] = 18
    elif mutation == "payload":
        plan["requests"][0]["payload"]["prompt"][0] += 1
    elif mutation == "logprobs":
        plan["requests"][0]["payload"]["logprobs"] = 2
    else:
        plan["phases"][0], plan["phases"][1] = plan["phases"][1], plan["phases"][0]
    with pytest.raises(ValueError, match="fixed 19-request"):
        probe.validate_plan(plan)


def test_complete_collection_preserves_differences_and_real_gate(tmp_path):
    server = Server()
    result = asyncio.run(collect(tmp_path, server))
    assert result["complete"] and server.seen == list(range(19))
    groups = result["summary"]["repeat_groups"]
    assert groups[0]["stable"] is False and groups[1]["stable"] is True
    assert groups[0]["pairwise_first_differences"][0]["first_difference"] == {
        "token_index": 4,
        "left_token_id": 1004,
        "right_token_id": 1024,
    }
    assert len(result["summary"]["different_history_comparisons"]) == 15
    for phase in [p for p in result["phases"] if p["kind"] == "mixed"]:
        p = result["requests"][phase["indices"][-1]]
        assert (
            max(phase["all_D_first_tokens_monotonic_s"])
            <= phase["gate_monotonic_s"]
            <= p["started_monotonic_s"]
        )
        assert all(
            value == 1
            for value in phase["D_token_events_before_P_first_token"].values()
        )
        assert phase["server_batch_or_device_overlap_proven"] is False
    for row in result["requests"]:
        raw = base64.b64decode(row["raw_sse_base64"], validate=True)
        assert hashlib.sha256(raw).hexdigest() == row["raw_sse_sha256"]
        assert len(raw) == row["raw_sse_bytes"]
        assert len(row["token_ids"]) == row["payload"]["max_tokens"]
        assert "\u2029" in row["text"]
        assert row["done_count"] == row["finish_count"] == row["usage_count"] == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert sum(event["kind"] == "request_started" for event in events) == 19
    assert sum(event["kind"] == "mixed_gate_passed" for event in events) == 2
    assert events[-1]["kind"] == "probe_finished" and events[-1]["complete"]


@pytest.mark.parametrize("kind", ["tokens", "prompt", "usage", "done"])
def test_bad_receipt_keeps_raw_and_stops_remaining_scope(tmp_path, kind):
    server = Server(broken=0, break_kind=kind)
    result = asyncio.run(collect(tmp_path, server))
    assert not result["complete"] and server.seen == [0]
    assert result["summary"]["not_dispatched"] == list(range(1, 19))
    assert result["requests"][0]["raw_sse_bytes"] > 0
    assert "error" in result["requests"][0]


def test_decode_failure_prevents_long_dispatch_and_joins_peers(tmp_path):
    server = Server(broken=6)
    result = asyncio.run(collect(tmp_path, server))
    assert not result["complete"] and 10 not in server.seen
    assert set(server.seen) == set(range(10))
    assert all("finished_monotonic_s" in row for row in result["requests"])
    assert any("CancelledError" in row.get("error", "") for row in result["requests"])


def test_gate_timeout_fails_before_long_prefill(tmp_path):
    server = Server(blocked=True)
    result = asyncio.run(collect(tmp_path, server, gate_timeout=0.02))
    assert not result["complete"] and "timed out" in result["error"]
    assert 10 not in server.seen


def test_already_completed_decode_is_not_contention(tmp_path):
    server = Server(finish_decode_early=True)
    result = asyncio.run(collect(tmp_path, server))
    assert not result["complete"] and "before" in result["error"]
    assert 10 not in server.seen


def test_external_cancellation_saves_failure_and_closes_streams(tmp_path):
    result = asyncio.run(collect(tmp_path, Server(blocked=True), cancel=True))
    assert not result["complete"] and "CancelledError" in result["error"]
    assert len(result["requests"]) == 10


def test_comparison_common_plan_and_missing_or_different_tokens(tmp_path):
    baseline = asyncio.run(collect(tmp_path, Server()))
    candidate = copy.deepcopy(baseline)
    candidate["requests"][0]["token_ids"][4] = 55
    comparison = probe.compare_reports(baseline, candidate)
    assert comparison["complete_evidence"] and comparison["exact_sequences"] == 18
    assert comparison["requests"][0]["first_difference"]["token_index"] == 4
    assert not comparison["numerical_equivalence_proven"]
    candidate["requests"].pop()
    with pytest.raises(ValueError, match="19 successful"):
        probe.compare_reports(baseline, candidate)
    candidate["complete"] = False
    assert not probe.compare_reports(baseline, candidate)["complete_evidence"]
    candidate["plan"]["model"] = "different"
    with pytest.raises(ValueError, match="plans/bodies differ"):
        probe.compare_reports(baseline, candidate)


def test_plan_only_default_no_overwrite_and_execute_file_hash_gate(tmp_path):
    path = tmp_path / "plan.json"
    assert probe.main(["--output", str(path)]) == 0
    plan = json.loads(path.read_text())
    probe.validate_plan(plan)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="never overwrite"):
        probe.main(["--output", str(path)])
    assert path.read_bytes() == original
    output = tmp_path / "execution.json"
    with pytest.raises(ValueError, match="Plan file SHA"):
        probe.main(
            [
                "--execute",
                "--plan",
                str(path),
                "--plan-sha256",
                "0" * 64,
                "--base-url",
                "http://must-not-be-accessed.invalid",
                "--label",
                "cpu",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_execute_requires_explicit_plan_hash_url_and_label(tmp_path):
    with pytest.raises(SystemExit):
        probe.parse_args(["--execute", "--output", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit):
        probe.parse_args(["--output", str(tmp_path / "out.json"), "--timeout", "nan"])
