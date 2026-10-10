"""Strict synthetic completion receipts for an opt-in decode-graph candidate.

This client cannot prove graph capture/replay or actual device batch sizes.
Pair the receipts with worker graph snapshots and scheduler/device traces.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:  # Direct CLI and namespace-package imports used by CPU tests.
    from compare_outputs import summarize as summarize_texts
except ModuleNotFoundError:
    from scripts.compare_outputs import summarize as summarize_texts

GENERATION = {
    "temperature": 0,
    "seed": 42,
    "ignore_eos": True,
    "add_special_tokens": False,
    "skip_special_tokens": False,
    "stream": True,
    "stream_options": {"include_usage": True},
}


def cases():
    result = [
        {
            "name": "one_token_prefill",
            "group": "single",
            "prompt": [100],
            "max_tokens": 16,
        }
    ]
    for count in (1, 2, 3, 5, 8):
        for index in range(count):
            serial = len(result)
            result.append(
                {
                    "name": f"burst{count}_{index}",
                    "group": f"burst{count}",
                    "prompt": [
                        100 + (serial * 97 + offset * 31) % 20000
                        for offset in range(16 + index)
                    ],
                    "max_tokens": 8,
                }
            )
    # Intended lengths can reduce live client requests 8 -> 5 -> 3 -> 1.
    # Actual model batching still requires scheduler/worker evidence.
    for index, length in enumerate((8, 8, 8, 10, 10, 12, 12, 16)):
        serial = len(result)
        result.append(
            {
                "name": f"shrinking8_{index}",
                "group": "shrinking8",
                "prompt": [
                    100 + (serial * 97 + offset * 31) % 20000 for offset in range(24)
                ],
                "max_tokens": length,
            }
        )
    return result


class CompletionSSE:
    """Incremental SSE framing; a partial final frame is a transport failure."""

    def __init__(self):
        self.pending = ""
        self.data_lines = []
        self.events = []
        self.text = ""
        self.usage = None
        self.finish_reason = None
        self.response_id = None
        self.done = False
        self.first_choice_s = None
        self.first_text_s = None

    def feed(self, text, received_s):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if self.data_lines:
                    data = "\n".join(self.data_lines)
                    self.data_lines = []
                    self._event(data, received_s)
            elif line.startswith("data:"):
                self.data_lines.append(line[5:].removeprefix(" "))
            elif not (
                line.startswith(":") or line.startswith(("event:", "id:", "retry:"))
            ):
                raise ValueError("Unexpected SSE field or unframed response data")

    def _event(self, data, received_s):
        if self.done:
            raise ValueError("SSE data received after [DONE]")
        if data == "[DONE]":
            self.done = True
            return
        event = json.loads(data)
        if not isinstance(event, dict) or "error" in event:
            raise ValueError(f"SSE error or invalid object: {event}")
        self.events.append({"received_monotonic_s": received_s, "data": event})
        identity = event.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Missing SSE response id")
        if self.response_id is not None and self.response_id != identity:
            raise ValueError("SSE response id changed")
        self.response_id = identity
        choices = event.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("Expected one completion choice or a usage-only event")
        if choices:
            choice = choices[0]
            if self.finish_reason is not None:
                raise ValueError("Completion choice received after finish")
            if not isinstance(choice, dict) or choice.get("index") != 0:
                raise ValueError("Unexpected completion choice index")
            text = choice.get("text")
            if not isinstance(text, str):
                raise ValueError("Completion text must be a string")
            self.first_choice_s = (
                self.first_choice_s if self.first_choice_s is not None else received_s
            )
            if text and self.first_text_s is None:
                self.first_text_s = received_s
            self.text += text
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason != "length":
                    raise ValueError("ignore_eos probe must finish with reason=length")
                self.finish_reason = reason
        usage = event.get("usage")
        if usage is not None:
            if not isinstance(usage, dict) or (
                self.usage is not None and self.usage != usage
            ):
                raise ValueError("Invalid or inconsistent usage")
            self.usage = usage

    def finish(self, case):
        if self.pending or self.data_lines:
            raise ValueError("Partial SSE frame at response EOF")
        if not self.done or self.finish_reason != "length" or self.usage is None:
            raise ValueError("Incomplete SSE response: require DONE, finish and usage")
        if not self.text or self.first_choice_s is None or self.first_text_s is None:
            raise ValueError("Missing generated completion text")
        expected = {
            "prompt_tokens": len(case["prompt"]),
            "completion_tokens": case["max_tokens"],
            "total_tokens": len(case["prompt"]) + case["max_tokens"],
        }
        for name, value in expected.items():
            actual = self.usage.get(name)
            if type(actual) is not int or actual != value:
                raise ValueError(
                    f"Unexpected {name}: expected {value}, received {actual}"
                )

    def receipt(self):
        return {
            "text": self.text,
            "text_sha256": hashlib.sha256(self.text.encode()).hexdigest(),
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "response_id": self.response_id,
            "done_received": self.done,
            "first_choice_monotonic_s": self.first_choice_s,
            "first_text_monotonic_s": self.first_text_s,
            "events": self.events,
        }


async def stream_case(client, args, base_url, case, repeat, gate, group_clock):
    await gate.wait()
    started = time.perf_counter()
    row = {
        "case": case["name"],
        "mode": case["group"],
        "repeat": repeat,
        "started_monotonic_s": started,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dispatch_offset_from_release_s": started - group_clock[0],
        "transport_passed": False,
    }
    body, parser = bytearray(), CompletionSSE()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        async with client.stream(
            "POST",
            base_url.rstrip("/") + "/v1/completions",
            json={
                "model": args.model,
                "prompt": case["prompt"],
                "max_tokens": case["max_tokens"],
                **GENERATION,
            },
        ) as response:
            row["http_status"] = response.status_code
            response.raise_for_status()
            if (
                "text/event-stream"
                not in response.headers.get("content-type", "").lower()
            ):
                raise ValueError("Expected text/event-stream Content-Type")
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                parser.feed(decoder.decode(chunk), time.perf_counter())
            parser.feed(decoder.decode(b"", final=True), time.perf_counter())
        parser.finish(case)
        row["transport_passed"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row.update(parser.receipt())
        row["ended_monotonic_s"] = time.perf_counter()
        row["duration_s"] = row["ended_monotonic_s"] - started
        row["sse_body"] = body.decode("utf-8", errors="replace")
        row["sse_body_sha256"] = hashlib.sha256(body).hexdigest()
    return row


async def run_endpoint(client, args, label, base_url, test_cases):
    endpoint = {"label": label, "base_url": base_url, "requests": [], "groups": []}
    try:
        health = await client.get(base_url.rstrip("/") + "/health")
        health.raise_for_status()
        for repeat in range(args.repeats):
            for group in dict.fromkeys(case["group"] for case in test_cases):
                selected = [case for case in test_cases if case["group"] == group]
                gate, clock = asyncio.Event(), []
                tasks = [
                    asyncio.create_task(
                        stream_case(client, args, base_url, case, repeat, gate, clock)
                    )
                    for case in selected
                ]
                await asyncio.sleep(0)  # Every task waits at the same dispatch gate.
                clock.append(time.perf_counter())
                gate.set()
                rows = await asyncio.gather(*tasks)
                endpoint["requests"].extend(rows)
                endpoint["groups"].append(
                    {
                        "name": group,
                        "repeat": repeat,
                        "released_monotonic_s": clock[0],
                        "requested_concurrency": len(selected),
                        "ended_monotonic_s": time.perf_counter(),
                        "dispatch_spread_s": max(r["started_monotonic_s"] for r in rows)
                        - min(r["started_monotonic_s"] for r in rows),
                    }
                )
    except Exception as exc:
        endpoint["error"] = f"{type(exc).__name__}: {exc}"
    return endpoint


def load_baseline(path, args, test_cases):
    contents = Path(path).read_bytes()
    report = json.loads(contents)
    if report.get("schema_version") != 1 or report.get("complete") is not True:
        raise ValueError("Baseline report must have a completed supported schema")
    if report.get("cases") != test_cases or report.get("generation") != GENERATION:
        raise ValueError("Baseline synthetic inputs/generation parameters differ")
    if any(
        report.get("parameters", {}).get(key) != getattr(args, key)
        for key in ("model", "repeats")
    ):
        raise ValueError("Baseline model or repeat count differs")
    endpoints = [
        row for row in report.get("endpoints", []) if row.get("label") == "baseline"
    ]
    if len(endpoints) != 1 or endpoints[0].get("error"):
        raise ValueError("Expected one successful baseline endpoint")
    endpoint = endpoints[0]
    expected = {
        (repeat, case["group"], case["name"]): case
        for repeat in range(args.repeats)
        for case in test_cases
    }
    seen = set()
    for row in endpoint.get("requests", []):
        key = (row.get("repeat"), row.get("mode"), row.get("case"))
        if key not in expected or key in seen:
            raise ValueError("Duplicate or unexpected baseline receipt")
        seen.add(key)
        if row.get("error") or row.get("transport_passed") is not True:
            raise ValueError("Baseline contains a failed transport")
        raw = row["sse_body"]
        if hashlib.sha256(raw.encode()).hexdigest() != row.get("sse_body_sha256"):
            raise ValueError("Baseline raw SSE hash mismatch")
        replay = CompletionSSE()
        replay.feed(raw, 0)
        replay.finish(expected[key])
        for field in (
            "text",
            "text_sha256",
            "usage",
            "finish_reason",
            "response_id",
            "done_received",
        ):
            if row.get(field) != replay.receipt()[field]:
                raise ValueError(f"Baseline receipt differs from raw SSE: {field}")
        times = [
            row.get(field)
            for field in (
                "started_monotonic_s",
                "first_choice_monotonic_s",
                "first_text_monotonic_s",
                "ended_monotonic_s",
            )
        ]
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in times
        ) or times != sorted(times):
            raise ValueError("Baseline request timestamps are invalid")
    if seen != set(expected):
        raise ValueError("Baseline request coverage is incomplete")
    endpoint["reused_from"] = str(path)
    endpoint["reused_report_sha256"] = hashlib.sha256(contents).hexdigest()
    return endpoint


def summarize(endpoints):
    # Reuse the natural-language checker’s exact-text/repeat indexing. No text
    # oracle is claimed here: its internal oracle field carries transport status.
    normalized = [
        {
            **endpoint,
            "requests": [
                {**row, "oracle_passed": row["transport_passed"]}
                for row in endpoint["requests"]
            ],
        }
        for endpoint in endpoints
    ]
    compared = summarize_texts(normalized)
    return {
        "transport_failures": [
            {
                "endpoint": endpoint["label"],
                "case": row["case"],
                "repeat": row["repeat"],
                "error": row.get("error"),
            }
            for endpoint in endpoints
            for row in endpoint["requests"]
            if not row["transport_passed"]
        ],
        "repeat_stability": {
            label: value["repeat_stability"]
            for label, value in compared["endpoints"].items()
        },
        "cross_endpoint_text_matches": compared["cross_endpoint_text_matches"],
        "device_graph_execution_proven": False,
        "actual_decode_batch_sizes_proven": False,
    }


async def main(args):
    test_cases = cases()
    report = {
        "schema_version": 1,
        "parameters": vars(args),
        "cases": test_cases,
        "generation": GENERATION,
        "endpoints": [],
        "complete": False,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": (
            "Synthetic token IDs only. Requested client concurrency is not actual "
            "D batch size. Exact decoded text is not generated-token identity or "
            "proof of device graph execution. Require worker graph snapshots and "
            "scheduler/device traces; this is not a performance benchmark."
        ),
    }
    targets = [] if args.baseline_only else [("candidate", args.candidate_url)]
    if args.baseline_report:
        report["endpoints"].append(
            load_baseline(args.baseline_report, args, test_cases)
        )
    else:
        targets.insert(0, ("baseline", args.baseline_url))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        for label, base_url in targets:
            report["endpoints"].append(
                await run_endpoint(client, args, label, base_url, test_cases)
            )
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["summary"] = summarize(report["endpoints"])
    report["complete"] = True
    failed = any(
        endpoint.get("error")
        or len(endpoint["requests"]) != args.repeats * len(test_cases)
        or any(not row["transport_passed"] for row in endpoint["requests"])
        for endpoint in report["endpoints"]
    )
    report["transport_passed"] = not failed
    if args.require_exact:
        failed |= any(
            not row["exact_text_match"]
            for row in report["summary"]["cross_endpoint_text_matches"]
        )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "transport_passed": report["transport_passed"],
                "summary": report["summary"],
            },
            indent=2,
        )
    )
    return int(failed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-url")
    baseline.add_argument("--baseline-report")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--candidate-url")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen3.5-35B-A3B")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("repeats and timeout must be finite and positive")
    if args.baseline_only:
        if not args.baseline_url or args.candidate_url or args.require_exact:
            parser.error(
                "--baseline-only requires --baseline-url and no candidate/exact option"
            )
    elif not args.candidate_url:
        parser.error("--candidate-url is required unless --baseline-only")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
