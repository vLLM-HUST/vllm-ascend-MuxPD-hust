"""Measure ongoing decode streams while full long prompts arrive.

Example: --base-url http://localhost:8000 --output contention.json
Use --prefill-requests 0 for the matched decode-only control, or
--prefill-requests 3 --prefill-input-tokens 12000 for repeated arrivals.
Client SSE arrival gaps are not device execution times. Multi-token chunks
remain indivisible observations; this script never fabricates token times.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

DECODE_SLOS_S = (0.1, 0.2)
PREFILL_SLOS_S = (1.0, 2.0)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def percentile(values, percent):
    """Linear interpolation, including endpoints; None for no observations."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "max": max(values) if values else None,
        **{f"p{p}": percentile(values, p) for p in (50, 90, 95, 99)},
    }


def request_plan(args):
    """Identical synthetic prompts across services/repeats, no user data."""
    rng = random.Random(args.seed)
    plan = []
    for phase, count, inputs, outputs in (
        ("decode", args.decode_requests, args.decode_input_tokens,
         args.decode_output_tokens),
        ("prefill", args.prefill_requests, args.prefill_input_tokens,
         args.prefill_output_tokens),
    ):
        for index in range(count):
            # Ordinary token IDs in the supported Qwen vocabulary. Retain all
            # IDs and their hash; output semantics are not a correctness oracle.
            prompt = [rng.randrange(100, 1000) for _ in range(inputs)]
            payload = {
                "model": args.model, "prompt": prompt, "max_tokens": outputs,
                "temperature": 0, "seed": args.seed, "ignore_eos": True,
                "add_special_tokens": False, "skip_special_tokens": False,
                "stream": True, "stream_options": {"include_usage": True},
                "return_token_ids": True,
            }
            plan.append({
                "phase": phase, "index": index, "payload": payload,
                "prompt_sha256": digest(prompt), "payload_sha256": digest(payload),
            })
    return plan


class SSEReceipt:
    """Retain decoded-HTTP bytes and validate LF/CRLF-delimited SSE frames.

    Frame assembly supports the server's LF/CRLF subset, not bare-CR or BOM
    streams. Within a frame only ASCII CR/LF separate fields; Unicode line
    separators in JSON strings are payload, not SSE line boundaries.
    """

    def __init__(self, row, first_token):
        self.row = row
        self.first_token = first_token
        self.raw = bytearray()
        self.buffer = bytearray()
        self.response_id = None
        self.finish_count = 0
        self.done_count = 0
        self.prompt_seen = False
        self.usage_count = 0
        row.update(events=[], byte_chunks=[], token_ids=[], text="")

    def feed(self, chunk, received):
        self.raw.extend(chunk)
        self.buffer.extend(chunk)
        self.row["byte_chunks"].append({
            "received_monotonic_s": received, "length": len(chunk),
        })
        while True:
            boundaries = [
                (self.buffer.find(mark), len(mark))
                for mark in (b"\n\n", b"\r\n\r\n")
                if self.buffer.find(mark) >= 0
            ]
            if not boundaries:
                return
            end, delimiter = min(boundaries)
            frame = bytes(self.buffer[:end]).decode("utf-8", errors="strict")
            del self.buffer[:end + delimiter]
            self._event(frame, received)

    def _event(self, frame, received):
        data = []
        for line in frame.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if field == "data":
                data.append(value[1:] if value.startswith(" ") else value)
        if not data:
            return
        raw_data = "\n".join(data)
        event = {
            "received_monotonic_s": received,
            "elapsed_s": received - self.row["started_monotonic_s"],
            "raw_data": raw_data,
        }
        self.row["events"].append(event)
        if self.done_count:
            raise ValueError("Data arrived after [DONE]")
        if raw_data == "[DONE]":
            self.done_count += 1
            event["kind"] = "done"
            return
        value = json.loads(raw_data)
        event["json"] = value
        if not isinstance(value, dict) or "error" in value:
            raise ValueError("SSE returned an error or non-object JSON")
        response_id = value.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("Missing response id")
        if self.response_id is not None and response_id != self.response_id:
            raise ValueError("Response id changed during stream")
        self.response_id = response_id
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("Expected zero or one streaming choices")
        usage = value.get("usage")
        if usage is not None:
            if choices or self.finish_count != 1 or self.usage_count:
                raise ValueError("Expected one final usage-only chunk after finish")
            self.usage_count += 1
            self.row["usage"] = usage
        if not choices:
            if usage is None:
                raise ValueError("Empty SSE chunk has no usage")
            event["kind"] = "usage"
            return
        if self.finish_count or self.usage_count:
            raise ValueError("Completion choice arrived after finish")
        choice = choices[0]
        if not isinstance(choice, dict) or type(choice.get("index")) is not int:
            raise ValueError("Invalid completion choice index")
        if choice["index"] != 0:
            raise ValueError("Expected completion choice index zero")
        tokens = choice.get("token_ids")
        if not isinstance(tokens, list) or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise ValueError("Missing or invalid generated token_ids")
        prompt = choice.get("prompt_token_ids")
        if prompt is not None:
            if prompt != self.row["payload"]["prompt"] or any(
                type(token) is not int for token in prompt
            ):
                raise ValueError("Returned prompt_token_ids differ from request")
            self.prompt_seen = True
        if not self.prompt_seen:
            raise ValueError("First completion chunk lacks prompt_token_ids")
        if not isinstance(choice.get("text"), str):
            raise ValueError("Missing completion text")
        self.row["text"] += choice["text"]
        self.row["token_ids"].extend(tokens)
        if len(self.row["token_ids"]) > self.row["payload"]["max_tokens"]:
            raise ValueError("Too many generated tokens")
        event["token_count"] = len(tokens)
        event["kind"] = (
            "single_token" if len(tokens) == 1
            else "multi_token" if tokens else "empty_choice"
        )
        if tokens and "first_token_monotonic_s" not in self.row:
            self.row["first_token_monotonic_s"] = received
            self.first_token()
        reason = choice.get("finish_reason")
        if reason is not None:
            if reason != "length":
                raise ValueError("ignore_eos request must finish with reason=length")
            self.finish_count += 1
            self.row["finish_reason"] = reason

    def finalize(self):
        if self.buffer.strip():
            raise ValueError("Incomplete trailing SSE frame")
        if self.done_count != 1 or self.finish_count != 1 or self.usage_count != 1:
            raise ValueError("Missing DONE, terminal finish, or final usage")
        payload = self.row["payload"]
        if not self.prompt_seen or len(self.row["token_ids"]) != payload["max_tokens"]:
            raise ValueError("Incomplete prompt or generated token IDs")
        usage = self.row.get("usage")
        expected = {
            "prompt_tokens": len(payload["prompt"]),
            "completion_tokens": payload["max_tokens"],
            "total_tokens": len(payload["prompt"]) + payload["max_tokens"],
        }
        if not isinstance(usage, dict) or any(
            type(usage.get(key)) is not int or usage[key] != value
            for key, value in expected.items()
        ):
            raise ValueError("Final usage differs from requested token counts")
        self.row["response_id"] = self.response_id

    def preserve_raw(self):
        self.row["raw_sse_base64"] = base64.b64encode(self.raw).decode("ascii")
        self.row["raw_sse_sha256"] = hashlib.sha256(self.raw).hexdigest()
        self.row["raw_sse_bytes"] = len(self.raw)


def request_metrics(row):
    events = [event for event in row["events"] if event.get("token_count", 0)]
    times = [event["received_monotonic_s"] for event in events]
    gaps = [b - a for a, b in zip(times, times[1:])]
    singles = sum(event["token_count"] == 1 for event in events)
    multi = len(events) - singles
    token_count = len(row["token_ids"])
    complete = row["transport_passed"]
    measurable = complete and bool(gaps)
    # Even single-token SSE messages can share one network read timestamp.
    # These are client observations, never claimed to be server generation times.
    return {
        "ttft_s": times[0] - row["started_monotonic_s"] if times else None,
        "token_event_gaps_s": gaps,
        "token_event_gap_s": distribution(gaps),
        "single_token_events": singles, "multi_token_events": multi,
        "token_events": len(events), "generated_tokens": token_count,
        "mean_tpot_s": ((times[-1] - times[0]) / (token_count - 1)
                        if complete and len(times) > 1 and token_count > 1 else None),
        "mean_tpot_first_chunk_contains_one_token": (
            bool(events) and events[0]["token_count"] == 1
        ),
        "event_gap_slo": {
            str(int(slo * 1000)): (max(gaps) <= slo if measurable else None)
            for slo in DECODE_SLOS_S
        },
        "single_token_observation_slo": {
            str(int(slo * 1000)): (
                max(gaps) <= slo if measurable and not multi else None
            ) for slo in DECODE_SLOS_S
        },
        "ttft_slo": {
            str(int(slo * 1000)): (
                times[0] - row["started_monotonic_s"] <= slo
                if complete and times else None
            ) for slo in PREFILL_SLOS_S
        },
    }


def gap_scope(rows, start, end):
    """Classify intact adjacent-event intervals; never clip at a scope boundary."""
    groups = {key: [] for key in (
        "fully_contained", "crossing_start", "crossing_end", "crossing_both",
    )}
    for row in rows:
        times = [e["received_monotonic_s"] for e in row["events"]
                 if e.get("token_count", 0)]
        for left, right in zip(times, times[1:]):
            if right < start or left > end:
                continue
            if (right == start and left < start) or (left == end and right > end):
                continue
            if left >= start and right <= end:
                category = "fully_contained"
            elif left < start and right > end:
                category = "crossing_both"
            elif left < start:
                category = "crossing_start"
            else:
                category = "crossing_end"
            groups[category].append({
                "request_id": row["request_id"], "start_monotonic_s": left,
                "end_monotonic_s": right, "gap_s": right - left,
            })
    return {
        "start_monotonic_s": start, "end_monotonic_s": end,
        "intervals": groups,
        "gap_s_by_category": {
            key: distribution([item["gap_s"] for item in intervals])
            for key, intervals in groups.items()
        },
        "all_intersecting_intact_gap_s": distribution([
            item["gap_s"] for intervals in groups.values() for item in intervals
        ]),
    }


async def generate(client, args, row, changed):
    row["started_monotonic_s"] = time.monotonic()
    row["dispatched"] = True
    row["transport_passed"] = False
    receipt = SSEReceipt(row, changed.set)

    async def consume():
        async with client.stream(
            "POST", args.base_url.rstrip("/") + "/v1/completions",
            json=row["payload"], headers={"X-Request-Id": row["request_id"]},
        ) as response:
            row["http_status"] = response.status_code
            row["response_headers"] = dict(response.headers)
            if response.status_code != 200:
                receipt.raw.extend(await response.aread())
                raise ValueError(f"Unexpected HTTP status {response.status_code}")
            if "text/event-stream" not in response.headers.get("content-type", ""):
                receipt.raw.extend(await response.aread())
                raise ValueError("Expected text/event-stream response")
            async for chunk in response.aiter_bytes():
                receipt.feed(chunk, time.monotonic())
        receipt.finalize()

    try:
        await asyncio.wait_for(consume(), timeout=args.timeout)
        row["transport_passed"] = True
    except (Exception, asyncio.CancelledError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        row["finished_monotonic_s"] = time.monotonic()
        row["duration_s"] = row["finished_monotonic_s"] - row["started_monotonic_s"]
        receipt.preserve_raw()
        row["metrics"] = request_metrics(row)
        changed.set()
    return row


def summarize_run(run):
    rows = run["requests"]
    ds = [row for row in rows if row["phase"] == "decode"]
    ps = [row for row in rows if row["phase"] == "prefill"]
    seconds = run["finished_monotonic_s"] - run["started_monotonic_s"]
    result = {
        "duration_s": seconds,
        "decode_completed": sum(row["transport_passed"] for row in ds),
        "prefill_completed": sum(row["transport_passed"] for row in ps),
        "decode_token_event_gap_s": distribution([
            gap for row in ds for gap in row["metrics"]["token_event_gaps_s"]
        ]),
        "decode_request_max_gap_s": distribution([
            row["metrics"]["token_event_gap_s"]["max"] for row in ds
            if row["metrics"]["token_event_gap_s"]["max"] is not None
        ]),
        "decode_request_mean_tpot_s": distribution([
            row["metrics"]["mean_tpot_s"] for row in ds
            if row["metrics"]["mean_tpot_s"] is not None
        ]),
        "prefill_ttft_s": distribution([
            row["metrics"]["ttft_s"] for row in ps
            if row["metrics"]["ttft_s"] is not None
        ]),
        "joint_slo": [],
    }
    valid = run["complete"]
    result["completion_throughput"] = {
        phase: {
            "completed_requests": sum(r["transport_passed"] for r in selected),
            "received_output_tokens": sum(len(r["token_ids"]) for r in selected),
            "output_tokens_per_s": (
                sum(len(r["token_ids"]) for r in selected) / seconds
                if valid else None
            ),
        } for phase, selected in (("decode", ds), ("prefill", ps), ("both", rows))
    }
    result["scoped_decode_gap_diagnostics"] = {
        "note": (
            "Overall per-request SLO includes all observed gaps and is unchanged. "
            "Scoped diagnostics preserve complete adjacent-event intervals, "
            "including boundary crossings; durations are never clipped. P wait "
            "windows are client dispatch to first token and may overlap each "
            "other, so their observations must not be pooled as independent gaps."
        ),
        "post_decode_gate": (
            gap_scope(ds, run["decode_gate_monotonic_s"], run["finished_monotonic_s"])
            if "decode_gate_monotonic_s" in run else None
        ),
        "prefill_wait_windows": [
            {"prefill_request_id": p["request_id"], **gap_scope(
                ds, p["started_monotonic_s"], p["first_token_monotonic_s"],
            )} for p in ps if "first_token_monotonic_s" in p
        ],
    }
    for d_slo in DECODE_SLOS_S:
        for p_slo in PREFILL_SLOS_S:
            good_d = [r for r in ds if r["metrics"]["event_gap_slo"][
                str(int(d_slo * 1000))] is True]
            good_p = [r for r in ps if r["metrics"]["ttft_slo"][
                str(int(p_slo * 1000))] is True]
            passed = valid and len(good_d) == len(ds) and len(good_p) == len(ps)
            result["joint_slo"].append({
                "decode_event_gap_limit_ms": int(d_slo * 1000),
                "prefill_ttft_limit_ms": int(p_slo * 1000),
                "decode_requests_meeting_slo": len(good_d),
                "prefill_requests_meeting_slo": len(good_p),
                "all_requests_meet_slo": passed if valid else None,
                "decode_slo_request_goodput_per_s": (
                    len(good_d) / seconds if valid else None
                ),
                "prefill_slo_request_goodput_per_s": (
                    len(good_p) / seconds if valid else None
                ),
                "slo_request_goodput_per_s": (
                    (len(good_d) + len(good_p)) / seconds if valid else None
                ),
                "joint_window_request_goodput_per_s": (
                    len(rows) / seconds if passed else 0.0 if valid else None
                ),
            })
    return result


async def run_once(client, args, plan, stage, repeat):
    run = {
        "stage": stage, "repeat": repeat, "requests": [],
        "started_monotonic_s": time.monotonic(), "complete": False,
        "contention_required": args.prefill_requests > 0,
    }
    changed = asyncio.Event()
    tasks = []
    decodes = []

    def launch(item, scheduled=None):
        row = {
            **item, "request_id": f"contention-{stage}-{repeat}-{item['phase']}-"
                                     f"{item['index']}",
            "scheduled_monotonic_s": scheduled,
        }
        if item["phase"] == "prefill":
            row["active_decode_ids_at_dispatch"] = [
                d["request_id"] for d in decodes
                if "finished_monotonic_s" not in d
                and len(d.get("token_ids", [])) < d["payload"]["max_tokens"]
            ]
            if not row["active_decode_ids_at_dispatch"]:
                raise ValueError("No ongoing decode at prefill dispatch")
        run["requests"].append(row)
        tasks.append(asyncio.create_task(generate(client, args, row, changed)))
        return row

    try:
        for item in plan:
            if item["phase"] == "decode":
                decodes.append(launch(item))
        while not all("first_token_monotonic_s" in row for row in decodes):
            if any("finished_monotonic_s" in row
                   and "first_token_monotonic_s" not in row for row in decodes):
                raise ValueError("A decode failed before its first token")
            await changed.wait()
            changed.clear()
        run["decode_gate_monotonic_s"] = time.monotonic()
        for item in plan:
            if item["phase"] != "prefill":
                continue
            scheduled = (run["decode_gate_monotonic_s"] + args.prefill_start_delay
                         + item["index"] * args.prefill_interval)
            await asyncio.sleep(max(0, scheduled - time.monotonic()))
            if any(row.get("error") for row in run["requests"]):
                raise ValueError("Request failed before a scheduled arrival")
            launch(item, scheduled)
        await asyncio.gather(*tasks)
        if len(run["requests"]) != len(plan) or not all(
            row["transport_passed"] for row in run["requests"]
        ):
            raise ValueError("Not all planned requests completed successfully")
        for row in run["requests"]:
            if row["phase"] == "prefill":
                row["dispatch_lateness_s"] = (
                    row["started_monotonic_s"] - row["scheduled_monotonic_s"]
                )
                # Require observed decode continuation after P starts, not just
                # a connection awaiting its trailing usage/DONE messages.
                continuation = [
                    {"request_id": d["request_id"],
                     "events_after_prefill_dispatch": sum(
                         e.get("token_count", 0) > 0
                         and e["received_monotonic_s"] > row["started_monotonic_s"]
                         for e in d["events"]),
                     "events_before_prefill_first_token": sum(
                         e.get("token_count", 0) > 0
                         and row["started_monotonic_s"]
                         < e["received_monotonic_s"] < row["first_token_monotonic_s"]
                         for e in d["events"])} for d in decodes
                ]
                row["decode_continuation"] = continuation
                if not any(r["events_after_prefill_dispatch"] for r in continuation):
                    raise ValueError("No decode token event after prefill dispatch")
        run["complete"] = True
    except (Exception, asyncio.CancelledError) as error:
        run["error"] = f"{type(error).__name__}: {error}"
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Cancellation can occur after create_task but before the task's first
        # instruction. Preserve that planned row without inventing a dispatch.
        for row in run["requests"]:
            if "started_monotonic_s" not in row:
                row.update(
                    dispatched=False, started_monotonic_s=None,
                    finished_monotonic_s=time.monotonic(), duration_s=None,
                    transport_passed=False,
                    error="Cancelled before transport dispatch",
                )
                receipt = SSEReceipt(row, lambda: None)
                receipt.preserve_raw()
                row["metrics"] = request_metrics(row)
    finally:
        run["finished_monotonic_s"] = time.monotonic()
        run["summary"] = summarize_run(run)
    return run


async def run_benchmark(args, client, checkpoint):
    plan = request_plan(args)
    report = {
        "schema_version": 1, "parameters": vars(args).copy(),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": digest(plan), "plan": plan,
        "complete": False, "warmup": [], "repeats": [],
        "measurement_contract": {
            "clock": "time.monotonic seconds; same client process only",
            "raw_body": "base64 of exact HTTP-decoded SSE body, before UTF-8 parsing",
            "event_timestamp": "arrival of bytes completing an SSE frame",
            "percentiles": "linear interpolation at (n - 1) * p / 100",
            "warmup": (
                "Full identical contention workload, including all prompt/output "
                "lengths and arrival intervals, completed before measured repeats. "
                "Warmup is retained separately and excluded from measured "
                "aggregate summaries."
            ),
            "decode_slo_ms": [100, 200], "prefill_ttft_slo_ms": [1000, 2000],
            "slo_scope": "max observed token-event gap per decode request",
            "chunk_warning": (
                "No timing is assigned inside multi-token chunks. Even separate "
                "SSE frames can share one network read. Per-token observation SLO "
                "is unknown for multi-token streams. Mean TPOT uses observed "
                "first/last token-event times and generated-token count."
            ),
            "joint_goodput": (
                "Qualified D (gap) + P (TTFT) requests divided by complete timed "
                "window; joint-window goodput additionally requires every request "
                "to meet both applicable phase criteria. Invalid runs get null."
            ),
            "server_prerequisites": (
                "Record matching model/hardware, disable prefix caching, warm "
                "graphs, and retain server configuration separately. Native "
                "chunk budgets and plugin full-prompt budget are explicit "
                "experimental factors, not inferred by this client."
            ),
            "limits": (
                "Finite long-lived decode cohort, not a stationary arrival-rate "
                "capacity test. Synthetic token-count/transport validation, not "
                "an output semantic oracle or device overlap proof."
            ),
        },
    }
    checkpoint(report)
    for stage, repeats in (("warmup", args.warmup_repeats), ("repeats", args.repeats)):
        for repeat in range(repeats):
            run = await run_once(client, args, plan, stage, repeat)
            report[stage].append(run)
            checkpoint(report)
            if not run["complete"]:
                report["error"] = f"{stage} {repeat}: {run['error']}"
                report["finished_utc"] = datetime.now(timezone.utc).isoformat()
                checkpoint(report)
                return report
    report["complete"] = True
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["summary"] = {
        "measured_repeats": len(report["repeats"]),
        "warmup_excluded": True,
        "duration_s": distribution([r["summary"]["duration_s"]
                                    for r in report["repeats"]]),
        "output_tokens_per_s": {
            phase: distribution([
                r["summary"]["completion_throughput"][phase]["output_tokens_per_s"]
                for r in report["repeats"]
            ]) for phase in ("decode", "prefill", "both")
        },
        "decode_request_max_gap_s": distribution([
            row["metrics"]["token_event_gap_s"]["max"]
            for run in report["repeats"] for row in run["requests"]
            if row["phase"] == "decode"
            and row["metrics"]["token_event_gap_s"]["max"] is not None
        ]),
        "decode_request_mean_tpot_s": distribution([
            row["metrics"]["mean_tpot_s"]
            for run in report["repeats"] for row in run["requests"]
            if row["phase"] == "decode"
            and row["metrics"]["mean_tpot_s"] is not None
        ]),
        "prefill_ttft_s": distribution([
            row["metrics"]["ttft_s"] for run in report["repeats"]
            for row in run["requests"] if row["phase"] == "prefill"
        ]),
    }
    checkpoint(report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen3.5-35B-A3B")
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=42)
    for name, default in (
        ("repeats", 3), ("warmup-repeats", 1), ("decode-requests", 4),
        ("decode-input-tokens", 128), ("decode-output-tokens", 256),
        ("prefill-requests", 1), ("prefill-input-tokens", 8192),
        ("prefill-output-tokens", 8),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--prefill-interval", type=float, default=0.8)
    parser.add_argument("--prefill-start-delay", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    for key in (
        "repeats", "warmup_repeats", "decode_requests", "decode_input_tokens",
        "decode_output_tokens", "prefill_input_tokens", "prefill_output_tokens",
    ):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.decode_output_tokens < 2:
        parser.error("--decode-output-tokens must be at least 2 to measure gaps")
    if args.prefill_requests < 0:
        parser.error("--prefill-requests must be nonnegative (0 selects control)")
    for key in ("timeout", "prefill_interval", "prefill_start_delay"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key == "timeout" and value == 0):
            parser.error(f"--{key.replace('_', '-')} has an invalid duration")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(report):
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    async def execute():
        limits = httpx.Limits(max_connections=args.decode_requests
                             + args.prefill_requests + 2)
        async with httpx.AsyncClient(
            timeout=args.timeout, trust_env=False, limits=limits,
        ) as client:
            return await run_benchmark(args, client, checkpoint)

    report = asyncio.run(execute())
    print(json.dumps({
        "complete": report["complete"], "output": str(output),
        "error": report.get("error"), "summary": report.get("summary"),
    }, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
