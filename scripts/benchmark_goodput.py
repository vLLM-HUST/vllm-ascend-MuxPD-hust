"""Open-loop arrivals with immutable plans and auditable client-side goodput.

Create once: --plan-only --output arrivals.json --request-rate .75 --seeds 1,2,3
Replay: --plan arrivals.json --base-url http://localhost:8000 --output run.json
No client concurrency semaphore is used. SSE event gaps are client observations,
not device timings; multi-token events never acquire invented per-token times.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:
    from .benchmark_contention import SSEReceipt
except ImportError:  # Direct execution from the scripts directory.
    from benchmark_contention import SSEReceipt

PLAN_SCHEMA = "muxpd-poisson-plan-v1"
REPORT_SCHEMA = "muxpd-goodput-v1"
GENERATOR_DEFAULTS = {
    "model": "Qwen3.5-35B-A3B",
    "duration": 300.0,
    "warmup": 60.0,
    "request_rate": 0.75,
    "seeds": "1,2,3",
    "long_fraction": 0.2,
    "short_input_tokens": 128,
    "short_output_tokens": 256,
    "long_input_tokens": 12000,
    "long_output_tokens": 8,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def percentile(values, p):
    """Linear interpolation at (n - 1) * p / 100, without rounding inputs."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values):
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "max": max(values) if values else None,
        **{f"p{p}": percentile(values, p) for p in (50, 90, 95, 99)},
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_plan(parameters):
    """Independent RNG streams keep arrival, class, and token draws separate."""
    p = dict(parameters)
    validate_parameters(p)
    runs = []
    for seed in p["seeds"]:
        arrival_rng = random.Random(f"arrival:{seed}")
        class_rng = random.Random(f"class:{seed}")
        prompt_rng = random.Random(f"prompt:{seed}")
        arrival = 0.0
        requests = []
        while True:
            arrival += arrival_rng.expovariate(p["request_rate"])
            if arrival >= p["warmup"] + p["duration"]:
                break
            require(len(requests) < 100000, "Plan exceeds 100000 requests per seed")
            kind = "long" if class_rng.random() < p["long_fraction"] else "short"
            prompt = [
                prompt_rng.randrange(100, 1000)
                for _ in range(p[f"{kind}_input_tokens"])
            ]
            payload = {
                "model": p["model"],
                "prompt": prompt,
                "max_tokens": p[f"{kind}_output_tokens"],
                "temperature": 0,
                "seed": seed,
                "ignore_eos": True,
                "add_special_tokens": False,
                "skip_special_tokens": False,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_token_ids": True,
            }
            requests.append(
                {
                    "id": f"seed{seed}-request{len(requests)}",
                    "class": kind,
                    "phase": "warmup" if arrival < p["warmup"] else "measurement",
                    "arrival_offset_s": arrival,
                    "payload": payload,
                    "payload_sha256": digest(payload),
                    "prompt_sha256": digest(prompt),
                }
            )
        runs.append({"seed": seed, "requests": requests})
    body = {"schema": PLAN_SCHEMA, "parameters": p, "runs": runs}
    plan = {**body, "plan_sha256": digest(body)}
    validate_plan(plan)
    return plan


def validate_parameters(p):
    require(isinstance(p, dict), "Generator parameters must be an object")
    require(set(p) == set(GENERATOR_DEFAULTS), "Unexpected generator parameters")
    for name in ("duration", "request_rate", "warmup", "long_fraction"):
        require(
            type(p[name]) in (int, float) and math.isfinite(p[name]), f"Invalid {name}"
        )
    require(
        p["duration"] > 0 and p["request_rate"] > 0 and p["warmup"] >= 0,
        "Duration/rate must be positive and warmup nonnegative",
    )
    require(0 <= p["long_fraction"] <= 1, "long_fraction must be within [0, 1]")
    for name in ("short_input_tokens", "long_input_tokens"):
        require(type(p[name]) is int and p[name] >= 1, f"Invalid {name}")
    for name in ("short_output_tokens", "long_output_tokens"):
        require(type(p[name]) is int and p[name] >= 2, f"Invalid {name}")
    require(isinstance(p["model"], str) and p["model"], "Invalid model")
    seeds = p["seeds"]
    require(
        isinstance(seeds, list)
        and seeds
        and all(type(s) is int for s in seeds)
        and len(set(seeds)) == len(seeds),
        "Seeds must be unique integers",
    )
    require(
        (p["warmup"] + p["duration"]) * p["request_rate"] <= 50000,
        "Expected plan exceeds 50000 requests per seed",
    )


def validate_plan(plan):
    require(
        isinstance(plan, dict) and plan.get("schema") == PLAN_SCHEMA,
        "Unsupported plan schema",
    )
    require(
        set(plan) == {"schema", "parameters", "runs", "plan_sha256"},
        "Unexpected plan fields",
    )
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    require(plan["plan_sha256"] == digest(body), "Plan SHA differs from payload")
    p = plan["parameters"]
    validate_parameters(p)
    seeds = p["seeds"]
    require(
        isinstance(plan["runs"], list)
        and [r.get("seed") for r in plan["runs"]] == seeds,
        "Run seeds differ from parameters",
    )
    for run in plan["runs"]:
        require(set(run) == {"seed", "requests"}, "Unexpected run fields")
        previous = -1.0
        measured = 0
        require(isinstance(run["requests"], list), "Requests must be a list")
        for index, item in enumerate(run["requests"]):
            require(
                set(item)
                == {
                    "id",
                    "class",
                    "phase",
                    "arrival_offset_s",
                    "payload",
                    "payload_sha256",
                    "prompt_sha256",
                },
                "Unexpected request fields",
            )
            t = item["arrival_offset_s"]
            require(
                type(t) in (int, float)
                and math.isfinite(t)
                and previous < t < p["warmup"] + p["duration"]
                and t >= 0,
                "Arrivals must be finite, increasing and inside the window",
            )
            previous = t
            require(
                item["id"] == f"seed{run['seed']}-request{index}",
                "Request identifier differs from plan order",
            )
            phase = "warmup" if t < p["warmup"] else "measurement"
            require(item["phase"] == phase, "Incorrect warmup classification")
            measured += phase == "measurement"
            kind, payload = item["class"], item["payload"]
            require(kind in ("short", "long"), "Unknown request class")
            require(isinstance(payload, dict), "Payload must be an object")
            expected = {
                "model": p["model"],
                "prompt": payload.get("prompt"),
                "max_tokens": p[f"{kind}_output_tokens"],
                "temperature": 0,
                "seed": run["seed"],
                "ignore_eos": True,
                "add_special_tokens": False,
                "skip_special_tokens": False,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_token_ids": True,
            }
            # JSON canonical equality distinguishes boolean values from integers.
            require(digest(payload) == digest(expected), "Altered request options")
            prompt = payload["prompt"]
            require(
                isinstance(prompt, list)
                and len(prompt) == p[f"{kind}_input_tokens"]
                and all(type(token) is int and 100 <= token < 1000 for token in prompt),
                "Invalid synthetic prompt",
            )
            require(
                item["payload_sha256"] == digest(payload)
                and item["prompt_sha256"] == digest(prompt),
                "Request payload or prompt SHA differs",
            )
        require(measured > 0, "Every seed must contain a measured arrival")
    return plan


class Journal:
    """Append raw transport evidence before parsing; periodically flush to disk."""

    def __init__(self, path, flush_interval=1.0):
        self.path = Path(path)
        self.file = self.path.open("x", buffering=256 * 1024)
        self.flush_interval = flush_interval
        self.last_flush = time.monotonic()

    def append(self, kind, *, flush=False, **fields):
        self.file.write(json.dumps({"kind": kind, **fields}, allow_nan=False) + "\n")
        now = time.monotonic()
        if flush or now - self.last_flush >= self.flush_interval:
            self.file.flush()
            self.last_flush = now

    def close(self):
        self.file.flush()
        self.file.close()


def request_metrics(row):
    events = [e for e in row.get("events", []) if e.get("token_count", 0)]
    times = [e["received_monotonic_s"] for e in events]
    gaps = [right - left for left, right in zip(times, times[1:])]
    complete = row.get("transport_passed", False)
    single = bool(events) and all(e["token_count"] == 1 for e in events)
    return {
        "scheduled_ttft_s": (
            times[0] - row["scheduled_monotonic_s"] if times else None
        ),
        "started_ttft_s": (times[0] - row["started_monotonic_s"] if times else None),
        "send_lag_s": (
            row["started_monotonic_s"] - row["scheduled_monotonic_s"]
            if "started_monotonic_s" in row
            else None
        ),
        "gap_s": distribution(gaps),
        "observed_gaps_s": gaps,
        "single_token_events": sum(e["token_count"] == 1 for e in events),
        "multi_token_events": sum(e["token_count"] > 1 for e in events),
        "single_token_observable": complete and single and bool(gaps),
        "event_gap_observable": complete and bool(gaps),
        "mean_tpot_s": (
            (times[-1] - times[0]) / (len(row["token_ids"]) - 1)
            if complete
            and len(times) > 1
            and events[0]["token_count"] == 1
            and len(row["token_ids"]) > 1
            else None
        ),
    }


def summarize_run(run, parameters, max_send_lag):
    """Independent of contention's role-specific and finite-window goodput."""
    rows = run["requests"]
    measured = [r for r in rows if r["phase"] == "measurement"]
    for row in rows:
        row["metrics"] = request_metrics(row)
    gaps = [gap for r in measured for gap in r["metrics"]["observed_gaps_s"]]
    ttfts = [
        r["metrics"]["scheduled_ttft_s"]
        for r in measured
        if r["metrics"]["scheduled_ttft_s"] is not None
    ]
    lags = [
        r["metrics"]["send_lag_s"]
        for r in rows
        if r["metrics"]["send_lag_s"] is not None
    ]
    duration = parameters["duration"]
    start, end = run["measurement_start_monotonic_s"], run["arrival_end_monotonic_s"]
    completed = sum(r.get("transport_passed", False) for r in measured)
    all_transport = all(r.get("transport_passed", False) for r in rows)
    control_valid = not run.get("controller_error") and not run.get("monitor_error")
    client_valid = len(lags) == len(rows) and max(lags, default=0) <= max_send_lag
    all_single = all(r["metrics"]["single_token_observable"] for r in measured)
    window_tokens = [
        (r, e["token_count"])
        for r in rows
        for e in r.get("events", [])
        if e.get("token_count", 0) and start <= e["received_monotonic_s"] < end
    ]
    result = {
        "arrival_window_s": duration,
        "planned_measured_requests": len(measured),
        "realized_measured_arrival_rate_per_s": len(measured) / duration,
        "measured_completed_requests": completed,
        "measured_incomplete_or_failed_requests": len(measured) - completed,
        "warmup_requests": len(rows) - len(measured),
        "all_transport_passed_including_warmup": all_transport,
        "run_control_valid": control_valid,
        "client_dispatch_valid": client_valid,
        "max_allowed_send_lag_s": max_send_lag,
        "send_lag_s": distribution(lags),
        "pooled_observed_event_gap_s": distribution(gaps),
        "all_measured_streams_single_token_observable": all_single,
        "scheduled_ttft_s": distribution(ttfts),
        "request_max_gap_s": distribution(
            [
                r["metrics"]["gap_s"]["max"]
                for r in measured
                if r["metrics"]["gap_s"]["max"] is not None
            ]
        ),
        "request_mean_tpot_s": distribution(
            [
                r["metrics"]["mean_tpot_s"]
                for r in measured
                if r["metrics"]["mean_tpot_s"] is not None
            ]
        ),
        "fixed_window_output": {
            "received_tokens": sum(n for _, n in window_tokens),
            "received_tokens_per_s": sum(n for _, n in window_tokens) / duration,
            "validated_tokens": sum(
                n for r, n in window_tokens if r.get("transport_passed", False)
            ),
            "validated_tokens_per_s": sum(
                n for r, n in window_tokens if r.get("transport_passed", False)
            )
            / duration,
            "includes_warmup_requests_only_for_tokens_inside_window": True,
        },
        "completed_cohort_output_tokens": sum(
            len(r.get("token_ids", []))
            for r in measured
            if r.get("transport_passed", False)
        ),
        "joint_slos": [],
        "backlog": backlog_summary(run),
        "capacity_claim": (
            "One finite run does not establish sustainable capacity. Inspect "
            "backlog, failures, held-out repeats and longer boundary runs."
        ),
    }
    for gap_limit in (0.1, 0.2):
        for ttft_limit in (1.0, 2.0):

            def good(row, single_token):
                m = row["metrics"]
                observable = (
                    m["single_token_observable"]
                    if single_token
                    else m["event_gap_observable"]
                )
                return (
                    observable
                    and m["scheduled_ttft_s"] <= ttft_limit
                    and m["gap_s"]["max"] <= gap_limit
                )

            good_event = sum(good(r, False) for r in measured)
            good_single = sum(good(r, True) for r in measured)
            pooled = (
                all_transport
                and all_single
                and bool(gaps)
                and percentile(gaps, 99) <= gap_limit
            )
            result["joint_slos"].append(
                {
                    "gap_limit_ms": int(gap_limit * 1000),
                    "ttft_limit_ms": int(ttft_limit * 1000),
                    "event_gap_qualified_requests": good_event,
                    "single_token_qualified_requests": good_single,
                    "single_token_unobservable_requests": sum(
                        not r["metrics"]["single_token_observable"] for r in measured
                    ),
                    "event_gap_request_goodput_per_s": good_event / duration,
                    "single_token_request_goodput_per_s": good_single / duration,
                    "single_token_request_attainment": good_single / len(measured),
                    "paper_like_pooled_gap_p99_passed": pooled,
                    "pooled_gap_and_ttft_p99_passed": (
                        pooled
                        and len(ttfts) == len(measured)
                        and percentile(ttfts, 99) <= ttft_limit
                    ),
                    "eligible_for_capacity_comparison": (
                        run.get("complete") is True
                        and client_valid
                        and control_valid
                        and all_transport
                        and all_single
                        and not run.get("drain_deadline_exceeded", False)
                    ),
                    "per_class_single_token_qualified": {
                        kind: {
                            "qualified": sum(
                                good(r, True) for r in measured if r["class"] == kind
                            ),
                            "planned": sum(r["class"] == kind for r in measured),
                        }
                        for kind in ("short", "long")
                    },
                }
            )
    return result


def backlog_point(run, now):
    arrived = [r for r in run["requests"] if r["scheduled_monotonic_s"] <= now]
    terminal = [r for r in arrived if r.get("finished_monotonic_s", math.inf) <= now]
    started = sum(r.get("started_monotonic_s", math.inf) <= now for r in arrived)
    return {
        "monotonic_s": now,
        "offset_s": now - run["origin_monotonic_s"],
        "arrived": len(arrived),
        "started": started,
        "terminal": len(terminal),
        "completed": sum(r.get("transport_passed", False) for r in terminal),
        "failed": sum(not r.get("transport_passed", False) for r in terminal),
        "outstanding": len(arrived) - len(terminal),
        "client_not_started": len(arrived) - started,
    }


def backlog_summary(run):
    points = [
        p
        for p in run["backlog_samples"]
        if run["measurement_start_monotonic_s"]
        <= p["monotonic_s"]
        <= run["arrival_end_monotonic_s"]
    ]
    xs = [p["offset_s"] for p in points]
    ys = [p["outstanding"] for p in points]
    if len(xs) > 1:
        mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
        variance = sum((x - mean_x) ** 2 for x in xs)
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
            if variance
            else None
        )
    else:
        slope = None
    return {
        "scope": "Client planned arrivals minus terminal requests; not server queue",
        "measurement_samples": len(points),
        "sampled_outstanding": distribution(ys),
        "ols_outstanding_drift_requests_per_s": slope,
        "at_measurement_start": backlog_point(
            run, run["measurement_start_monotonic_s"]
        ),
        "at_arrival_end": backlog_point(run, run["arrival_end_monotonic_s"]),
        "stability_is_not_inferred_from_a_single_slope": True,
    }


async def execute_request(client, args, row, journal):
    row["started_monotonic_s"] = time.monotonic()
    row["status"] = "running"
    row["transport_passed"] = False
    journal.append(
        "request_started",
        request_id=row["request_id"],
        monotonic_s=row["started_monotonic_s"],
        scheduled_monotonic_s=row["scheduled_monotonic_s"],
    )
    receipt = SSEReceipt(row, lambda: None)

    async def consume():
        async with client.stream(
            "POST",
            args.base_url.rstrip("/") + "/v1/completions",
            json=row["payload"],
            headers={"X-Request-Id": row["request_id"]},
        ) as response:
            row["http_status"] = response.status_code
            row["headers_received_monotonic_s"] = time.monotonic()
            # Keep transport metadata only; do not persist authorization/cookies.
            row["response_headers"] = {
                key: response.headers[key]
                for key in ("content-type", "content-encoding", "x-request-id")
                if key in response.headers
            }
            journal.append(
                "response_headers",
                request_id=row["request_id"],
                status=response.status_code,
                headers=row["response_headers"],
            )
            valid_http = (
                response.status_code == 200
                and "text/event-stream" in response.headers.get("content-type", "")
            )
            async for chunk in response.aiter_bytes():
                now = time.monotonic()
                journal.append(
                    "http_body",
                    request_id=row["request_id"],
                    monotonic_s=now,
                    body_base64=base64.b64encode(chunk).decode("ascii"),
                    length=len(chunk),
                )
                if valid_http:
                    receipt.feed(chunk, now)
                else:
                    receipt.raw.extend(chunk)
                    row["byte_chunks"].append(
                        {
                            "received_monotonic_s": now,
                            "length": len(chunk),
                        }
                    )
            require(
                response.status_code == 200,
                f"Unexpected HTTP status {response.status_code}",
            )
            require(valid_http, "Expected text/event-stream content type")
            receipt.finalize()

    try:
        await asyncio.wait_for(consume(), timeout=args.request_timeout)
        row["transport_passed"] = True
        row["status"] = "completed"
    except asyncio.CancelledError:
        row["status"] = "cancelled_at_deadline"
        row["error"] = "Cancelled at global drain deadline or controller interruption"
    except Exception as error:
        row["status"] = "failed"
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        row["finished_monotonic_s"] = time.monotonic()
        receipt.preserve_raw()
        journal.append(
            "request_finished",
            flush=True,
            request_id=row["request_id"],
            status=row["status"],
            monotonic_s=row["finished_monotonic_s"],
            transport_passed=row["transport_passed"],
            raw_sse_sha256=row["raw_sse_sha256"],
            raw_sse_bytes=row["raw_sse_bytes"],
            error=row.get("error"),
        )


async def run_once(client, args, planned, parameters, journal, progress):
    # The small preparation margin is outside both arrival/warmup windows.
    origin = time.monotonic() + 0.05
    start = origin + parameters["warmup"]
    end = start + parameters["duration"]
    run = {
        "seed": planned["seed"],
        "origin_monotonic_s": origin,
        "measurement_start_monotonic_s": start,
        "arrival_end_monotonic_s": end,
        "drain_deadline_monotonic_s": end + args.drain_timeout,
        "drain_deadline_exceeded": False,
        "complete": False,
        "requests": [
            {
                **item,
                "request_id": f"goodput-{item['id']}",
                "status": "planned",
                "scheduled_monotonic_s": origin + item["arrival_offset_s"],
                "transport_passed": False,
            }
            for item in planned["requests"]
        ],
        "backlog_samples": [],
    }
    journal.append(
        "run_started",
        flush=True,
        seed=run["seed"],
        origin_monotonic_s=origin,
        measurement_start_monotonic_s=start,
        arrival_end_monotonic_s=end,
        drain_deadline_monotonic_s=end + args.drain_timeout,
    )
    finished = asyncio.Event()
    tasks = []

    async def monitor():
        while not finished.is_set():
            point = backlog_point(run, time.monotonic())
            run["backlog_samples"].append(point)
            journal.append("backlog", flush=True, seed=run["seed"], **point)
            progress(run, point)
            try:
                await asyncio.wait_for(finished.wait(), args.checkpoint_interval)
            except asyncio.TimeoutError:
                pass

    watcher = asyncio.create_task(monitor())
    try:
        for row in run["requests"]:
            await asyncio.sleep(max(0, row["scheduled_monotonic_s"] - time.monotonic()))
            row["dispatched_monotonic_s"] = time.monotonic()
            tasks.append(
                asyncio.create_task(execute_request(client, args, row, journal))
            )
        # Keep the declared observation window intact even if every request ends.
        await asyncio.sleep(max(0, end - time.monotonic()))
        remaining = max(0, run["drain_deadline_monotonic_s"] - time.monotonic())
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=remaining)
            if pending:
                run["drain_deadline_exceeded"] = True
                for task in pending:
                    task.cancel()
            await asyncio.gather(*tasks)
        run["complete"] = all(r.get("transport_passed", False) for r in run["requests"])
    except (Exception, asyncio.CancelledError) as error:
        run["controller_error"] = f"{type(error).__name__}: {error}"
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        finished.set()
        try:
            await watcher
        except Exception as error:
            run["monitor_error"] = f"{type(error).__name__}: {error}"
            run["complete"] = False
        run["finished_monotonic_s"] = time.monotonic()
        run["backlog_samples"].append(backlog_point(run, run["finished_monotonic_s"]))
        run["summary"] = summarize_run(run, parameters, args.max_send_lag)
        journal.append(
            "run_finished",
            flush=True,
            seed=run["seed"],
            complete=run["complete"],
            summary=run["summary"],
        )
    return run


def contract():
    return {
        "arrival": "Precomputed absolute monotonic offsets; no client semaphore",
        "warmup": "Continuous arrival prefix; not drained before measured arrivals",
        "slo_ttft": "First observed token minus planned arrival; send TTFT also kept",
        "send_lag": (
            "Planned arrival to request coroutine entry, before JSON encoding "
            "and connection setup; not a socket-write timestamp"
        ),
        "event_time": "Arrival of HTTP bytes completing each SSE frame",
        "percentiles": "Linear interpolation at (n - 1) * p / 100",
        "request_slo": "Each measured request must meet BOTH TTFT and max gap",
        "goodput_denominator": (
            "Fixed measured arrival duration; drain never changes it"
        ),
        "throughput": "Tokens observed inside fixed window, including warmup requests",
        "multi_token": "No fabricated per-token times; token-SLO unknown, event only",
        "failures": (
            "Retained; zero qualified numerator; count in all-arrival denominator"
        ),
        "backlog": "Client arrived minus terminal; not a server waiting-queue metric",
        "capacity": (
            "Requires multiple rates/seeds, queue stability and longer validation"
        ),
        "journal": (
            "Raw HTTP-decoded bytes appended before parser, flushed periodically"
        ),
        "prerequisites": (
            "Same hardware/model/graph/cache policy; retain server manifest"
        ),
        "synthetic": (
            "Exact transport/token count validation, not a semantic quality oracle"
        ),
    }


async def run_benchmark(args, plan, client, journal, save_report, save_progress):
    dependency = Path(sys.modules[SSEReceipt.__module__].__file__)
    report = {
        "schema": REPORT_SCHEMA,
        "started_utc": utc_now(),
        "complete": False,
        "label": args.label,
        "parameters": vars(args),
        "plan": plan,
        "plan_sha256": plan["plan_sha256"],
        "measurement_contract": contract(),
        "producer_sha256": file_sha(__file__),
        "sse_parser_sha256": file_sha(dependency),
        "runs": [],
    }
    save_report(report)
    journal.append(
        "benchmark_started",
        flush=True,
        plan_sha256=plan["plan_sha256"],
        producer_sha256=report["producer_sha256"],
        sse_parser_sha256=report["sse_parser_sha256"],
    )
    try:
        health = await client.get(args.base_url.rstrip("/") + "/health", timeout=30)
        health.raise_for_status()
        for planned in plan["runs"]:
            run = await run_once(
                client, args, planned, plan["parameters"], journal, save_progress
            )
            report["runs"].append(run)
            save_report(report)
            if not run["complete"] or not run["summary"]["client_dispatch_valid"]:
                report["stopped_early_reason"] = (
                    "Transport/control failure or excessive client send lag; "
                    "remaining seeds are not started on potentially undrained service"
                )
                break
        report["complete"] = len(report["runs"]) == len(plan["runs"]) and all(
            r["complete"] and r["summary"]["client_dispatch_valid"]
            for r in report["runs"]
        )
    except (Exception, asyncio.CancelledError) as error:
        report["controller_error"] = f"{type(error).__name__}: {error}"
    finally:
        report["finished_utc"] = utc_now()
        journal.append("benchmark_finished", flush=True, complete=report["complete"])
        save_report(report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--label", default="")
    parser.add_argument("--plan", help="Replay an existing immutable plan")
    parser.add_argument("--plan-output", help="Persist a newly generated plan here")
    parser.add_argument("--plan-only", action="store_true")
    for name in GENERATOR_DEFAULTS:
        kind = (
            str
            if name in ("model", "seeds")
            else int
            if name.endswith("_tokens")
            else float
        )
        parser.add_argument("--" + name.replace("_", "-"), type=kind)
    parser.add_argument("--drain-timeout", type=float, default=120)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--checkpoint-interval", type=float, default=1)
    parser.add_argument("--max-send-lag", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.plan and (
        args.plan_only
        or args.plan_output
        or any(getattr(args, name) is not None for name in GENERATOR_DEFAULTS)
    ):
        parser.error("--plan cannot be combined with plan generation arguments")
    if args.plan_only and args.plan_output:
        parser.error("--plan-only writes to --output; do not pass --plan-output")
    if not args.plan_only and not args.base_url:
        parser.error("--base-url is required unless --plan-only")
    for name in (
        "drain_timeout",
        "request_timeout",
        "checkpoint_interval",
        "max_send_lag",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not args.plan:
        for name, value in GENERATOR_DEFAULTS.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
        try:
            args.seeds = [int(s.strip()) for s in args.seeds.split(",")]
        except ValueError:
            parser.error("--seeds must be comma-separated unique integers")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    require(not output.exists(), "Output already exists; choose a new evidence path")
    if args.plan:
        plan = validate_plan(json.loads(Path(args.plan).read_text()))
    else:
        plan = build_plan({name: getattr(args, name) for name in GENERATOR_DEFAULTS})
    if args.plan_only:
        atomic_json(output, plan)
        print(
            json.dumps(
                {
                    "plan": str(output),
                    "plan_sha256": plan["plan_sha256"],
                    "requests": [len(r["requests"]) for r in plan["runs"]],
                }
            )
        )
        return 0
    plan_path = (
        Path(args.plan)
        if args.plan
        else Path(args.plan_output or str(output.with_suffix(".plan.json")))
    )
    journal_path = output.with_suffix(".events.jsonl")
    progress_path = output.with_suffix(".progress.json")
    paths = [output, journal_path, progress_path, plan_path]
    require(
        len({p.resolve() for p in paths}) == len(paths),
        "Plan, report, journal and progress paths must differ",
    )
    for path in (journal_path, progress_path):
        require(not path.exists(), f"Evidence path already exists: {path}")
    if not args.plan:
        require(not plan_path.exists(), "Plan output already exists")
        atomic_json(plan_path, plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = Journal(journal_path)

    async def execute():
        async with httpx.AsyncClient(
            timeout=None,
            trust_env=False,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        ) as client:
            return await run_benchmark(
                args,
                plan,
                client,
                journal,
                lambda value: atomic_json(output, value),
                lambda run, point: atomic_json(
                    progress_path,
                    {
                        "complete": False,
                        "plan_sha256": plan["plan_sha256"],
                        "seed": run["seed"],
                        "backlog": point,
                        "journal": str(journal_path),
                        "updated_utc": utc_now(),
                    },
                ),
            )

    try:
        report = asyncio.run(execute())
    finally:
        journal.close()
    report["journal"] = {
        "path": str(journal_path),
        "sha256": file_sha(journal_path),
        "bytes": journal_path.stat().st_size,
    }
    atomic_json(output, report)
    atomic_json(
        progress_path,
        {
            "complete": report["complete"],
            "finished_utc": report["finished_utc"],
            "report": str(output),
            "journal": report["journal"],
        },
    )
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "output": str(output),
                "summaries": [r["summary"] for r in report["runs"]],
            },
            indent=2,
        )
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"Goodput benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1)
