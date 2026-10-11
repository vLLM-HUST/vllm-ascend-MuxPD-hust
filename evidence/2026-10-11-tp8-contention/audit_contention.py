#!/usr/bin/env python3
"""Independently audit completed contention receipts; never import the producer.

Frames are located in the immutable raw byte body first. The timestamp of a
frame comes from the byte-chunk endpoint containing its last delimiter byte,
using a cumulative-length search instead of the producer's incremental parser.
Only manifest-finalized benchmark files are eligible. Failed/incomplete runs
are listed but never contribute to a successful aggregate.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import gzip
import hashlib
import json
import math
import random
import re
from copy import deepcopy
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PHASES = ("decode", "prefill")
WORKLOADS = ("decode-control", "p8k", "p12k", "p12k-burst")
ROLES = ("native16k", "native2k", "old-work256", "guard")
DEFAULT_ROLE_STAGES = {
    "native16k": ["native16k-initial", "native16k-confirm"],
    "native2k": ["native2k-initial", "native2k-confirm"],
    "old-work256": ["old-work256-initial", "old-work256-confirm-clean"],
    "guard": ["guard-v2-initial", "guard-v2-confirm"],
}
FROZEN_PRODUCER_SHA256 = (
    "6f373ff7383ccc61579d4e9406ee0c9b9523ee8a4ef0d5330d76471e421f6712"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def same(actual, expected, where="value"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{where}: expected mapping")
        require(set(actual) == set(expected), f"{where}: keys differ")
        for key in expected:
            same(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list), f"{where}: expected list")
        require(len(actual) == len(expected), f"{where}: list length differs")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{where}[{index}]")
    elif type(expected) in (int, float):
        require(type(actual) in (int, float), f"{where}: expected numeric")
        require(math.isfinite(actual), f"{where}: nonfinite number")
        require(
            math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
            f"{where}: {actual} != {expected}",
        )
    else:
        require(
            type(actual) is type(expected) and actual == expected,
            f"{where}: {actual!r} != {expected!r}",
        )


def stats(values):
    values = sorted(values)
    n = len(values)
    result = {
        "count": n,
        "mean": sum(values) / n if n else None,
        "max": values[-1] if n else None,
    }
    for p in (50, 90, 95, 99):
        if n:
            index, remainder = divmod((n - 1) * p, 100)
            right = min(index + 1, n - 1)
            result[f"p{p}"] = (
                values[index] * (100 - remainder) + values[right] * remainder
            ) / 100
        else:
            result[f"p{p}"] = None
    return result


def finite_time(value):
    return type(value) in (int, float) and math.isfinite(value)


def replay(row):
    """Validate raw transport and reconstruct all data events and token arrivals."""
    require(row.get("dispatched") is True, "request not dispatched")
    require(
        row.get("transport_passed") is True and not row.get("error"),
        "request transport did not pass",
    )
    require(row["http_status"] == 200, "HTTP status is not 200")
    headers = {k.lower(): v for k, v in row["response_headers"].items()}
    require(
        "text/event-stream" in headers.get("content-type", ""),
        "HTTP content type is not SSE",
    )
    start, finish = row["started_monotonic_s"], row["finished_monotonic_s"]
    require(
        finite_time(start) and finite_time(finish) and finish >= start,
        "invalid request lifetime",
    )
    same(row["duration_s"], finish - start, "request duration")
    payload = row["payload"]
    require(
        all(type(t) is int and t >= 0 for t in payload["prompt"]),
        "invalid requested prompt token IDs",
    )
    same(row["prompt_sha256"], canonical(payload["prompt"]), "prompt hash")
    same(row["payload_sha256"], canonical(payload), "payload hash")
    raw = base64.b64decode(row["raw_sse_base64"], validate=True)
    same(row["raw_sse_sha256"], sha(raw), "raw SSE hash")
    same(row["raw_sse_bytes"], len(raw), "raw SSE byte length")
    endpoints, clocks = [], []
    offset, previous = 0, start
    for chunk in row["byte_chunks"]:
        size, clock = chunk["length"], chunk["received_monotonic_s"]
        require(type(size) is int and size > 0, "invalid byte chunk size")
        require(
            finite_time(clock) and previous <= clock <= finish,
            "byte chunk clock outside monotonic request lifetime",
        )
        offset += size
        endpoints.append(offset)
        clocks.append(clock)
        previous = clock
    same(offset, len(raw), "byte chunks do not cover raw body")
    require(bool(endpoints), "empty raw SSE receipt")
    events, tokens, token_times, counts, text = [], [], [], [], []
    response_id, usage, terminal, done, prompt_seen = None, None, False, False, False
    position = 0
    for delimiter in re.finditer(rb"\r?\n\r?\n", raw):
        frame = raw[position : delimiter.start()].decode("utf-8", errors="strict")
        position = delimiter.end()
        fields = []
        for line in frame.replace("\r\n", "\n").split("\n"):
            if not line or line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if field == "data":
                fields.append(value.removeprefix(" ") if separator else "")
        if not fields:
            continue
        require(not done, "data after DONE")
        value = "\n".join(fields)
        received = clocks[bisect.bisect_left(endpoints, position)]
        event = {
            "received_monotonic_s": received,
            "elapsed_s": received - start,
            "raw_data": value,
        }
        if value == "[DONE]":
            require(terminal and usage is not None, "DONE before finish/usage")
            done = True
            event["kind"] = "done"
        else:
            obj = json.loads(value)
            require(
                isinstance(obj, dict) and "error" not in obj,
                "non-object or error SSE JSON",
            )
            event["json"] = obj
            identifier = obj.get("id")
            require(
                isinstance(identifier, str) and bool(identifier), "missing response ID"
            )
            require(response_id in (None, identifier), "response ID changed")
            response_id = identifier
            require(obj.get("model") == payload["model"], "response model changed")
            choices = obj.get("choices")
            require(
                isinstance(choices, list) and len(choices) in (0, 1), "invalid choices"
            )
            if not choices:
                require(
                    terminal and usage is None and isinstance(obj.get("usage"), dict),
                    "invalid usage ordering or repeated usage",
                )
                usage = obj["usage"]
                event["kind"] = "usage"
            else:
                require(
                    not terminal and usage is None and obj.get("usage") is None,
                    "choice after terminal or mixed usage",
                )
                choice = choices[0]
                require(
                    type(choice.get("index")) is int and choice["index"] == 0,
                    "invalid choice index",
                )
                ids = choice.get("token_ids")
                require(
                    isinstance(ids, list)
                    and all(type(t) is int and t >= 0 for t in ids),
                    "invalid generated token IDs",
                )
                prompt = choice.get("prompt_token_ids")
                if prompt is not None:
                    require(
                        all(type(t) is int for t in prompt), "invalid prompt ID type"
                    )
                    same(prompt, payload["prompt"], "returned prompt IDs")
                    prompt_seen = True
                require(prompt_seen, "first choice missing prompt IDs")
                require(isinstance(choice.get("text"), str), "missing choice text")
                tokens.extend(ids)
                text.append(choice["text"])
                event["token_count"] = len(ids)
                event["kind"] = (
                    "empty_choice"
                    if not ids
                    else "single_token"
                    if len(ids) == 1
                    else "multi_token"
                )
                if ids:
                    token_times.append(received)
                    counts.append(len(ids))
                reason = choice.get("finish_reason")
                if reason is not None:
                    require(reason == "length", "non-length terminal finish")
                    terminal = True
        events.append(event)
    require(not raw[position:].strip(), "incomplete trailing SSE frame")
    require(
        done and terminal and usage is not None and prompt_seen, "incomplete stream"
    )
    require(len(tokens) == payload["max_tokens"], "generated length differs")
    for key, expected in {
        "prompt_tokens": len(payload["prompt"]),
        "completion_tokens": len(tokens),
        "total_tokens": len(payload["prompt"]) + len(tokens),
    }.items():
        require(type(usage.get(key)) is int and usage[key] == expected, "usage differs")
    same(row["events"], events, "replayed events")
    same(row["token_ids"], tokens, "generated IDs")
    same(row["text"], "".join(text), "completion text")
    same(row["response_id"], response_id, "response ID")
    same(row["usage"], usage, "usage")
    same(row["finish_reason"], "length", "finish")
    same(row["first_token_monotonic_s"], token_times[0], "first-token timestamp")
    gaps = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    multi = sum(c > 1 for c in counts)
    metrics = {
        "ttft_s": token_times[0] - start,
        "token_event_gaps_s": gaps,
        "token_event_gap_s": stats(gaps),
        "single_token_events": counts.count(1),
        "multi_token_events": multi,
        "token_events": len(counts),
        "generated_tokens": len(tokens),
        "mean_tpot_s": (
            (token_times[-1] - token_times[0]) / (len(tokens) - 1)
            if len(token_times) > 1 and len(tokens) > 1
            else None
        ),
        "mean_tpot_first_chunk_contains_one_token": counts[0] == 1,
        "event_gap_slo": {
            str(s): max(gaps) <= s / 1000 if gaps else None for s in (100, 200)
        },
        "single_token_observation_slo": {
            str(s): max(gaps) <= s / 1000 if gaps and not multi else None
            for s in (100, 200)
        },
        "ttft_slo": {str(s): token_times[0] - start <= s / 1000 for s in (1000, 2000)},
    }
    same(row["metrics"], metrics, "request metrics")
    return {
        "request_id": row["request_id"],
        "phase": row["phase"],
        "index": row["index"],
        "prompt_sha256": canonical(payload["prompt"]),
        "payload_sha256": canonical(payload),
        "non_prompt_payload_sha256": canonical(
            {k: v for k, v in payload.items() if k != "prompt"}
        ),
        "_generated_token_ids": tokens,
        "times": token_times,
        "counts": counts,
        "metrics": metrics,
        "raw_sse_sha256": sha(raw),
        "raw_sse_bytes": len(raw),
        "token_ids_sha256": canonical(tokens),
        "byte_chunks": len(clocks),
    }


def scope(rows, start, finish):
    groups = {
        k: []
        for k in ("fully_contained", "crossing_start", "crossing_end", "crossing_both")
    }
    for row in rows:
        for left, right in zip(row["times"], row["times"][1:]):
            if right < start or left > finish:
                continue
            if (right == start and left < start) or (left == finish and right > finish):
                continue
            boundaries = (left < start, right > finish)
            kind = {
                (False, False): "fully_contained",
                (True, False): "crossing_start",
                (False, True): "crossing_end",
                (True, True): "crossing_both",
            }[boundaries]
            groups[kind].append(
                {
                    "request_id": row["request_id"],
                    "start_monotonic_s": left,
                    "end_monotonic_s": right,
                    "gap_s": right - left,
                }
            )
    return {
        "start_monotonic_s": start,
        "end_monotonic_s": finish,
        "intervals": groups,
        "gap_s_by_category": {
            k: stats([v["gap_s"] for v in values]) for k, values in groups.items()
        },
        "all_intersecting_intact_gap_s": stats(
            [v["gap_s"] for vs in groups.values() for v in vs]
        ),
    }


def audit_run(run, plan, parameters):
    require(run.get("complete") is True and not run.get("error"), "incomplete run")
    rows = run["requests"]
    require(len(rows) == len(plan), "request plan length differs")
    start, end = run["started_monotonic_s"], run["finished_monotonic_s"]
    require(finite_time(start) and finite_time(end) and end > start, "invalid run time")
    derived = []
    for row, intended in zip(rows, plan):
        for key in intended:
            same(row[key], intended[key], "request plan")
        require(
            start <= row["started_monotonic_s"] <= row["finished_monotonic_s"] <= end,
            "request outside run time",
        )
        derived.append(replay(row))
    ds = [r for r in derived if r["phase"] == "decode"]
    ps = [r for r in derived if r["phase"] == "prefill"]
    by_id = {r["request_id"]: r for r in derived}
    require(len(by_id) == len(rows), "duplicate request ID")
    originals = {r["request_id"]: r for r in rows}
    gate = run["decode_gate_monotonic_s"]
    require(max(d["times"][0] for d in ds) <= gate <= end, "decode gate invalid")
    require(run["contention_required"] is bool(ps), "contention flag differs")
    contention = []
    for p in ps:
        row = originals[p["request_id"]]
        dispatched = row["started_monotonic_s"]
        planned = (
            gate
            + parameters["prefill_start_delay"]
            + row["index"] * parameters["prefill_interval"]
        )
        same(row["scheduled_monotonic_s"], planned, "P scheduled arrival")
        require(dispatched >= planned, "P dispatched before its schedule")
        same(row["dispatch_lateness_s"], dispatched - planned, "P dispatch lateness")
        claimed = row["active_decode_ids_at_dispatch"]
        require(bool(claimed) and len(set(claimed)) == len(claimed), "missing active D")
        require(set(claimed) <= {d["request_id"] for d in ds}, "unknown active D")
        # The producer snapshots IDs before create_task; transport starts a few
        # loop turns later. Preserve both sets, and require actual continuation.
        active = [
            d["request_id"]
            for d in ds
            if originals[d["request_id"]]["started_monotonic_s"]
            <= dispatched
            < originals[d["request_id"]]["finished_monotonic_s"]
            and sum(n for t, n in zip(d["times"], d["counts"]) if t <= dispatched)
            < originals[d["request_id"]]["payload"]["max_tokens"]
        ]
        require(bool(active), "no actual active D at P transport start")
        continued = [
            {
                "request_id": d["request_id"],
                "events_after_prefill_dispatch": sum(
                    t > dispatched for t in d["times"]
                ),
                "events_before_prefill_first_token": sum(
                    dispatched < t < p["times"][0] for t in d["times"]
                ),
            }
            for d in ds
        ]
        same(row["decode_continuation"], continued, "D continuation")
        require(
            any(v["events_after_prefill_dispatch"] for v in continued),
            "no D tokens continued after P transport dispatch",
        )
        contention.append(
            {
                "prefill_request_id": p["request_id"],
                "reported_launch_active_ids": claimed,
                "observed_transport_start_active_ids": active,
                "decode_continuation": continued,
            }
        )
    seconds = end - start
    phase_rows = {"decode": ds, "prefill": ps, "both": derived}
    throughput = {
        phase: {
            "completed_requests": len(selected),
            "received_output_tokens": sum(
                r["metrics"]["generated_tokens"] for r in selected
            ),
            "output_tokens_per_s": sum(
                r["metrics"]["generated_tokens"] for r in selected
            )
            / seconds,
        }
        for phase, selected in phase_rows.items()
    }
    joint = []
    for d_ms in (100, 200):
        for p_ms in (1000, 2000):
            good_d = sum(r["metrics"]["event_gap_slo"][str(d_ms)] is True for r in ds)
            good_p = sum(r["metrics"]["ttft_slo"][str(p_ms)] is True for r in ps)
            passed = good_d == len(ds) and good_p == len(ps)
            joint.append(
                {
                    "decode_event_gap_limit_ms": d_ms,
                    "prefill_ttft_limit_ms": p_ms,
                    "decode_requests_meeting_slo": good_d,
                    "prefill_requests_meeting_slo": good_p,
                    "all_requests_meet_slo": passed,
                    "decode_slo_request_goodput_per_s": good_d / seconds,
                    "prefill_slo_request_goodput_per_s": good_p / seconds,
                    "slo_request_goodput_per_s": (good_d + good_p) / seconds,
                    "joint_window_request_goodput_per_s": len(rows) / seconds
                    if passed
                    else 0.0,
                }
            )
    summary = {
        "duration_s": seconds,
        "decode_completed": len(ds),
        "prefill_completed": len(ps),
        "decode_token_event_gap_s": stats(
            [g for d in ds for g in d["metrics"]["token_event_gaps_s"]]
        ),
        "decode_request_max_gap_s": stats(
            [
                d["metrics"]["token_event_gap_s"]["max"]
                for d in ds
                if d["metrics"]["token_event_gap_s"]["max"] is not None
            ]
        ),
        "decode_request_mean_tpot_s": stats(
            [
                d["metrics"]["mean_tpot_s"]
                for d in ds
                if d["metrics"]["mean_tpot_s"] is not None
            ]
        ),
        "prefill_ttft_s": stats([p["metrics"]["ttft_s"] for p in ps]),
        "joint_slo": joint,
        "completion_throughput": throughput,
    }
    for key, value in summary.items():
        same(run["summary"][key], value, f"run summary.{key}")
    require(
        set(run["summary"]) == set(summary) | {"scoped_decode_gap_diagnostics"},
        "unknown unvalidated run summary fields",
    )
    scoped = run["summary"]["scoped_decode_gap_diagnostics"]
    after_gate = scope(ds, gate, end)
    same(scoped["post_decode_gate"], after_gate, "post-gate intact intervals")
    windows = [
        {
            "prefill_request_id": p["request_id"],
            **scope(
                ds, originals[p["request_id"]]["started_monotonic_s"], p["times"][0]
            ),
        }
        for p in ps
    ]
    same(scoped["prefill_wait_windows"], windows, "P wait intact intervals")
    compact = []
    for item in derived:
        copy = {
            k: v
            for k, v in item.items()
            if k not in ("times", "counts", "metrics", "_generated_token_ids")
        }
        copy["metrics"] = {
            k: v for k, v in item["metrics"].items() if k != "token_event_gaps_s"
        }
        compact.append(copy)
    return {
        "stage": run["stage"],
        "repeat": run["repeat"],
        "passed": True,
        "summary": summary,
        "requests": compact,
        "_token_observations": [
            {k: v for k, v in item.items() if k not in ("times", "counts", "metrics")}
            for item in derived
        ],
        "contention": contention,
        "post_gate_intact_gap_s": after_gate["all_intersecting_intact_gap_s"],
        "post_gate_by_category": after_gate["gap_s_by_category"],
        "post_gate_fully_contained_gap_s": after_gate["gap_s_by_category"][
            "fully_contained"
        ],
        "prefill_wait_diagnostics": [
            {
                "prefill_request_id": w["prefill_request_id"],
                "by_category": w["gap_s_by_category"],
                "all_intact": w["all_intersecting_intact_gap_s"],
            }
            for w in windows
        ],
    }


def audit_report(report, source_path, manifest, run_receipt, producer_sources):
    require(report.get("schema_version") == 1, "unknown report schema")
    require(
        report.get("complete") is True and not report.get("error"), "incomplete report"
    )
    producer = report["producer_sha256"]
    require(producer in producer_sources, "producer source bytes unavailable")
    same(run_receipt["producer_sha256"], producer, "manifest producer hash")
    same(manifest["benchmark_sha256"], producer, "stage producer hash")
    require(manifest["parameters"].get("profile") is False, "profile mixed into timing")
    parameters = report["parameters"]
    argv = run_receipt["argv"]
    same(parameters["output"], argv[argv.index("--output") + 1], "recorded output path")
    same(report["plan_sha256"], canonical(report["plan"]), "plan hash")
    rng, expected = random.Random(parameters["seed"]), []
    for phase in PHASES:
        for index in range(parameters[f"{phase}_requests"]):
            prompt = [
                rng.randrange(100, 1000)
                for _ in range(parameters[f"{phase}_input_tokens"])
            ]
            payload = {
                "model": parameters["model"],
                "prompt": prompt,
                "max_tokens": parameters[f"{phase}_output_tokens"],
                "temperature": 0,
                "seed": parameters["seed"],
                "ignore_eos": True,
                "add_special_tokens": False,
                "skip_special_tokens": False,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_token_ids": True,
            }
            expected.append(
                {
                    "phase": phase,
                    "index": index,
                    "payload": payload,
                    "prompt_sha256": canonical(prompt),
                    "payload_sha256": canonical(payload),
                }
            )
    same(report["plan"], expected, "seeded workload plan")
    validated, previous_end = {}, None
    for phase, count in (
        ("warmup", parameters["warmup_repeats"]),
        ("repeats", parameters["repeats"]),
    ):
        require(len(report[phase]) == count, "missing warmup or measured repeats")
        validated[phase] = []
        for index, run in enumerate(report[phase]):
            require(
                run["stage"] == phase and run["repeat"] == index,
                "repeat sequence differs",
            )
            if previous_end is not None:
                require(
                    previous_end <= run["started_monotonic_s"], "repeat windows overlap"
                )
            previous_end = run["finished_monotonic_s"]
            validated[phase].append(audit_run(run, expected, parameters))
    runs = validated["repeats"]
    requests = [r for run in runs for r in run["requests"]]
    ds = [r for r in requests if r["phase"] == "decode"]
    ps = [r for r in requests if r["phase"] == "prefill"]
    aggregate = {
        "measured_repeats": len(runs),
        "warmup_excluded": True,
        "duration_s": stats([r["summary"]["duration_s"] for r in runs]),
        "output_tokens_per_s": {
            phase: stats(
                [
                    r["summary"]["completion_throughput"][phase]["output_tokens_per_s"]
                    for r in runs
                ]
            )
            for phase in (*PHASES, "both")
        },
        "decode_request_max_gap_s": stats(
            [
                r["metrics"]["token_event_gap_s"]["max"]
                for r in ds
                if r["metrics"]["token_event_gap_s"]["max"] is not None
            ]
        ),
        "decode_request_mean_tpot_s": stats(
            [
                r["metrics"]["mean_tpot_s"]
                for r in ds
                if r["metrics"]["mean_tpot_s"] is not None
            ]
        ),
        "prefill_ttft_s": stats([r["metrics"]["ttft_s"] for r in ps]),
    }
    same(report["summary"], aggregate, "measured aggregate")
    seconds = sum(r["summary"]["duration_s"] for r in runs)
    token_observations = [
        {**item, "run_group": run_group, "repeat": run["repeat"]}
        for run_group, phase_runs in validated.items()
        for run in phase_runs
        for item in run.pop("_token_observations")
    ]
    return {
        "passed": True,
        "label": parameters["label"],
        "plan_sha256": report["plan_sha256"],
        "producer_sha256": producer,
        "producer_sources": producer_sources[producer],
        "parameters": parameters,
        "warmup_excluded": True,
        "warmup_audited": len(validated["warmup"]),
        "warmup_request_count": sum(len(r["requests"]) for r in validated["warmup"]),
        "summary": aggregate,
        "repeats": runs,
        "_token_observations": token_observations,
        "pooled_output_tokens_per_s": {
            phase: sum(
                r["summary"]["completion_throughput"][phase]["received_output_tokens"]
                for r in runs
            )
            / seconds
            for phase in (*PHASES, "both")
        },
        "measured_single_token_events": sum(
            r["metrics"]["single_token_events"] for r in requests
        ),
        "measured_multi_token_events": sum(
            r["metrics"]["multi_token_events"] for r in requests
        ),
        "decode_request_slo_pass_counts": {
            str(s): sum(r["metrics"]["event_gap_slo"][str(s)] is True for r in ds)
            for s in (100, 200)
        },
        "prefill_request_slo_pass_counts": {
            str(s): sum(r["metrics"]["ttft_slo"][str(s)] is True for r in ps)
            for s in (1000, 2000)
        },
    }


def excluded_stage(stage, parameters=None):
    if parameters and parameters.get("profile") is True:
        return "diagnostic profile; excluded from timing"
    if re.search(r"(^|-)(profile|diagnostic|aborted)(-|$)", stage):
        return "diagnostic or aborted startup stage; excluded from timing"
    return None


def role_selection(overrides):
    selected = deepcopy(DEFAULT_ROLE_STAGES)
    seen = set()
    for item in overrides:
        role, separator, values = item.partition("=")
        require(
            separator and role in ROLES and role not in seen,
            "--role-stages requires one unique known ROLE=initial,confirm",
        )
        stages = values.split(",")
        require(
            len(stages) == 2 and len(set(stages)) == 2 and all(stages),
            "Each final role must explicitly select two distinct stages",
        )
        require(
            not any(excluded_stage(s) for s in stages),
            "Diagnostic/aborted stages cannot be selected for final pooling",
        )
        selected[role] = stages
        seen.add(role)
    stages = [s for group in selected.values() for s in group]
    require(len(stages) == len(set(stages)), "A stage cannot belong to two roles")
    return selected


def pool_complete_pair(selected):
    """Pool primitive counts and request samples, never mean stage percentiles."""
    runs = [run for report in selected for run in report["repeats"]]
    requests = [request for run in runs for request in run["requests"]]
    ds = [r for r in requests if r["phase"] == "decode"]
    ps = [r for r in requests if r["phase"] == "prefill"]
    wall = sum(run["summary"]["duration_s"] for run in runs)
    require(wall > 0, "Final pool has no measured wall time")
    totals = {
        phase: sum(r["metrics"]["generated_tokens"] for r in rows)
        for phase, rows in (("decode", ds), ("prefill", ps), ("both", requests))
    }
    stage_throughputs = {}
    for phase in (*PHASES, "both"):
        values = [r["pooled_output_tokens_per_s"][phase] for r in selected]
        repeat_values = [
            r["summary"]["completion_throughput"][phase]["output_tokens_per_s"]
            for r in runs
        ]
        stage_throughputs[phase] = {
            "minimum": min(values),
            "maximum": max(values),
            "stages": [
                {
                    "stage": r["stage"],
                    "pooled_tokens_per_s": r["pooled_output_tokens_per_s"][phase],
                }
                for r in selected
            ],
            "repeat_minimum": min(repeat_values),
            "repeat_maximum": max(repeat_values),
        }
    joint = []
    for d_limit in (100, 200):
        for p_limit in (1000, 2000):
            good_d = sum(
                r["metrics"]["event_gap_slo"][str(d_limit)] is True for r in ds
            )
            good_p = sum(r["metrics"]["ttft_slo"][str(p_limit)] is True for r in ps)
            qualified = []
            for run in runs:
                passing = all(
                    (
                        r["metrics"]["event_gap_slo"][str(d_limit)] is True
                        if r["phase"] == "decode"
                        else r["metrics"]["ttft_slo"][str(p_limit)] is True
                    )
                    for r in run["requests"]
                )
                if passing:
                    qualified.append(run)
            qualified_requests = sum(len(run["requests"]) for run in qualified)
            joint.append(
                {
                    "decode_event_gap_limit_ms": d_limit,
                    "prefill_ttft_limit_ms": p_limit,
                    "windows": len(runs),
                    "joint_windows_meeting_slo": len(qualified),
                    "joint_window_qualified_requests": qualified_requests,
                    "joint_window_request_goodput_per_s": qualified_requests / wall,
                    "decode_requests_meeting_slo": good_d,
                    "prefill_requests_meeting_slo": good_p,
                    "decode_slo_request_goodput_per_s": good_d / wall,
                    "prefill_slo_request_goodput_per_s": good_p / wall,
                    "slo_request_goodput_per_s": (good_d + good_p) / wall,
                }
            )
    return {
        "measured_windows": len(runs),
        "warmup_excluded": True,
        "total_wall_s": wall,
        "received_output_tokens": totals,
        "pooled_output_tokens_per_s": {p: n / wall for p, n in totals.items()},
        "decode_requests": len(ds),
        "prefill_requests": len(ps),
        "decode_request_max_gap_s": stats(
            [
                r["metrics"]["token_event_gap_s"]["max"]
                for r in ds
                if r["metrics"]["token_event_gap_s"]["max"] is not None
            ]
        ),
        "decode_request_mean_tpot_s": stats(
            [
                r["metrics"]["mean_tpot_s"]
                for r in ds
                if r["metrics"]["mean_tpot_s"] is not None
            ]
        ),
        "prefill_ttft_s": stats([r["metrics"]["ttft_s"] for r in ps]),
        "decode_request_slo_pass_counts": {
            str(limit): sum(
                r["metrics"]["event_gap_slo"][str(limit)] is True for r in ds
            )
            for limit in (100, 200)
        },
        "prefill_request_slo_pass_counts": {
            str(limit): sum(r["metrics"]["ttft_slo"][str(limit)] is True for r in ps)
            for limit in (1000, 2000)
        },
        "joint_slo": joint,
        "stage_throughput_ranges": stage_throughputs,
        "multi_token_requests": sum(
            r["metrics"]["multi_token_events"] > 0 for r in requests
        ),
        "scope": "Full-request D gaps; startup and boundary-crossing gaps retained. "
        "All pooled rates divide by the sum of every measured window's wall time.",
    }


def final_aggregates(reports, selection, expected_producer, exclusion_evidence=None):
    chosen_stages = {s for stages in selection.values() for s in stages}
    chosen = [r for r in reports if r["stage"] in chosen_stages]
    result = {
        "status": "pending",
        "role_stages": selection,
        "expected_producer_sha256": expected_producer,
        "expected_report_count": len(chosen_stages) * len(WORKLOADS),
        "audited_selected_report_count": len(chosen),
        "unselected_reports": [
            r["file"] for r in reports if r["stage"] not in chosen_stages
        ],
        "unselected_report_reasons": [
            {
                "file": r["file"],
                "stage": r["stage"],
                "reason": (
                    "Guard v1 retained separately; final guard comparison selects v2."
                    if r["stage"] == "guard-initial"
                    else "CPU regression overlapped serving readiness; entire four-"
                    "workload stage retained and replaced by clean confirmation. "
                    "Exclusion was decided from timestamps before inspecting metrics."
                    if r["stage"] == "old-work256-confirm"
                    else "Not included in the explicit final stage selection."
                ),
                "decision_evidence": (
                    exclusion_evidence if r["stage"] == "old-work256-confirm" else None
                ),
            }
            for r in reports
            if r["stage"] not in chosen_stages
        ],
        "comparison_checks": {},
        "rows": [],
    }
    for workload in WORKLOADS:
        lookup = {}
        errors = []
        for report in chosen:
            expected_name = f"{report['stage']}-{workload}.json"
            if Path(report["recorded_output"]).name != expected_name:
                continue
            if report["stage"] in lookup:
                errors.append(f"Duplicate {report['stage']} / {workload}")
            lookup[report["stage"]] = report
            if not report.get("passed"):
                errors.append(f"Audit failed: {expected_name}")
            elif report["producer_sha256"] != expected_producer:
                errors.append(f"Frozen producer mismatch: {expected_name}")
            elif report["label"] != f"{report['stage']}-{workload}":
                errors.append(f"Label disagrees with selected stage: {expected_name}")
        available = [r for r in lookup.values() if r.get("passed")]
        plans = sorted({r["plan_sha256"] for r in available})
        if len(plans) > 1:
            errors.append("Request-plan SHA differs across selected role/stage reports")
        missing = sorted(chosen_stages - lookup.keys())
        check_status = "failed" if errors else "pending" if missing else "passed"
        result["comparison_checks"][workload] = {
            "status": check_status,
            "missing_stages": missing,
            "errors": errors,
            "plan_sha256_values": plans,
            "producer_sha256_values": sorted({r["producer_sha256"] for r in available}),
        }
        for role, stages in selection.items():
            selected = [lookup[s] for s in stages if s in lookup]
            row = {
                "role": role,
                "workload": workload,
                "selected_stages": stages,
                "status": check_status,
                "missing_stages": [s for s in stages if s not in lookup],
                "comparison_missing_stages": missing,
                "sources": [
                    {"stage": r["stage"], "file": r["file"], "sha256": r["sha256"]}
                    for r in selected
                ],
                "aggregate": None,
            }
            if check_status == "passed":
                # Require every registered repeat in both stages, not just
                # whichever warmup/measurement windows happen to be present.
                if any(r["summary"]["measured_repeats"] != 3 for r in selected):
                    row["status"] = "failed"
                    row["error"] = "Final experiment requires three repeats per stage"
                else:
                    row["aggregate"] = pool_complete_pair(selected)
            result["rows"].append(row)
    statuses = {r["status"] for r in result["rows"]}
    result["status"] = (
        "failed"
        if "failed" in statuses
        else "pending"
        if "pending" in statuses
        else "passed"
    )
    return result


def token_variant(stage):
    if stage.startswith("guard-v2-"):
        return "guard-v2"
    if stage == "guard-initial":
        return "guard-v1"
    for role in ("native16k", "native2k", "old-work256"):
        if stage.startswith(role + "-"):
            return role
    return stage


def token_stability(reports, selection):
    """Compare reconstructed arrays, keeping hashes only in the compact result.

    No output is treated as a correctness oracle. Pairing requires the same
    workload, logical phase/index, complete payload, and warmup/measurement
    group. Each initial/confirmation repeat is paired only to its matching
    native16k stage slot. All-to-all comparisons expose a variable reference.
    """
    chosen = {
        stage: (role, slot)
        for role, stages in selection.items()
        for slot, stage in enumerate(stages)
    }
    samples = []
    report_lookup = {}
    for report in reports:
        observations = report.pop("_token_observations", [])
        if not report.get("passed"):
            continue
        workload = next(w for w in WORKLOADS if report["label"].endswith("-" + w))
        report_lookup[report["stage"], workload] = report
        expected = report["warmup_request_count"] + sum(
            len(run["requests"]) for run in report["repeats"]
        )
        require(len(observations) == expected, "Missing reconstructed token arrays")
        for observation in observations:
            tokens = observation["_generated_token_ids"]
            require(bool(tokens), "Empty generated token sequence")
            same(canonical(tokens), observation["token_ids_sha256"], "sequence hash")
            samples.append(
                {
                    **observation,
                    "stage": report["stage"],
                    "workload": workload,
                    "report_sha256": report["sha256"],
                    "plan_sha256": report["plan_sha256"],
                }
            )

    def identity(sample):
        return (
            sample["workload"],
            sample["run_group"],
            sample["phase"],
            sample["index"],
            sample["prompt_sha256"],
            sample["payload_sha256"],
            sample["non_prompt_payload_sha256"],
        )

    def origin(sample):
        return {
            k: sample[k]
            for k in (
                "stage",
                "workload",
                "run_group",
                "repeat",
                "request_id",
                "report_sha256",
                "raw_sse_sha256",
            )
        }

    references = {}
    for sample in samples:
        if sample["stage"] in selection["native16k"]:
            references.setdefault(identity(sample), []).append(sample)

    def stage_slot(stage):
        if stage in chosen:
            return chosen[stage][1]
        if stage.endswith("-initial"):
            return 0
        if "-confirm" in stage:
            return 1
        return None

    def summarize(groups, stages, scope_name, role=None):
        result = []
        for (workload, run_group), cohort in sorted(groups.items()):
            logical = {}
            for sample in cohort:
                logical.setdefault(identity(sample), []).append(sample)
            details = []
            totals = {
                k: 0
                for k in (
                    "logical_inputs",
                    "full_sequence_stable_inputs",
                    "first_token_stable_inputs",
                    "observations",
                    "observations_with_native_reference",
                    "matches_any_native_sequence",
                    "native_pairwise_comparisons",
                    "native_pairwise_exact",
                    "native_pairwise_first_token_exact",
                    "native_paired_repeat_comparisons",
                    "native_paired_repeat_exact",
                    "native_paired_repeat_first_token_exact",
                    "stage_pair_repeat_comparisons",
                    "stage_pair_repeat_exact",
                    "stage_pair_repeat_first_token_exact",
                )
            }
            for key, observations in sorted(logical.items()):
                variants = {}
                for sample in observations:
                    seq = tuple(sample["_generated_token_ids"])
                    variants.setdefault(seq, []).append(sample)
                native = references.get(key, [])
                native_sequences = {tuple(r["_generated_token_ids"]) for r in native}
                pairs, exact, first_exact, paired, paired_exact, paired_first = (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                any_exact = 0
                mismatch = None
                stage_pairs = [
                    (first, second)
                    for first in observations
                    for second in observations
                    if len(stages) == 2
                    and first["stage"] == stages[0]
                    and second["stage"] == stages[1]
                    and first["repeat"] == second["repeat"]
                ]
                for sample in observations:
                    seq = sample["_generated_token_ids"]
                    any_exact += tuple(seq) in native_sequences
                    for reference in native:
                        ref = reference["_generated_token_ids"]
                        equal = seq == ref
                        pairs += 1
                        exact += equal
                        first_exact += seq[0] == ref[0]
                        slot = stage_slot(sample["stage"])
                        if (
                            slot is None
                            or reference["stage"] != selection["native16k"][slot]
                        ):
                            continue
                        if sample["repeat"] != reference["repeat"]:
                            continue
                        paired += 1
                        paired_exact += equal
                        paired_first += seq[0] == ref[0]
                        if not equal and mismatch is None:
                            index = next(
                                (i for i, (a, b) in enumerate(zip(seq, ref)) if a != b),
                                min(len(seq), len(ref)),
                            )
                            mismatch = {
                                "candidate": origin(sample),
                                "reference": origin(reference),
                                "first_difference_index_zero_based": index,
                                "candidate_token_id": seq[index]
                                if index < len(seq)
                                else None,
                                "reference_token_id": ref[index]
                                if index < len(ref)
                                else None,
                            }
                counts = {
                    "logical_inputs": 1,
                    "full_sequence_stable_inputs": int(len(variants) == 1),
                    "first_token_stable_inputs": int(
                        len({seq[0] for seq in variants}) == 1
                    ),
                    "observations": len(observations),
                    "observations_with_native_reference": len(observations)
                    if native
                    else 0,
                    "matches_any_native_sequence": any_exact,
                    "native_pairwise_comparisons": pairs,
                    "native_pairwise_exact": exact,
                    "native_pairwise_first_token_exact": first_exact,
                    "native_paired_repeat_comparisons": paired,
                    "native_paired_repeat_exact": paired_exact,
                    "native_paired_repeat_first_token_exact": paired_first,
                    "stage_pair_repeat_comparisons": len(stage_pairs),
                    "stage_pair_repeat_exact": sum(
                        a["_generated_token_ids"] == b["_generated_token_ids"]
                        for a, b in stage_pairs
                    ),
                    "stage_pair_repeat_first_token_exact": sum(
                        a["_generated_token_ids"][0] == b["_generated_token_ids"][0]
                        for a, b in stage_pairs
                    ),
                }
                for name, value in counts.items():
                    totals[name] += value
                example = observations[0]
                details.append(
                    {
                        **{
                            k: example[k]
                            for k in (
                                "phase",
                                "index",
                                "prompt_sha256",
                                "payload_sha256",
                                "non_prompt_payload_sha256",
                            )
                        },
                        **counts,
                        "distinct_full_sequences": len(variants),
                        "distinct_first_tokens": len({seq[0] for seq in variants}),
                        "native_reference_observations": len(native),
                        "native_reference_distinct_sequences": len(native_sequences),
                        "sequence_variants": [
                            {
                                "token_ids_sha256": rows[0]["token_ids_sha256"],
                                "generated_tokens": len(seq),
                                "first_token_id": seq[0],
                                "count": len(rows),
                                "origins": [origin(r) for r in rows],
                            }
                            for seq, rows in variants.items()
                        ],
                        "first_paired_mismatch": mismatch,
                    }
                )
            available = sorted({sample["stage"] for sample in cohort})
            missing = [s for s in stages if (s, workload) not in report_lookup]
            result.append(
                {
                    "scope": scope_name,
                    "role": role,
                    "variants": sorted({token_variant(s) for s in available}),
                    "selected_stages": stages,
                    "available_stages": available,
                    "coverage": "pending" if missing else "complete",
                    "missing_stages": missing,
                    "workload": workload,
                    "run_group": run_group,
                    **totals,
                    "inputs": details,
                }
            )
        return result

    per_stage = []
    for stage in sorted({s["stage"] for s in samples}):
        groups = {}
        for sample in samples:
            if sample["stage"] == stage:
                groups.setdefault((sample["workload"], sample["run_group"]), []).append(
                    sample
                )
        per_stage.extend(summarize(groups, [stage], "per_stage", token_variant(stage)))
    pooled = []
    for role, stages in selection.items():
        groups = {}
        for sample in samples:
            if sample["stage"] in stages:
                groups.setdefault((sample["workload"], sample["run_group"]), []).append(
                    sample
                )
        pooled.extend(summarize(groups, stages, "selected_role_stages", role))
    return {
        "reconstructed_observations": len(samples),
        "reference_stages": selection["native16k"],
        "per_stage": per_stage,
        "selected_roles": pooled,
        "meaning": (
            "Descriptive generated-token exactness only, with no pass threshold or "
            "semantic oracle. Raw SSE arrays were independently reconstructed and "
            "compared before retaining compact hashes/counts. Inputs match workload, "
            "phase/index, prompt and complete non-prompt payload. Warmup and measured "
            "repeats are separate. Paired repeat exactness matches initial or "
            "confirmation slots; missing reference slots are not counted as "
            "mismatches. Stage-pair exactness compares initial and confirmation "
            "at matching repeat indices within each role. Pairwise exactness "
            "compares every observation to all "
            "available matched native16k observations. Guard v1/v2 stay separate; "
            "unselected stages remain per-stage only. Native graph output variation "
            "remains an open issue (#44); these counts assign no cause or establish "
            "semantic correctness, even when transport validation passes."
        ),
    }


def markdown(report):
    lines = [
        "# Independent contention receipt audit",
        "",
        f"Final selected initial+confirm comparison: **{report['final']['status']}**. "
        "Passing individual reports does not complete a missing final stage.",
        "",
        "Raw SSE bytes, byte-chunk arrival times, tokens, terminal messages, "
        "per-request metrics, scoped intact gaps, and all measured aggregate "
        "fields were independently recomputed. The benchmark module was not "
        "imported. Warmup was checked and excluded from measured summaries; "
        "all repeats are retained.",
        "",
        "| Report | Audit | D SLO 100/200 ms | P SLO 1/2 s | "
        "D max-gap p95 (ms) | Post-gate max: intact / contained (ms) | "
        "P TTFT p95 (ms) | Pooled output tok/s | Multi-token events |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["reports"]:
        if not item.get("passed"):
            lines.append(f"| {item['file']} | FAIL: {item.get('error')} | | | | | | |")
            continue
        d, p = (
            item["decode_request_slo_pass_counts"],
            item["prefill_request_slo_pass_counts"],
        )
        ds = item["summary"]["decode_request_max_gap_s"]["count"]
        ps = item["summary"]["prefill_ttft_s"]["count"]
        dg, pt = (
            item["summary"]["decode_request_max_gap_s"]["p95"],
            item["summary"]["prefill_ttft_s"]["p95"],
        )
        d_text = f"{dg * 1000:.2f}" if dg is not None else "n/a"
        p_text = f"{pt * 1000:.2f}" if pt is not None else "n/a"
        scoped = []
        for key in ("post_gate_intact_gap_s", "post_gate_fully_contained_gap_s"):
            values = [
                r[key]["max"] for r in item["repeats"] if r[key]["max"] is not None
            ]
            scoped.append(f"{max(values) * 1000:.2f}" if values else "n/a")
        lines.append(
            f"| {item['label']} | PASS | {d['100']}/{ds}; {d['200']}/{ds} | "
            f"{p['1000']}/{ps}; {p['2000']}/{ps} | {d_text} | "
            f"{' / '.join(scoped)} | {p_text} | "
            f"{item['pooled_output_tokens_per_s']['both']:.2f} | "
            f"{item['measured_multi_token_events']} |"
        )
    lines += ["", "## Selected final pools", ""]
    lines += [
        f"- {role}: {', '.join(stages)}"
        for role, stages in report["final"]["role_stages"].items()
    ]
    lines += [
        "",
        "| Role | Workload | Final status | Missing own stages |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["final"]["rows"]:
        missing = ", ".join(row["missing_stages"]) or (
            "waiting for other selected roles" if row["status"] == "pending" else "none"
        )
        lines.append(
            f"| {row['role']} | {row['workload']} | {row['status']} | {missing} |"
        )
    complete = [r for r in report["final"]["rows"] if r["status"] == "passed"]
    if complete:
        lines += [
            "",
            "| Role / workload | Windows | Pooled output tok/s | "
            "Stage tok/s min–max | D request max-gap p95 ms | "
            "D mean TPOT p95 ms | P TTFT p95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in complete:
            a = row["aggregate"]
            values = [
                a[key]["p95"]
                for key in (
                    "decode_request_max_gap_s",
                    "decode_request_mean_tpot_s",
                    "prefill_ttft_s",
                )
            ]
            times = " | ".join(
                f"{v * 1000:.2f}" if v is not None else "n/a" for v in values
            )
            span = a["stage_throughput_ranges"]["both"]
            lines.append(
                f"| {row['role']} / {row['workload']} | {a['measured_windows']} | "
                f"{a['pooled_output_tokens_per_s']['both']:.2f} | "
                f"{span['minimum']:.2f}–{span['maximum']:.2f} | {times} |"
            )
        lines += [
            "",
            "| Role / workload | D SLO 100/200 ms | P SLO 1/2 s | "
            "Joint windows 100 ms + 1 s | Joint-window goodput req/s |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        for row in complete:
            a = row["aggregate"]
            d, p = (
                a["decode_request_slo_pass_counts"],
                a["prefill_request_slo_pass_counts"],
            )
            joint = a["joint_slo"][0]
            lines.append(
                f"| {row['role']} / {row['workload']} | "
                f"{d['100']}/{a['decode_requests']}; "
                f"{d['200']}/{a['decode_requests']} | "
                f"{p['1000']}/{a['prefill_requests']}; "
                f"{p['2000']}/{a['prefill_requests']} | "
                f"{joint['joint_windows_meeting_slo']}/{joint['windows']} | "
                f"{joint['joint_window_request_goodput_per_s']:.3f} |"
            )
    lines += [
        "",
        "Final rates use total generated tokens or qualified requests divided "
        "by the sum of all six measured window durations. Percentiles pool "
        "individual requests, not stage percentiles. JSON retains all four "
        "D/P SLO combinations and phase-specific goodputs. Missing stages "
        "produce no final aggregate; mismatched plans or producers fail validation.",
        "",
        "Each request's overall D SLO includes startup gaps. Post-gate and "
        "P-wait diagnostics are retained separately in JSON; boundary-crossing "
        "gaps keep their full duration and are never clipped. A P's observed "
        "wait windows may overlap, so those gaps are not pooled as independent "
        "observations.",
        "",
        "The active-D list is recorded just before create_task, whereas "
        "transport start is recorded in the coroutine. The audit retains the "
        "reported list and independently reconstructs the active set at actual "
        "transport start, requiring at least one active D and subsequent D "
        "token arrivals.",
        "",
        "Token-event timestamps are client byte-arrival observations. Multiple "
        "SSE frames can share one read timestamp; multi-token events have no "
        "fabricated internal token times. These results are not device-overlap "
        "evidence, semantic answer correctness, or stationary capacity/SLO "
        "guarantees.",
        "",
        f"Excluded/unfinalized files: {len(report['excluded'])}. "
        f"Failures: {sum(not r.get('passed') for r in report['reports'])}. "
        "Input, manifest, producer-source, and audit-script SHA256 values "
        "are in JSON.",
        "",
    ]
    lines += ["## Retained unselected stages", ""]
    reasons = report["final"]["unselected_report_reasons"]
    for stage in sorted({r["stage"] for r in reasons}):
        row = next(r for r in reasons if r["stage"] == stage)
        lines.append(f"- {stage}: {row['reason']}")
    lines += [
        "",
        "## Synthetic generated-token agreement",
        "",
        report["synthetic_outputs"]["meaning"],
        "",
        "A one-observation warmup group is trivially identical, not a repeated "
        "stability measurement. JSON retains observation counts, sequence hashes, "
        "native-reference variability and the first paired mismatch for each input.",
        "",
        "| Stage | Scope | Workload | Full / first-token stable inputs | "
        "Paired native16k full exact | Matches any native16k sequence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["synthetic_outputs"]["per_stage"]:
        if row["role"] == "native16k":
            comparison = "reference"
        else:
            comparison = (
                f"{row['native_paired_repeat_exact']}/"
                f"{row['native_paired_repeat_comparisons']}"
            )
        lines.append(
            f"| {row['available_stages'][0]} | {row['run_group']} | "
            f"{row['workload']} | "
            f"{row['full_sequence_stable_inputs']}/{row['logical_inputs']}; "
            f"{row['first_token_stable_inputs']}/{row['logical_inputs']} | "
            f"{comparison} | {row['matches_any_native_sequence']}/"
            f"{row['observations_with_native_reference']} |"
        )
    lines.append("")
    lines += [
        "| Selected role | Scope | Workload | Coverage | Stable full inputs | "
        "Initial/confirm full exact | Initial/confirm first-token exact |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["synthetic_outputs"]["selected_roles"]:
        denominator = row["stage_pair_repeat_comparisons"]
        lines.append(
            f"| {row['role']} | {row['run_group']} | {row['workload']} | "
            f"{row['coverage']} | {row['full_sequence_stable_inputs']}/"
            f"{row['logical_inputs']} | {row['stage_pair_repeat_exact']}/"
            f"{denominator} | {row['stage_pair_repeat_first_token_exact']}/"
            f"{denominator} |"
        )
    lines.append("")
    return "\n".join(lines)


def fault_checks(row):
    """Confirm independent replay rejects corruptions of a real valid receipt."""
    cases = {
        "raw_hash": (("raw_sse_sha256",), "0" * 64),
        "chunk_length": (
            ("byte_chunks", 0, "length"),
            row["byte_chunks"][0]["length"] + 1,
        ),
        "event_timestamp": (
            ("events", 0, "received_monotonic_s"),
            row["events"][0]["received_monotonic_s"] + 0.01,
        ),
        "generated_id": (("token_ids", 0), row["token_ids"][0] + 1),
        "usage": (("usage", "total_tokens"), row["usage"]["total_tokens"] - 1),
        "metric": (("metrics", "ttft_s"), row["metrics"]["ttft_s"] + 0.01),
        "transport": (("transport_passed",), False),
        "terminal_marker": None,
    }
    results = []
    for name, mutation in cases.items():
        corrupt = deepcopy(row)
        if mutation is None:
            raw = base64.b64decode(corrupt["raw_sse_base64"], validate=True)
            require(b"[DONE]" in raw, "fault fixture lacks DONE")
            raw = raw.replace(b"[DONE]", b"[BOOM]")
            corrupt["raw_sse_base64"] = base64.b64encode(raw).decode()
            corrupt["raw_sse_sha256"] = sha(raw)
        else:
            path, value = mutation
            target = corrupt
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        try:
            replay(corrupt)
        except (ValueError, TypeError, KeyError, IndexError, UnicodeError) as error:
            results.append({"case": name, "rejected": True, "reason": str(error)})
        else:
            raise RuntimeError(f"Auditor did not reject corruption: {name}")
    return results


def aggregation_checks():
    """Small synthetic arithmetic fixtures, never experiment measurements."""
    checked = []

    def check(name, condition):
        require(condition, f"Aggregation self-check failed: {name}")
        checked.append({"case": name, "passed": True})

    def request(phase, gap, ttft):
        return {
            "phase": phase,
            "metrics": {
                "generated_tokens": 10 if phase == "decode" else 2,
                "token_event_gap_s": {"max": gap},
                "mean_tpot_s": gap / 2,
                "ttft_s": ttft,
                "multi_token_events": 0,
                "event_gap_slo": {str(x): gap <= x / 1000 for x in (100, 200)},
                "ttft_slo": {str(x): ttft <= x / 1000 for x in (1000, 2000)},
            },
        }

    def stage(name, index, workload):
        duration = 1 if index == 0 else 3
        gaps = [0.01, 0.02, 0.03] if index == 0 else [0.2, 0.3, 0.4]
        ttfts = [0.1, 0.2, 0.3] if index == 0 else [1.5, 1.8, 2.5]
        rates = {
            p: n / duration for p, n in (("decode", 10), ("prefill", 2), ("both", 12))
        }
        return {
            "stage": name,
            "recorded_output": f"/synthetic/{name}-{workload}.json",
            "file": f"/synthetic/{name}-{workload}.json",
            "sha256": "b" * 64,
            "passed": True,
            "producer_sha256": FROZEN_PRODUCER_SHA256,
            "label": f"{name}-{workload}",
            "plan_sha256": canonical(workload),
            "summary": {"measured_repeats": 3},
            "pooled_output_tokens_per_s": rates,
            "repeats": [
                {
                    "summary": {
                        "duration_s": duration,
                        "completion_throughput": {
                            p: {"output_tokens_per_s": rate}
                            for p, rate in rates.items()
                        },
                    },
                    "requests": [
                        request("decode", gap, 0.1),
                        request("prefill", gap, ttft),
                    ],
                }
                for gap, ttft in zip(gaps, ttfts)
            ],
        }

    selection = role_selection([])
    reports = [
        stage(name, index, workload)
        for stages in selection.values()
        for index, name in enumerate(stages)
        for workload in WORKLOADS
    ]
    complete = final_aggregates(reports, selection, FROZEN_PRODUCER_SHA256)
    check(
        "complete_32_report_comparison",
        complete["status"] == "passed" and len(complete["rows"]) == 16,
    )
    pool = complete["rows"][0]["aggregate"]
    check(
        "six_windows_exclude_warmup",
        pool["measured_windows"] == 6 and pool["warmup_excluded"],
    )
    check(
        "pooled_counts_over_total_wall_not_stage_average",
        pool["received_output_tokens"]["both"] == 72
        and pool["total_wall_s"] == 12
        and pool["pooled_output_tokens_per_s"]["both"] == 6,
    )
    check(
        "pooled_request_quantile",
        math.isclose(pool["decode_request_max_gap_s"]["p95"], 0.375),
    )
    check(
        "request_slo_counts",
        pool["decode_request_slo_pass_counts"] == {"100": 3, "200": 4}
        and pool["prefill_request_slo_pass_counts"] == {"1000": 3, "2000": 5},
    )
    joint = pool["joint_slo"][0]
    check(
        "joint_uses_all_window_denominator",
        joint["joint_windows_meeting_slo"] == 3
        and joint["joint_window_qualified_requests"] == 6
        and joint["joint_window_request_goodput_per_s"] == 0.5,
    )
    check(
        "four_joint_threshold_combinations",
        len(pool["joint_slo"]) == 4
        and pool["joint_slo"][3]["joint_windows_meeting_slo"] == 4,
    )
    span = pool["stage_throughput_ranges"]["both"]
    check(
        "stage_and_repeat_ranges",
        span["minimum"] == span["repeat_minimum"] == 4
        and span["maximum"] == span["repeat_maximum"] == 12,
    )
    missing_stage = selection["guard"][1]
    pending = final_aggregates(
        [r for r in reports if r["stage"] != missing_stage],
        selection,
        FROZEN_PRODUCER_SHA256,
    )
    check(
        "missing_confirmation_pending_without_pools",
        pending["status"] == "pending"
        and all(row["aggregate"] is None for row in pending["rows"]),
    )
    for field, value, name in (
        ("plan_sha256", "0" * 64, "plan_mismatch"),
        ("producer_sha256", "0" * 64, "producer_mismatch"),
        ("passed", False, "failed_receipt"),
    ):
        corrupt = deepcopy(reports)
        corrupt[0][field] = value
        check(
            name,
            final_aggregates(corrupt, selection, FROZEN_PRODUCER_SHA256)["status"]
            == "failed",
        )
    check(
        "duplicate_report",
        final_aggregates(reports + [reports[0]], selection, FROZEN_PRODUCER_SHA256)[
            "status"
        ]
        == "failed",
    )
    for value, name in (
        ("guard=guard-v2-initial,guard-v2-initial", "duplicate_stage"),
        ("guard=guard-diagnostic,guard-v2-confirm", "diagnostic_stage"),
        ("guard=guard-aborted,guard-v2-confirm", "aborted_stage"),
        ("guard=native16k-initial,guard-v2-confirm", "cross_role_stage_alias"),
    ):
        try:
            role_selection([value])
        except ValueError:
            check(name, True)
        else:
            check(name, False)
    return {
        "fixture": "Synthetic arithmetic only; not experiment measurements.",
        "all_passed": True,
        "checks": checked,
    }


def token_comparison_checks():
    """Exercise matching/denominators without calling the benchmark producer."""
    checked = []

    def check(name, condition):
        require(condition, f"Token comparison self-check failed: {name}")
        checked.append({"case": name, "passed": True})

    def report(stage, sequences):
        observations = []
        for group, values in (("warmup", [[77, 88]]), ("repeats", sequences)):
            for repeat, seq in enumerate(values):
                observations.append(
                    {
                        "phase": "decode",
                        "index": 0,
                        "run_group": group,
                        "repeat": repeat,
                        "request_id": f"{stage}-{group}-{repeat}",
                        "prompt_sha256": "a" * 64,
                        "payload_sha256": "b" * 64,
                        "non_prompt_payload_sha256": "c" * 64,
                        "_generated_token_ids": seq,
                        "token_ids_sha256": canonical(seq),
                        "raw_sse_sha256": canonical([stage, group, repeat, seq]),
                    }
                )
        return {
            "stage": stage,
            "label": f"{stage}-p8k",
            "passed": True,
            "sha256": canonical(stage),
            "plan_sha256": canonical("plan"),
            "warmup_request_count": 1,
            "repeats": [{"requests": [{}]} for _ in sequences],
            "_token_observations": observations,
        }

    first = [[1, 2], [1, 3], [1, 2]]
    second = [[1, 2], [1, 2], [9, 2]]
    reports = [
        report("native16k-initial", first),
        report("native16k-confirm", second),
        report("guard-v2-initial", second),
        report("guard-initial", [[4, 5]] * 3),
    ]
    selection = role_selection([])
    result = token_stability(deepcopy(reports), selection)
    candidate = next(
        r
        for r in result["per_stage"]
        if r["available_stages"] == ["guard-v2-initial"] and r["run_group"] == "repeats"
    )
    check("all_arrays_reconstructed", result["reconstructed_observations"] == 16)
    check(
        "paired_repeat_exactness",
        candidate["native_paired_repeat_comparisons"] == 3
        and candidate["native_paired_repeat_exact"] == 1
        and candidate["native_paired_repeat_first_token_exact"] == 2,
    )
    check(
        "variable_native_reference_not_single_oracle",
        candidate["matches_any_native_sequence"] == 3
        and candidate["inputs"][0]["native_reference_distinct_sequences"] == 3,
    )
    warmup = next(
        r
        for r in result["per_stage"]
        if r["available_stages"] == ["guard-v2-initial"] and r["run_group"] == "warmup"
    )
    check(
        "warmup_separate",
        warmup["observations"] == 1 and candidate["observations"] == 3,
    )
    native = next(
        r
        for r in result["selected_roles"]
        if r["role"] == "native16k" and r["run_group"] == "repeats"
    )
    check(
        "native_cross_stage_not_self_comparison",
        native["stage_pair_repeat_comparisons"] == 3
        and native["stage_pair_repeat_exact"] == 1
        and native["stage_pair_repeat_first_token_exact"] == 2
        and native["full_sequence_stable_inputs"] == 0,
    )
    guard = next(
        r
        for r in result["selected_roles"]
        if r["role"] == "guard" and r["run_group"] == "repeats"
    )
    check(
        "v1_v2_separate_and_missing_stage_pending",
        guard["coverage"] == "pending"
        and guard["variants"] == ["guard-v2"]
        and guard["observations"] == 3
        and guard["stage_pair_repeat_comparisons"] == 0,
    )
    altered = deepcopy(reports)
    for observation in altered[2]["_token_observations"]:
        observation["payload_sha256"] = "d" * 64
    changed = token_stability(altered, selection)
    candidate = next(
        r
        for r in changed["per_stage"]
        if r["available_stages"] == ["guard-v2-initial"] and r["run_group"] == "repeats"
    )
    check(
        "different_sampling_payload_not_compared",
        candidate["observations_with_native_reference"] == 0
        and candidate["native_paired_repeat_comparisons"] == 0,
    )
    for fault in ("missing_array", "wrong_hash"):
        altered = deepcopy(reports)
        if fault == "missing_array":
            altered[0]["_token_observations"].pop()
        else:
            altered[0]["_token_observations"][0]["token_ids_sha256"] = "0" * 64
        try:
            token_stability(altered, selection)
        except ValueError:
            check(fault, True)
        else:
            check(fault, False)
    return {
        "fixture": "Synthetic token arrays only; not experiment measurements.",
        "all_passed": True,
        "checks": checked,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=BASE)
    parser.add_argument(
        "--stages", nargs="*", help="Optional stage names; default all manifests"
    )
    parser.add_argument("--producer", action="append", type=Path, default=[])
    parser.add_argument(
        "--role-stages",
        action="append",
        default=[],
        metavar="ROLE=INITIAL,CONFIRM",
        help="Override a known role's two final stages; other roles keep defaults",
    )
    parser.add_argument("--expected-producer-sha256", default=FROZEN_PRODUCER_SHA256)
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="Return exit status 2 while any selected final stage is pending",
    )
    parser.add_argument("--output", type=Path, default=BASE / "contention-audit.json")
    args = parser.parse_args()
    try:
        selection = role_selection(args.role_stages)
        require(
            re.fullmatch(r"[0-9a-f]{64}", args.expected_producer_sha256),
            "Expected producer SHA256 must have 64 lowercase hex characters",
        )
    except ValueError as error:
        parser.error(str(error))
    sources = {}
    for path in [ROOT / "scripts/benchmark_contention.py", *args.producer]:
        if path.is_file():
            sources.setdefault(sha(path.read_bytes()), []).append(str(path.resolve()))
    result = {
        "schema_version": 1,
        "audit_script_sha256": sha(Path(__file__).read_bytes()),
        "independent_of_benchmark_imports": True,
        "reports": [],
        "excluded": [],
    }
    included = set()
    for manifest_path in sorted(args.directory.glob("*-manifest.json")):
        content = manifest_path.read_bytes()
        try:
            manifest = json.loads(content)
        except ValueError:
            result["excluded"].append(
                {
                    "file": str(manifest_path),
                    "reason": "manifest is not a complete JSON snapshot; not audited",
                }
            )
            continue
        reason = excluded_stage(
            manifest_path.name.removesuffix("-manifest.json")
        ) or excluded_stage(manifest.get("stage", ""), manifest.get("parameters"))
        if reason:
            result["excluded"].append(
                {
                    "file": str(manifest_path),
                    "stage": manifest.get("stage"),
                    "manifest_sha256": sha(content),
                    "reason": reason,
                }
            )
            continue
        if args.stages and manifest.get("stage") not in args.stages:
            continue
        for entry in manifest.get("client_runs", []):
            argv = entry.get("argv", [])
            if not any(Path(part).name == "benchmark_contention.py" for part in argv):
                continue
            require("--output" in argv, "manifest benchmark lacks output path")
            recorded_path = Path(argv[argv.index("--output") + 1])
            path = args.directory / recorded_path.name
            if not path.is_file() and path.with_suffix(".json.gz").is_file():
                path = path.with_suffix(".json.gz")
            if not path.is_file() and recorded_path.is_file():
                path = recorded_path
            if not path.is_file():
                result["excluded"].append(
                    {"file": str(path), "reason": "missing report"}
                )
                continue
            if entry.get("returncode") != 0 or not entry.get("output_sha256"):
                result["excluded"].append(
                    {
                        "file": str(path),
                        "reason": "not finalized successfully in manifest",
                    }
                )
                continue
            stored_bytes = path.read_bytes()
            data = (
                gzip.decompress(stored_bytes) if path.suffix == ".gz" else stored_bytes
            )
            item = {
                "file": str(path),
                "sha256": sha(data),
                "stored_file_sha256": sha(stored_bytes),
                "compression": "gzip" if path.suffix == ".gz" else "none",
                "recorded_output": str(recorded_path),
                "manifest": str(manifest_path),
                "manifest_sha256": sha(content),
                "stage": manifest.get("stage"),
                "manifest_role": manifest.get("parameters"),
                "passed": False,
            }
            included.add(recorded_path.name)
            try:
                same(item["sha256"], entry["output_sha256"], "manifest output hash")
                decoded = json.loads(data)
                item.update(audit_report(decoded, path, manifest, entry, sources))
                if "corruption_self_checks" not in result:
                    result["corruption_self_checks"] = fault_checks(
                        decoded["repeats"][0]["requests"][0]
                    )
            except (ValueError, TypeError, KeyError, IndexError, UnicodeError) as error:
                item["passed"] = False
                item["error"] = f"{type(error).__name__}: {error}"
            result["reports"].append(item)
            print(path.name, "PASS" if item["passed"] else item["error"], flush=True)
    for path in sorted(args.directory.glob("*.json*")):
        name = path.name.removesuffix(".gz")
        if name in included or not any(name.endswith(f"-{w}.json") for w in WORKLOADS):
            continue
        if args.stages and not any(path.name.startswith(f"{s}-") for s in args.stages):
            continue
        result["excluded"].append(
            {
                "file": str(path),
                "reason": "no successful finalized manifest receipt; body not audited",
            }
        )
    result["all_audited_reports_passed"] = bool(result["reports"]) and all(
        r["passed"] for r in result["reports"]
    )
    result["plan_comparisons"] = {}
    for workload in WORKLOADS:
        matching = [
            r
            for r in result["reports"]
            if r.get("passed")
            and Path(r["recorded_output"]).name.endswith(f"-{workload}.json")
        ]
        hashes = sorted({r["plan_sha256"] for r in matching})
        result["plan_comparisons"][workload] = {
            "reports": [r["label"] for r in matching],
            "plan_sha256_values": hashes,
            "same_request_plan": bool(hashes) and len(hashes) == 1,
        }
    decision_path = args.directory / "confirmation-cpu-overlap.json"
    exclusion_evidence = None
    if decision_path.is_file():
        decision_bytes = decision_path.read_bytes()
        exclusion_evidence = {
            "file": str(decision_path),
            "sha256": sha(decision_bytes),
            "decision": json.loads(decision_bytes),
        }
    result["final"] = final_aggregates(
        result["reports"], selection, args.expected_producer_sha256, exclusion_evidence
    )
    result["synthetic_outputs"] = token_stability(result["reports"], selection)
    result["aggregation_self_checks"] = aggregation_checks()
    result["token_comparison_self_checks"] = token_comparison_checks()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(markdown(result))
    if (
        not result["all_audited_reports_passed"]
        or result["final"]["status"] == "failed"
    ):
        return 1
    if args.require_final and result["final"]["status"] == "pending":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
