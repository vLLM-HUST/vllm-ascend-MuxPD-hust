"""Offline three-role numeric audit. No producer imports, HTTP, or device access.

Source draft: run only after root opens the CPU window and stops all three
registered stages. Preserve every original report. Audit success means evidence
consistency and complete transport/history, not numerical equivalence.
"""

import argparse
import base64
import hashlib
import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN_SHA = "b92c7dcd36b93fe9d18796436175584ba80e0d39105108d44d13e95e1f3665d9"
CONTROLLER_SHA = "a32b0016d3228218c044fc885cf194fbb322324516390dd0f3bded8d6e0aeb49"
LAUNCHER_SHA = "f2e0ea0c9e13be045c6f3cbc65f56bda0c07a6921a00937998c4340c3e4cde8e"
SOURCE_PINS = {
    "control": "c10da511cc7b792c881e9b5d20acd1491462d44a7c40dc60c761a4286733629c",
    "transport": "c16c8fc637737449d9eaa608ba32647642d4b8bd0b9d572453aea54db7ff6981",
    "parser": "571eb1ce9b8a2218bb00f4c64d2b82d56c889d8922f4cfb5ab21f47fa6d49a9f",
    "controller": CONTROLLER_SHA,
    "numeric_plan": PLAN_SHA,
}
STAGES = {
    "native16k_cap512": "numeric21-native16k-cap512-omp1",
    "fixed4_raw": "numeric21-fixed4-raw-omp1",
    "short_piecewise": "numeric21-short-piecewise-omp1-v2",
}
P128 = [13, 14, 17, 18, 19, 20]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def strict_json(raw):
    def reject(value):
        raise ValueError("Non-finite JSON: " + value)

    return json.loads(raw, parse_constant=reject)


def record(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": sha(raw)}


def checked(saved, expected_sha=None):
    actual = record(saved["path"])
    require(
        all(actual[k] == saved[k] for k in ("bytes", "sha256")),
        "Artifact changed: " + saved["path"],
    )
    require(
        expected_sha is None or actual["sha256"] == expected_sha,
        "Unexpected frozen source: " + saved["path"],
    )
    return actual


def read(path):
    return strict_json(Path(path).read_bytes())


def without_id(payload):
    return {k: v for k, v in payload.items() if k != "request_id"}


def first_diff(left, right):
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else None
        b = right[i] if i < len(right) else None
        if a != b:
            return {
                "token_index_zero_based": i,
                "left_token_id": a,
                "right_token_id": b,
            }
    return None


def raw_sse(row):
    """Collect complete byte-framed SSE records before validating their order."""
    raw = base64.b64decode(row["raw_sse_base64"], validate=True)
    require(
        len(raw) == row["raw_sse_bytes"] and sha(raw) == row["raw_sse_sha256"],
        "Raw SSE digest/count differs",
    )
    normalized = raw.replace(b"\r\n", b"\n")
    require(normalized.endswith(b"\n\n"), "Raw SSE lacks a complete final frame")
    values = []
    done = object()
    for frame in normalized.split(b"\n\n")[:-1]:
        fields = []
        for line in frame.split(b"\n"):
            if line.startswith(b"data:"):
                fields.append(line[5:].removeprefix(b" "))
            else:
                require(
                    not line or line.startswith((b":", b"event:", b"id:", b"retry:")),
                    "Unknown SSE field",
                )
        if not fields:
            continue
        data = b"\n".join(fields)
        values.append(
            done if data == b"[DONE]" else strict_json(data.decode("utf-8", "strict"))
        )
    require(
        values and values[-1] is done and sum(v is done for v in values) == 1,
        "DONE missing, repeated, or out of order",
    )
    records = values[:-1]
    require(
        all(isinstance(v, dict) and "error" not in v for v in records),
        "SSE error/non-object",
    )
    require(
        {v.get("id") for v in records} == {row["expected_response_id"]},
        "SSE response identity differs",
    )
    choices, usages = [], []
    for index, value in enumerate(records):
        group = value.get("choices")
        require(
            isinstance(group, list) and len(group) <= 1,
            "Unexpected completion choice count",
        )
        if group:
            require(value.get("usage") is None, "Usage mixed with completion choice")
            choices.append((index, group[0]))
        else:
            require(isinstance(value.get("usage"), dict), "Empty SSE has no usage")
            usages.append((index, value["usage"]))
    require(
        choices and len(usages) == 1 and usages[0][0] == len(records) - 1,
        "Usage must be the final JSON record",
    )
    require(
        all(type(c.get("index")) is int and c["index"] == 0 for _, c in choices),
        "Wrong choice index",
    )
    finished = [
        (i, c["finish_reason"])
        for i, c in choices
        if c.get("finish_reason") is not None
    ]
    require(
        finished == [(choices[-1][0], "length")],
        "Finish missing, repeated, or not final choice",
    )
    prompt = row["payload"]["prompt"]
    require(
        choices[0][1].get("prompt_token_ids") == prompt,
        "First completion has wrong/missing prompt IDs",
    )
    tokens, text = [], []
    event_token_counts = []
    for _, choice in choices:
        ids = choice.get("token_ids")
        require(
            isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids),
            "Invalid output token IDs",
        )
        if choice.get("prompt_token_ids") is not None:
            require(
                choice["prompt_token_ids"] == prompt
                and all(type(x) is int for x in choice["prompt_token_ids"]),
                "Prompt ID echo differs",
            )
        require(isinstance(choice.get("text"), str), "Missing completion text")
        tokens.extend(ids)
        text.append(choice["text"])
        event_token_counts.append(len(ids))
    usage = usages[0][1]
    expected = {
        "prompt_tokens": len(prompt),
        "completion_tokens": row["payload"]["max_tokens"],
        "total_tokens": len(prompt) + row["payload"]["max_tokens"],
    }
    require(len(tokens) == expected["completion_tokens"], "Incomplete output count")
    require(
        all(type(usage.get(k)) is int and usage[k] == v for k, v in expected.items()),
        "Usage differs from request",
    )
    require(
        tokens == row["token_ids"]
        and "".join(text) == row["text"]
        and usage == row["usage"],
        "Raw output differs from producer receipt",
    )
    require(
        row["http_status"] == 200
        and "text/event-stream" in row["content_type"]
        and row["transport_passed"] is True
        and row["intentionally_cancelled"] is False
        and row["finish_reason"] == "length",
        "Incomplete/failed transport",
    )
    require(
        row["response_id"] == row["expected_response_id"]
        and row["prompt_ids_observed"] is True
        and row["done_count"] == row["usage_count"] == row["finish_count"] == 1,
        "Terminal receipt differs",
    )
    return (
        raw,
        tokens,
        {
            "raw_sse_bytes": len(raw),
            "raw_sse_sha256": sha(raw),
            "data_events": len(values),
            "usage": usage,
            "prompt_ids_exact": True,
            "raw_token_ids_exact": True,
            "event_token_counts": event_token_counts,
            "multi_token_events": sum(x > 1 for x in event_token_counts),
        },
    )


def journal_check(report, requests, raw_bodies):
    source = checked(report["journal"])
    raw = Path(source["path"]).read_bytes()
    require(raw.endswith(b"\n"), "Journal partial final record")
    events = [strict_json(line) for line in raw.split(b"\n")[:-1]]
    require(
        [e["seq"] for e in events] == list(range(1, len(events) + 1)),
        "Journal sequence incomplete",
    )
    require(
        all(a["monotonic_ns"] <= b["monotonic_ns"] for a, b in zip(events, events[1:])),
        "Journal clocks reversed",
    )
    require(
        all(e["monotonic_s"] == e["monotonic_ns"] / 1e9 for e in events),
        "Journal second/nanosecond clocks differ",
    )
    expected_ids = {r["request_id"] for r in requests}
    actual_ids = {
        e["request_id"]
        for e in events
        if e["event"]
        in ("request_started", "request_finished", "byte_chunk", "response_headers")
    }
    require(expected_ids == actual_ids, "Journal has unknown/missing request IDs")
    chunks = 0
    for row, expected_raw in zip(requests, raw_bodies, strict=True):
        rid = row["request_id"]
        related = [e for e in events if e.get("request_id") == rid]
        start = [e for e in related if e["event"] == "request_started"]
        finish = [e for e in related if e["event"] == "request_finished"]
        headers = [e for e in related if e["event"] == "response_headers"]
        require(
            len(start) == len(finish) == 1
            and start[0]["payload"] == row["payload"]
            and start[0]["seq"] < finish[0]["seq"],
            "Journal request lifecycle differs",
        )
        require(
            len(headers) == 1
            and start[0]["seq"] < headers[0]["seq"] < finish[0]["seq"]
            and headers[0]["http_status"] == row["http_status"]
            and headers[0]["content_type"] == row["content_type"],
            "Journal headers differ/repeat",
        )
        require(
            finish[0]["transport_passed"] is True
            and finish[0]["raw_sse_sha256"] == row["raw_sse_sha256"]
            and finish[0]["intentionally_cancelled"] is False
            and finish[0]["error"] is None,
            "Journal terminal state differs",
        )
        require(
            row["started_monotonic_s"]
            <= start[0]["monotonic_s"]
            <= headers[0]["monotonic_s"]
            <= row["finished_monotonic_s"]
            <= finish[0]["monotonic_s"]
            and row["completed_monotonic_s"] == row["finished_monotonic_s"],
            "Request/journal lifetime clocks differ",
        )
        body_events = [e for e in related if e["event"] == "byte_chunk"]
        require(
            body_events and len(body_events) == len(row["byte_chunks"]),
            "Journal chunk count differs",
        )
        bodies, previous_received = [], headers[0]["monotonic_s"]
        for event, chunk in zip(body_events, row["byte_chunks"], strict=True):
            body = base64.b64decode(event["body_base64"], validate=True)
            require(
                headers[0]["seq"] < event["seq"] < finish[0]["seq"],
                "Raw bytes outside header/finish lifetime",
            )
            require(
                chunk
                == {
                    "received_monotonic_s": event["received_monotonic_s"],
                    "length": len(body),
                },
                "Recorded chunk length/time differs",
            )
            require(
                previous_received
                <= event["received_monotonic_s"]
                <= event["monotonic_s"]
                <= row["finished_monotonic_s"],
                "Chunk receive/journal clocks reverse",
            )
            previous_received = event["received_monotonic_s"]
            bodies.append(body)
        require(b"".join(bodies) == expected_raw, "Journal raw bytes differ")
        chunks += len(body_events)
    return {
        "source": source,
        "events": len(events),
        "byte_chunks": chunks,
        "requests": 21,
        "header_chunk_timestamp_binding": True,
        "passed": True,
    }, events


def history_check(report, requests):
    saved = report["server_log_retained"]
    source = checked(saved["snapshot"])
    raw = Path(source["path"]).read_bytes()
    before = report["server_log_before"]
    require(
        saved["initial"] == before and sha(raw[: before["bytes"]]) == before["sha256"],
        "Startup prefix changed",
    )
    require(
        b"Added request " not in raw[: before["bytes"]] and raw.endswith(b"\n"),
        "Prior generation or partial log line",
    )
    # Cursor framing is independent of the producer's split-based implementation.
    bindings, start, line = {}, 0, 0
    while start < len(raw):
        end = raw.find(b"\n", start)
        require(end >= 0, "Missing terminal LF")
        complete = raw[start : end + 1]
        line += 1
        if b"Added request " in complete:
            require(
                complete.count(b"Added request ") == 1,
                "Multiple Added records on a line",
            )
            tail = complete.split(b"Added request ", 1)[1]
            token = tail.split(None, 1)[0]
            require(token.endswith(b"."), "Malformed Added terminator")
            internal = token[:-1].decode("ascii")
            matching = [
                r
                for r in requests
                if re.fullmatch(
                    re.escape(r["expected_engine_request_id"]) + r"-[0-9a-f]{8}",
                    internal,
                )
            ]
            require(
                len(matching) == 1,
                "Unknown/malformed request polluted retained history",
            )
            rid = matching[0]["request_id"]
            require(rid not in bindings, "Duplicate Added request")
            bindings[rid] = {
                "engine_request_id": internal,
                "line_number": line,
                "offset_start": start,
                "offset_end": end + 1,
                "line_sha256": sha(complete),
            }
        start = end + 1
    require(
        set(bindings) == {r["request_id"] for r in requests},
        "Added history does not cover exactly21",
    )
    recorded = report["complete_history_check"]
    require(
        recorded["passed"] is True
        and recorded["retained_log_sha256"] == source["sha256"]
        and recorded["observed_unique_added_requests"] == 21
        and set(recorded["bindings"]) == set(bindings),
        "Producer history summary differs",
    )
    for rid, computed in bindings.items():
        prior = recorded["bindings"][rid]
        require(
            prior["engine_request_id"] == computed["engine_request_id"]
            and all(
                prior["source"][k] == v
                for k, v in computed.items()
                if k != "engine_request_id"
            ),
            "Added source mapping differs",
        )
    return {
        "source": source,
        "passed": True,
        "unique_added_requests": 21,
        "initial_added_requests": 0,
        "bindings": bindings,
        "scope": "Immutable retained prefix; no identical internal-batch inference",
    }


def phase_check(report, plan, requests, history, events):
    phases = report["phases"]
    require(
        len(phases) == 12
        and [p["name"] for p in phases] == [p["name"] for p in plan["phases"]],
        "Phase order differs",
    )
    previous = None
    phase_events = [e for e in events if e["event"] == "phase_started"]
    require(
        [e["name"] for e in phase_events] == [p["name"] for p in phases],
        "Journal phase order differs",
    )
    retained_log = Path(history["source"]["path"]).read_bytes()
    for actual, wanted, phase_event in zip(
        phases, plan["phases"], phase_events, strict=True
    ):
        require(
            all(actual[k] == v for k, v in wanted.items())
            and actual["status"] == "passed",
            "Incomplete or changed phase",
        )
        begin, end = actual["started_monotonic_s"], actual["finished_monotonic_s"]
        require(
            begin <= end and (previous is None or previous <= begin),
            "Phase lifetimes overlap/reverse",
        )
        require(
            begin <= phase_event["monotonic_s"] <= end,
            "Journal phase start outside phase",
        )
        selected = [requests[i] for i in wanted["request_indices"]]
        require(
            all(
                begin <= r["started_monotonic_s"] <= r["finished_monotonic_s"] <= end
                for r in selected
            ),
            "Request outside sequential phase",
        )
        added = actual["added_requests"]
        require(
            {r["request_id"] for r in added} == {r["request_id"] for r in selected},
            "Phase Added set differs",
        )
        require(
            all(
                history["bindings"][r["request_id"]]["engine_request_id"]
                == r["engine_request_id"]
                for r in added
            ),
            "Phase internal ID differs",
        )
        for observed in added:
            source = observed["source"]
            binding = history["bindings"][observed["request_id"]]
            require(
                all(
                    source[k] == binding[k]
                    for k in (
                        "line_number",
                        "offset_start",
                        "offset_end",
                        "line_sha256",
                    )
                ),
                "Phase Added offsets differ",
            )
            require(
                base64.b64decode(source["line_base64"], validate=True)
                == retained_log[binding["offset_start"] : binding["offset_end"]],
                "Phase Added bytes differ",
            )
        previous = end
    controls = report["controls"]
    expected = []
    for phase in phases[:3]:
        for suffix, method, path, params, reply in (
            (
                "_pause",
                "POST",
                "/pause",
                {"mode": "keep", "clear_cache": "false"},
                {"status": "paused"},
            ),
            ("_paused", "GET", "/is_paused", {}, {"is_paused": True}),
            ("_barrier", "GET", "/is_paused", {}, {"is_paused": True}),
            ("_resume", "POST", "/resume", {}, {"status": "resumed"}),
        ):
            expected.append((phase["name"] + suffix, method, path, params, reply))
    require(len(controls) == len(expected) == 12, "Unexpected pause/resume call count")
    utility_events = [
        e for e in events if e["event"] in ("utility_started", "utility_finished")
    ]
    require(
        len(utility_events) == 24
        and {e["utility_id"] for e in utility_events} == set(range(1, 13)),
        "Unknown/missing utility journal records",
    )
    previous_finished = 0
    for index, (actual, wanted) in enumerate(
        zip(controls, expected, strict=True), start=1
    ):
        require(
            (
                actual["label"],
                actual["method"],
                actual["path"],
                actual["params"],
                actual["response_json"],
            )
            == wanted
            and actual["utility_id"] == index
            and actual["http_status"] == 200
            and actual["passed"] is True,
            "Control body/response/order differs",
        )
        require(
            strict_json(
                base64.b64decode(actual["response_body_base64"], validate=True).decode(
                    "utf-8", "strict"
                )
            )
            == wanted[-1],
            "Raw utility response differs",
        )
        paired = [e for e in utility_events if e["utility_id"] == index]
        require(
            [e["event"] for e in paired] == ["utility_started", "utility_finished"],
            "Utility start/finish journal mismatch",
        )
        start, finish = paired
        require(
            previous_finished
            < start["seq"]
            == actual["started_seq"]
            < finish["seq"]
            == actual["finished_seq"]
            and start["monotonic_ns"] == actual["started_monotonic_ns"]
            and finish["monotonic_ns"] == actual["finished_monotonic_ns"],
            "Utility journal sequence/clock differs",
        )
        require(
            all(
                start[k] == actual[k]
                for k in ("utility_id", "label", "method", "path", "params")
            ),
            "Utility start fields differ",
        )
        require(
            all(
                finish[k] == value
                for k, value in actual.items()
                if k not in ("finished_seq", "finished_monotonic_ns")
            ),
            "Utility finish fields differ",
        )
        previous_finished = finish["seq"]
    added_events = [e for e in events if e["event"] == "added_request_observed"]
    require(
        len(added_events) == 12
        and {e["request_id"] for e in added_events}
        == {r["request_id"] for r in requests[:12]},
        "Unexpected queued Added observations",
    )
    for phase_index, phase in enumerate(phases[:3]):
        pause, paused, barrier, resume = controls[phase_index * 4 : phase_index * 4 + 4]
        selected_ids = {requests[i]["request_id"] for i in phase["request_indices"]}
        starts = [
            e
            for e in events
            if e["event"] == "request_started" and e["request_id"] in selected_ids
        ]
        finishes = [
            e
            for e in events
            if e["event"] == "request_finished" and e["request_id"] in selected_ids
        ]
        added = [e for e in added_events if e["request_id"] in selected_ids]
        require(
            len(added) == len(selected_ids), "Queued Added observations repeat/missing"
        )
        require(
            phase_events[phase_index]["seq"] < pause["started_seq"]
            and pause["finished_seq"] < paused["started_seq"]
            and paused["finished_seq"] < min(e["seq"] for e in starts)
            and max(e["seq"] for e in starts) < min(e["seq"] for e in added)
            and max(e["seq"] for e in added) < barrier["started_seq"]
            and barrier["finished_seq"] < resume["started_seq"]
            and resume["started_seq"] < min(e["seq"] for e in finishes)
            and resume["finished_monotonic_ns"] / 1e9 <= phase["finished_monotonic_s"],
            "Queued pause/Added/barrier/resume order differs",
        )
        expected_added = {r["request_id"]: r for r in phase["added_requests"]}
        for event in added:
            require(
                event["case"] == phase["name"]
                and all(
                    event[k] == value
                    for k, value in expected_added[event["request_id"]].items()
                ),
                "Journal Added evidence differs from retained source",
            )
    require(
        report["remaining_tasks"] == 0
        and report["pause_state_uncertain"] is False
        and not report.get("cleanup_resume_error")
        and not report.get("cleanup_stream_error"),
        "Cleanup incomplete",
    )
    require(
        len(report["health_checks"]) == 2
        and all(r["passed"] is True for r in report["health_checks"]),
        "Health checks incomplete",
    )
    return {
        "passed": True,
        "phases": 12,
        "controls": 12,
        "remaining_tasks": 0,
        "utility_journal_paired": True,
        "queued_barrier_order_proven": True,
        "added_observations_bound_to_retained_log": 12,
        "resume_limit": (
            "Requests may finish before the /resume HTTP response is received; "
            "phase completion follows both."
        ),
    }


def sequence_group(rows, indices):
    selected = [rows[i] for i in indices]
    require(
        len({r["body_without_request_id_sha256"] for r in selected}) == 1,
        "Repeated P128 bodies differ",
    )
    pairs = [
        {
            "left_sequence_index": a["sequence_index"],
            "right_sequence_index": b["sequence_index"],
            "exact_token_ids": a["token_ids"] == b["token_ids"],
            "first_difference": first_diff(a["token_ids"], b["token_ids"]),
        }
        for a, b in itertools.combinations(selected, 2)
    ]
    return {
        "request_indices": indices,
        "samples": len(selected),
        "all_sequences_equal": len({tuple(r["token_ids"]) for r in selected}) == 1,
        "all_first_tokens_equal": len({r["token_ids"][0] for r in selected}) == 1,
        "distinct_sequences": len({tuple(r["token_ids"]) for r in selected}),
        "raw_token_ids_by_index": {
            str(r["sequence_index"]): r["token_ids"] for r in selected
        },
        "pairs": pairs,
    }


def review_role(directory, role, stage, plan):
    path = directory / (stage + "-report.json")
    report_source = record(path)
    report = read(path)
    require(
        report.get("schema") == "short-piecewise-numeric-replay-v1"
        and report["role"] == role
        and report["stage"] == stage,
        "Wrong stage/role report",
    )
    require(
        report["executed"] is True
        and report["complete"] is True
        and report["transport_passed"] is True
        and report["passed"] is True
        and report["exit_code"] == 0,
        "Role transport/controller incomplete",
    )
    sources = {
        "report": report_source,
        "producer": checked(report["producer"], CONTROLLER_SHA),
        "manifest": checked(report["manifest"]),
        "numeric_plan": checked(report["numeric_plan"], PLAN_SHA),
        "registration": checked(report["stage_registration"]),
        "launch_receipt": checked(report["launch_receipt"]),
    }
    manifest = read(sources["manifest"]["path"])
    require(
        manifest == report["manifest_contents"]
        and manifest["role"] == role
        and manifest["stage"] == stage,
        "Manifest snapshot differs",
    )
    sources["launcher"] = checked(manifest["launcher"], LAUNCHER_SHA)
    require(
        manifest["instrumentation"]
        == {"observer": False, "trace": False, "profiler": False},
        "Unexpected instrumentation",
    )
    require(
        manifest["runtime_commit"] == plan["roles"][role].get("runtime_commit"),
        "Wrong role runtime",
    )
    sources["client_sources"] = {}
    for name, expected in SOURCE_PINS.items():
        sources["client_sources"][name] = {
            key: checked(report["client_sources"][name][key], expected)
            for key in ("source", "snapshot")
        }
    sources["plugin_files"] = [
        record(Path(manifest["plugin_source"]) / "vllm_muxpd" / name)
        for name in manifest["source_hashes"]
    ]
    require(
        all(
            row["sha256"] == manifest["source_hashes"][Path(row["path"]).name]
            for row in sources["plugin_files"]
        ),
        "Runtime snapshot hash differs",
    )
    expanded = report["plan"]
    require(
        sha(canonical(expanded)) == report["plan_sha256"]
        and expanded["generation_requests"] == 21,
        "Expanded plan hash/count differs",
    )
    registration = read(sources["registration"]["path"])
    require(
        registration == report["stage_registration_contents"]
        and registration["execution_registered"] is True
        and registration["role"] == role
        and registration["stage"] == stage
        and registration["expanded_plan_sha256"] == report["plan_sha256"]
        and registration["manifest_sha256"] == sources["manifest"]["sha256"]
        and registration["controller_sha256"] == CONTROLLER_SHA
        and registration["numeric_plan_sha256"] == PLAN_SHA,
        "Execution registration differs",
    )
    rows = report["requests"]
    require(
        len(rows) == len(expanded["requests"]) == 21
        and [r["sequence_index"] for r in rows] == list(range(21)),
        "Missing/extra/reordered request",
    )
    summaries, raw_bodies = [], []
    for index, (row, planned, source) in enumerate(
        zip(rows, expanded["requests"], plan["requests"], strict=True)
    ):
        require(
            all(row[k] == v for k, v in planned.items()),
            "Receipt changed its request plan",
        )
        require(
            without_id(row["payload"]) == without_id(source["source_payload"]),
            "Request body changed beyond unique ID",
        )
        require(
            row["source_request_index"] == source["source_request_index"]
            and row["body_without_request_id_sha256"]
            == sha(canonical(without_id(row["payload"])))
            == source["body_without_request_id_sha256"],
            "Body/source provenance differs",
        )
        require(
            row["request_id"]
            == row["payload"]["request_id"]
            == row["request_headers"]["X-Request-Id"]
            and row["expected_response_id"] == "cmpl-" + row["request_id"]
            and row["expected_engine_request_id"] == row["expected_response_id"] + "-0",
            "Explicit request identity differs",
        )
        body, tokens, raw_summary = raw_sse(row)
        raw_bodies.append(body)
        summaries.append(
            {
                "sequence_index": index,
                "case": row["case"],
                "role": row["role"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "request_id": row["request_id"],
                "body_without_request_id_sha256": row["body_without_request_id_sha256"],
                "prompt_sha256": sha(canonical(row["payload"]["prompt"])),
                "token_ids": tokens,
                "token_ids_sha256": sha(canonical(tokens)),
                "raw": raw_summary,
            }
        )
    history = history_check(report, rows)
    journal, events = journal_check(report, rows, raw_bodies)
    phases = phase_check(report, plan, rows, history, events)
    return {
        "passed": True,
        "role": role,
        "stage": stage,
        "sources": sources,
        "configuration_bundle": {
            "runtime_commit": manifest["runtime_commit"],
            "argv": manifest["argv"],
            "environment_policy": manifest["environment_policy"],
            "native_repositories": manifest["native_repositories"],
        },
        "transport_requests": 21,
        "requests": summaries,
        "history": history,
        "phases": phases,
        "journal": journal,
        "p128": {
            "initial_pair": sequence_group(summaries, [13, 14]),
            "post_p511_four": sequence_group(summaries, [17, 18, 19, 20]),
            "all_six": sequence_group(summaries, P128),
        },
    }


def audit(directory):
    result = {
        "schema": "numeric21-three-role-independent-review-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer": record(__file__),
        "passed": False,
        "status": "incomplete",
        "roles": {},
        "errors": [],
        "interpretation": [
            "Audit passed means complete, internally consistent transport and "
            "retained request history, not output equivalence.",
            "Initial P128 pair and post-P511 repetitions have different histories; "
            "keep them separate.",
            "Three preserved configuration bundles differ in graph mode, asynchronous "
            "scheduling, communication and runtime implementation; "
            "no single-mechanism causal inference.",
            "No numerical-error, cache-contamination, semantic-quality or performance "
            "conclusion follows from token equality/divergence alone.",
            "Added/pause history does not prove identical internal model batches. "
            "These are finite sequential, nonrandomized diagnostics.",
        ],
    }
    try:
        plan_path = directory / "short-piecewise-numeric-replay-plan.json"
        result["plan"] = record(plan_path)
        require(result["plan"]["sha256"] == PLAN_SHA, "Wrong frozen numeric plan")
        plan = read(plan_path)
        result["original126"] = checked(plan["source_report"])
        original = read(result["original126"]["path"])["requests"]
        for index, saved in enumerate(plan["requests"]):
            require(
                saved["source_payload"]
                == original[saved["source_request_index"]]["payload"]
                and saved["source_request_index"] == (index if index < 17 else 13),
                "Frozen plan original-body alignment failed",
            )
        for role, stage in STAGES.items():
            try:
                result["roles"][role] = review_role(directory, role, stage, plan)
            except Exception as error:
                path = directory / (stage + "-report.json")
                result["roles"][role] = {
                    "passed": False,
                    "stage": stage,
                    "report": record(path)
                    if path.exists()
                    else {"path": str(path), "missing": True},
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
                result["errors"].append(
                    {"role": role, "type": type(error).__name__, "message": str(error)}
                )
        if result["errors"]:
            result["status"] = "failed_or_missing_input"
            return result
        require(
            len(
                {
                    r["request_id"]
                    for role in result["roles"].values()
                    for r in role["requests"]
                }
            )
            == 63,
            "Request IDs reused across roles",
        )
        cross = []
        for left, right in itertools.combinations(STAGES, 2):
            rows = []
            for a, b in zip(
                result["roles"][left]["requests"],
                result["roles"][right]["requests"],
                strict=True,
            ):
                require(
                    a["body_without_request_id_sha256"]
                    == b["body_without_request_id_sha256"],
                    "Cross-role request bodies differ",
                )
                rows.append(
                    {
                        "sequence_index": a["sequence_index"],
                        "case": a["case"],
                        "body_without_request_id_sha256": a[
                            "body_without_request_id_sha256"
                        ],
                        "exact_token_ids": a["token_ids"] == b["token_ids"],
                        "first_difference": first_diff(a["token_ids"], b["token_ids"]),
                    }
                )
            cross.append(
                {
                    "left_role": left,
                    "right_role": right,
                    "requests": rows,
                    "descriptive_exact_count": sum(r["exact_token_ids"] for r in rows),
                    "denominator": 21,
                }
            )
        result["cross_role_request_comparisons"] = cross
        original_pair = []
        for index in (13, 14):
            _, tokens, raw_summary = raw_sse(original[index])
            original_pair.append(
                {
                    "source_request_index": index,
                    "case": original[index]["case"],
                    "token_ids": tokens,
                    "body_without_request_id_sha256": sha(
                        canonical(without_id(original[index]["payload"]))
                    ),
                    "raw": raw_summary,
                }
            )
        result["original126_initial_p128_pair"] = {
            "requests": original_pair,
            "first_difference": first_diff(
                original_pair[0]["token_ids"], original_pair[1]["token_ids"]
            ),
            "per_role_same_sequence_comparisons": [
                {
                    "role": role,
                    "sequence_index": old["source_request_index"],
                    "exact_token_ids": old["token_ids"]
                    == result["roles"][role]["requests"][old["source_request_index"]][
                        "token_ids"
                    ],
                    "first_difference": first_diff(
                        old["token_ids"],
                        result["roles"][role]["requests"][old["source_request_index"]][
                            "token_ids"
                        ],
                    ),
                }
                for role in STAGES
                for old in original_pair
            ],
            "limit": (
                "Original126 used observer/trace and an earlier fresh process; "
                "differences are not solely attributable to short-P graphs."
            ),
        }
        result.update(
            passed=True,
            status="passed_evidence_consistency",
            audited_requests=63,
            cross_role_body_match=True,
            numerical_equivalence_claimed=False,
            causal_claimed=False,
        )
    except Exception as error:
        result["errors"].append({"type": type(error).__name__, "message": str(error)})
        result["status"] = "failed"
    return result


def markdown(result):
    lines = [
        "# 三配置匹配 21 请求：独立数值审计",
        "",
        "通过仅表示原始传输、请求体、来源和保留历史一致，"
        "不表示数值等价或因果结论。所有首次分歧位置从 0 开始。",
        "",
        f"审计状态：`{result['status']}`。",
    ]
    if result["passed"]:
        lines += [
            "",
            "| 配置 | 完整传输／Added | 初始 P128 两次全序列一致 "
            "| P511 后四次一致 | 六次不同序列数 |",
            "|---|---:|---|---|---:|",
        ]
        for role, value in result["roles"].items():
            p = value["p128"]
            lines.append(
                f"| {role} | 21 / 21 | {p['initial_pair']['all_sequences_equal']} "
                f"| {p['post_p511_four']['all_sequences_equal']} "
                f"| {p['all_six']['distinct_sequences']} |"
            )
        lines += ["", "跨配置按相同序号、相同请求体（仅 ID 不同）逐条对齐：", ""]
        for pair in result["cross_role_request_comparisons"]:
            lines.append(
                f"- `{pair['left_role']}` → `{pair['right_role']}`："
                f"{pair['descriptive_exact_count']}/21 条完整输出 token 序列相同；"
                "这是描述统计，不是精度评分。"
            )
        lines += [
            "",
            "各次短输出 token IDs、全部 15 对 P128 首次分歧、初始／后四次分组、"
            "原 126 初始对照及逐请求跨配置分歧均保留在 JSON；不复制大 prompt 数组。",
        ]
    else:
        lines += ["", "不完整／失败证据保持原样，未推导跨配置数值结论。", ""]
        for error in result["errors"]:
            lines.append(
                f"- {error.get('role', 'input')}：{error['type']}：{error['message']}"
            )
    lines += ["", "解释边界：", ""]
    lines += ["- " + note for note in result["interpretation"]]
    lines += [
        "",
        f"离线审计源码 SHA：`{result['producer']['sha256']}`。"
        "原报告、manifest、客户端源码、raw SSE、日志和 journal 的 SHA 均在 JSON。",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output, md = args.output.resolve(), args.output.resolve().with_suffix(".md")
    require(
        output != md and not output.exists() and not md.exists(),
        "Never overwrite an existing audit",
    )
    result = audit(args.directory.resolve())
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with md.open("x") as stream:
        stream.write(markdown(result))
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "status": result["status"],
                "output": record(output),
                "markdown": record(md),
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
