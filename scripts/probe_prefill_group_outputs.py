"""Finite 19-request token-ID diagnostic; defaults to writing a plan only.

Create: --plan-only --output plan.json
Run: --execute --plan plan.json --plan-sha256 SHA --base-url URL --label LABEL
     --output report.json [--baseline-report baseline.json]
Offline: --compare-report candidate.json --baseline-report baseline.json
         --output diff.json
No logprobs, server control, model changes, priming, or performance claims.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import hashlib
import itertools
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:
    from .benchmark_contention import SSEReceipt
except ImportError:
    from benchmark_contention import SSEReceipt

PLAN_SCHEMA = "muxpd-prefill-group-numeric-plan-v1"
REPORT_SCHEMA = "muxpd-prefill-group-numeric-report-v1"
# Exact P128 IDs from the matched numeric21 plan request 13. This retains the
# known index-4 repeat-instability probe; other prompts below are synthetic.
SHORT_PROMPT = [
    int(token)
    for token in (
        "961 416 220 985 208 264 948 156 395 291 839 240 275 136 177 401 "
        "628 959 429 957 250 228 127 243 691 278 120 878 741 442 575 822 "
        "477 387 411 151 203 891 728 363 161 965 205 554 990 527 781 639 "
        "174 301 973 403 972 827 189 951 582 895 705 283 552 799 406 566 "
        "601 898 546 404 956 332 534 990 608 133 961 357 672 570 866 869 "
        "454 930 536 368 738 556 660 838 285 679 759 118 935 371 648 212 "
        "144 951 346 831 231 696 601 913 853 532 448 492 737 953 953 386 "
        "186 739 970 576 352 939 418 889 659 594 212 223 344 186 233 616"
    ).split()
]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".checkpoint")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def build_plan(model="Qwen3.5-35B-A3B"):
    require(isinstance(model, str) and model, "A model name is required")
    requests, phases = [], []
    long_prompt = [100 + (i * 31) % 900 for i in range(12000)]

    def add(phase, role, repeat, prompt, outputs, lane=None):
        payload = {
            "model": model,
            "prompt": list(prompt),
            "max_tokens": outputs,
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        }
        index = len(requests)
        requests.append(
            {
                "sequence_index": index,
                "phase": phase,
                "role": role,
                "repeat": repeat,
                "lane": lane,
                "payload": payload,
                "payload_sha256": digest(payload),
                "prompt_sha256": digest(prompt),
            }
        )
        return index

    for phase, prompt in (("initial_short", SHORT_PROMPT), ("solo_long", long_prompt)):
        for repeat in range(3):
            index = add(phase, "prefill", repeat, prompt, 8)
            phases.append(
                {"kind": "serial", "name": f"{phase}-{repeat}", "indices": [index]}
            )
    for repeat in range(2):
        indices = [
            add(
                "mixed_decode",
                "decode",
                repeat,
                [100 + (i * 31 + lane * 17) % 900 for i in range(128)],
                128,
                lane,
            )
            for lane in range(4)
        ]
        indices.append(add("mixed_long", "prefill", repeat, long_prompt, 8))
        phases.append({"kind": "mixed", "name": f"mixed-{repeat}", "indices": indices})
    for repeat in range(3):
        index = add("after_short", "prefill", repeat, SHORT_PROMPT, 8)
        phases.append(
            {"kind": "serial", "name": f"after_short-{repeat}", "indices": [index]}
        )
    body = {
        "schema": PLAN_SCHEMA,
        "model": model,
        "generation_requests": 19,
        "requests": requests,
        "phases": phases,
        "history_contract": (
            "3 short, 3 solo long, twice (4 ongoing D then long P), 3 short; "
            "each phase fully drains"
        ),
        "mixed_gate": (
            "All four D first token receipts and all four client tasks "
            "still active before P dispatch"
        ),
        "limits": (
            "Client concurrency is not proof of server batches/device overlap; "
            "request IDs vary by execution"
        ),
    }
    return {**body, "plan_sha256": digest(body)}


def validate_plan(plan):
    require(
        plan == build_plan(plan.get("model")),
        "Plan differs from the fixed 19-request contract",
    )


def first_difference(left, right):
    for index, (a, b) in enumerate(itertools.zip_longest(left, right)):
        if a != b:
            return {"token_index": index, "left_token_id": a, "right_token_id": b}
    return None


def pair_differences(rows):
    return [
        {
            "left_sequence_index": a["sequence_index"],
            "right_sequence_index": b["sequence_index"],
            "first_difference": first_difference(a["token_ids"], b["token_ids"]),
        }
        for a, b in itertools.combinations(rows, 2)
    ]


def summarize(report):
    successful = [r for r in report["requests"] if r.get("transport_passed")]
    groups = []
    for phase, lane, expected in [
        ("initial_short", None, 3),
        ("solo_long", None, 3),
        ("mixed_long", None, 2),
        ("after_short", None, 3),
        *[("mixed_decode", lane, 2) for lane in range(4)],
    ]:
        rows = [r for r in successful if r["phase"] == phase and r["lane"] == lane]
        sequences = {tuple(r["token_ids"]) for r in rows}
        groups.append(
            {
                "phase": phase,
                "lane": lane,
                "expected": expected,
                "samples": len(rows),
                "complete": len(rows) == expected,
                "distinct_sequences": len(sequences),
                "stable": len(sequences) == 1 if len(rows) == expected else None,
                "pairwise_first_differences": pair_differences(rows),
            }
        )
    history = []
    for a, b in itertools.product(successful, repeat=2):
        if (a["phase"], b["phase"]) in (
            ("initial_short", "after_short"),
            ("solo_long", "mixed_long"),
        ):
            history.append(
                {
                    "left_sequence_index": a["sequence_index"],
                    "right_sequence_index": b["sequence_index"],
                    "left_history": a["phase"],
                    "right_history": b["phase"],
                    "first_difference": first_difference(
                        a["token_ids"], b["token_ids"]
                    ),
                }
            )
    return {
        "successful_requests": len(successful),
        "repeat_groups": groups,
        "different_history_comparisons": history,
        "not_dispatched": sorted(
            set(range(19)) - {r["sequence_index"] for r in report["requests"]}
        ),
        "interpretation": (
            "Differences do not stop collection or establish their cause. "
            "Compare only common prefixes before first divergence."
        ),
    }


def compare_reports(baseline, candidate):
    require(
        baseline["schema"] == candidate["schema"] == REPORT_SCHEMA,
        "Wrong report schema",
    )
    require(baseline["plan"] == candidate["plan"], "Comparison plans/bodies differ")
    validate_plan(baseline["plan"])

    def rows(report):
        found = {}
        for row in report["requests"]:
            index = row["sequence_index"]
            require(
                index not in found and 0 <= index < 19,
                "Duplicate/invalid request ordinal",
            )
            expected = report["plan"]["requests"][index]["payload"]
            require(
                {k: v for k, v in row["payload"].items() if k != "request_id"}
                == expected,
                "Response request body differs from plan",
            )
            if row.get("transport_passed"):
                tokens = row.get("token_ids")
                require(
                    isinstance(tokens, list)
                    and len(tokens) == expected["max_tokens"]
                    and all(type(token) is int and token >= 0 for token in tokens),
                    "Successful report contains invalid token IDs/count",
                )
            found[index] = row
        require(
            not report["complete"]
            or len(found) == 19
            and all(row.get("transport_passed") for row in found.values()),
            "Complete report lacks 19 successful requests",
        )
        return found

    left, right = rows(baseline), rows(candidate)
    comparisons = []
    for index in range(19):
        a, b = left.get(index), right.get(index)
        comparable = bool(
            a and b and a.get("transport_passed") and b.get("transport_passed")
        )
        delta = first_difference(a["token_ids"], b["token_ids"]) if comparable else None
        comparisons.append(
            {
                "sequence_index": index,
                "phase": baseline["plan"]["requests"][index]["phase"],
                "comparable": comparable,
                "exact_token_ids": delta is None if comparable else None,
                "first_difference": delta,
            }
        )
    return {
        "baseline_label": baseline["label"],
        "candidate_label": candidate["label"],
        "complete_evidence": bool(
            baseline["complete"]
            and candidate["complete"]
            and all(r["comparable"] for r in comparisons)
        ),
        "exact_sequences": sum(r["exact_token_ids"] is True for r in comparisons),
        "requests": comparisons,
        "numerical_equivalence_proven": False,
        "limits": (
            "Finite observations, different service histories/batches, "
            "not a semantic score or causal proof"
        ),
    }


class Journal:
    def __init__(self, path):
        self.stream = path.open("x")
        self.sequence = 0

    def event(self, kind, **fields):
        self.stream.write(
            json.dumps(
                {
                    "sequence": self.sequence,
                    "kind": kind,
                    "monotonic_s": time.monotonic(),
                    **fields,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.stream.flush()
        self.sequence += 1

    def close(self):
        self.stream.close()


async def execute_request(
    client, base_url, row, first, timeout, journal, save, active=None
):
    row.update(
        started_utc=utc_now(),
        started_monotonic_s=time.monotonic(),
        transport_passed=False,
    )
    receipt = SSEReceipt(row, first.set)

    async def receive():
        if active is not None:
            require(
                all(not task.done() for task in active),
                "D ended before long-P client dispatch",
            )
        async with client.stream(
            "POST",
            base_url.rstrip("/") + "/v1/completions",
            json=row["payload"],
            headers={"X-Request-Id": row["payload"]["request_id"]},
        ) as response:
            row.update(
                http_status=response.status_code,
                response_headers=dict(response.headers),
            )
            journal.event(
                "response_headers",
                sequence_index=row["sequence_index"],
                status=response.status_code,
                headers=dict(response.headers),
            )
            valid_http = (
                response.status_code == 200
                and response.headers.get("content-type", "").split(";")[0]
                == "text/event-stream"
            )
            async for chunk in response.aiter_bytes():
                received = time.monotonic()
                journal.event(
                    "http_body",
                    sequence_index=row["sequence_index"],
                    received_monotonic_s=received,
                    bytes_base64=base64.b64encode(chunk).decode("ascii"),
                )
                if valid_http:
                    receipt.feed(chunk, received)
                else:
                    receipt.raw.extend(chunk)
                    row["byte_chunks"].append(
                        {"received_monotonic_s": received, "length": len(chunk)}
                    )
            require(valid_http, "Non-200 or non-SSE response")
            receipt.finalize()
            require(
                row["response_id"] == "cmpl-" + row["payload"]["request_id"],
                "Response request ID differs",
            )

    try:
        journal.event(
            "request_started",
            sequence_index=row["sequence_index"],
            payload=row["payload"],
        )
        save()
        await asyncio.wait_for(receive(), timeout)
        row["transport_passed"] = True
    except (Exception, asyncio.CancelledError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt.preserve_raw()
        row.update(
            finished_utc=utc_now(),
            finished_monotonic_s=time.monotonic(),
            finish_count=receipt.finish_count,
            done_count=receipt.done_count,
            usage_count=receipt.usage_count,
        )
        journal.event(
            "request_finished",
            sequence_index=row["sequence_index"],
            transport_passed=row["transport_passed"],
            error=row.get("error"),
            raw_sse_sha256=row["raw_sse_sha256"],
            raw_sse_bytes=row["raw_sse_bytes"],
        )
        save()
    return row["transport_passed"]


async def await_gate(tasks, events, timeout):
    waiters = [asyncio.create_task(event.wait()) for event in events]
    deadline = time.monotonic() + timeout
    try:
        while not all(event.is_set() for event in events):
            require(
                all(not task.done() for task in tasks),
                "D terminated before all four first tokens",
            )
            remaining = deadline - time.monotonic()
            require(remaining > 0, "First-token gate timed out")
            await asyncio.wait(
                [*tasks, *[t for t in waiters if not t.done()]],
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
        require(
            all(not task.done() for task in tasks),
            "D completed before mixed-prefill gate",
        )
    finally:
        for waiter in waiters:
            if not waiter.done():
                waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)


async def complete_tasks(tasks):
    remaining = set(tasks)
    while remaining:
        finished, remaining = await asyncio.wait(
            remaining, return_when=asyncio.FIRST_COMPLETED
        )
        require(
            all(task.result() for task in finished), "Request transport/control failure"
        )


async def run_probe(plan, args, client, report, journal, save):
    tasks = []
    try:
        for phase in plan["phases"]:
            phase_row = {
                **phase,
                "started_monotonic_s": time.monotonic(),
                "complete": False,
            }
            report["phases"].append(phase_row)
            journal.event("phase_started", name=phase["name"])
            tasks = []
            rows, events = [], []

            def start(index, active=None):
                row = copy.deepcopy(plan["requests"][index])
                row["payload"]["request_id"] = f"group-{report['run_id']}-{index:02d}"
                row["executed_payload_sha256"] = digest(row["payload"])
                report["requests"].append(row)
                first = asyncio.Event()
                task = asyncio.create_task(
                    execute_request(
                        client,
                        args.base_url,
                        row,
                        first,
                        args.timeout,
                        journal,
                        save,
                        active,
                    )
                )
                rows.append(row)
                events.append(first)
                tasks.append(task)

            if phase["kind"] == "mixed":
                for index in phase["indices"][:4]:
                    start(index)
                await await_gate(tasks, events, args.gate_timeout)
                phase_row["all_D_first_tokens_monotonic_s"] = [
                    row["first_token_monotonic_s"] for row in rows
                ]
                phase_row["gate_monotonic_s"] = time.monotonic()
                phase_row["active_decode_indices_at_gate"] = phase["indices"][:4]
                journal.event(
                    "mixed_gate_passed",
                    name=phase["name"],
                    **{k: v for k, v in phase_row.items() if k.endswith("_s")},
                )
                start(phase["indices"][4], list(tasks))
            else:
                start(phase["indices"][0])
            await complete_tasks(tasks)
            if phase["kind"] == "mixed":
                prefill = rows[-1]
                phase_row["D_token_events_before_P_first_token"] = {
                    str(row["sequence_index"]): sum(
                        event.get("token_count", 0) > 0
                        and prefill["started_monotonic_s"]
                        <= event["received_monotonic_s"]
                        < prefill["first_token_monotonic_s"]
                        for event in row["events"]
                    )
                    for row in rows[:4]
                }
                phase_row["server_batch_or_device_overlap_proven"] = False
            phase_row.update(complete=True, finished_monotonic_s=time.monotonic())
            journal.event("phase_finished", name=phase["name"])
            save()
        report["complete"] = len(report["requests"]) == 19 and all(
            row["transport_passed"] for row in report["requests"]
        )
    except (Exception, asyncio.CancelledError) as error:
        report["complete"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        report["summary"] = summarize(report)
        report["finished_utc"] = utc_now()
        journal.event(
            "probe_finished", complete=report["complete"], error=report.get("error")
        )
        save()
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--compare-report", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--base-url")
    parser.add_argument("--label")
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--model", default="Qwen3.5-35B-A3B")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--gate-timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("timeout", "gate_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error("Timeouts must be positive finite seconds")
    if args.execute:
        if not (
            args.plan
            and args.plan_sha256
            and re.fullmatch(r"[0-9a-f]{64}", args.plan_sha256)
            and args.base_url
            and args.label
        ):
            parser.error(
                "Execution requires --plan, --plan-sha256, --base-url and --label"
            )
    elif args.compare_report:
        if not args.baseline_report or any(
            (args.plan, args.plan_sha256, args.base_url, args.label)
        ):
            parser.error(
                "Offline comparison requires baseline-report "
                "and rejects execution arguments"
            )
    elif any(
        (args.plan, args.plan_sha256, args.base_url, args.label, args.baseline_report)
    ):
        parser.error("Plan-only does not accept execution/comparison arguments")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    require(not output.exists(), "Output exists; never overwrite evidence")
    if args.compare_report:
        baseline, candidate = (
            json.loads(args.baseline_report.read_text()),
            json.loads(args.compare_report.read_text()),
        )
        value = {
            "baseline": source(args.baseline_report),
            "candidate": source(args.compare_report),
            "comparison": compare_reports(baseline, candidate),
        }
    elif not args.execute:
        value = build_plan(args.model)
    else:
        plan_source = source(args.plan)
        require(plan_source["sha256"] == args.plan_sha256, "Plan file SHA changed")
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        baseline = None
        if args.baseline_report:
            baseline = json.loads(args.baseline_report.read_text())
            require(
                baseline["schema"] == REPORT_SCHEMA
                and baseline["plan"] == plan
                and baseline["complete"],
                "Baseline is incomplete or uses a different plan",
            )
            compare_reports(baseline, baseline)
        journal_path = output.with_suffix(".events.jsonl")
        require(not journal_path.exists(), "Journal exists; never overwrite evidence")
        output.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "schema": REPORT_SCHEMA,
            "label": args.label,
            "base_url": args.base_url,
            "run_id": uuid.uuid4().hex[:16],
            "created_utc": utc_now(),
            "complete": False,
            "plan_source": plan_source,
            "plan": plan,
            "requests": [],
            "phases": [],
            "probe_source": source(__file__),
            "sse_parser_source": source(
                Path(__file__).with_name("benchmark_contention.py")
            ),
            "parameters": {
                "request_timeout_s": args.timeout,
                "gate_timeout_s": args.gate_timeout,
            },
            "limits": (
                "Not performance, model equivalence or proof of "
                "internal batch/physical overlap"
            ),
        }
        with output.open("x") as stream:
            json.dump(value, stream)
        journal = Journal(journal_path)
        try:
            journal.event("probe_started", plan_source=plan_source, label=args.label)

            async def execute():
                async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                    return await run_probe(
                        plan,
                        args,
                        client,
                        value,
                        journal,
                        lambda: checkpoint(output, value),
                    )

            asyncio.run(execute())
            if baseline is not None:
                value["baseline_report"] = source(args.baseline_report)
                value["comparison"] = compare_reports(baseline, value)
        except (Exception, KeyboardInterrupt) as error:
            value.update(complete=False, error=f"{type(error).__name__}: {error}")
            value["summary"] = summarize(value)
        finally:
            journal.close()
            value["journal"] = source(journal_path)
            checkpoint(output, value)
        print(json.dumps({"output": source(output), "complete": value["complete"]}))
        return int(not value["complete"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": source(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
