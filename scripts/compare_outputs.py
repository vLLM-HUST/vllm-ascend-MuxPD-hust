"""Compare synthetic correctness cases across two serving configurations.

Run each endpoint separately, first with one request at a time, then with two
concurrent requests. The long-prompt pair is dispatched after a counting request
has emitted its first content event. Text differences are reported, not assumed
to be either numerical drift or a state bug. This is not a latency benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import httpx


def cases():
    context = "\n".join(f"条目{i:03d}：本条标记为青色，数量为2。" for i in range(180))
    return [
        {
            "name": "counting",
            "prompt": "只输出从1到100的十进制整数，用英文逗号分隔，不要空格或说明。",
            "max_tokens": 64,
            "oracle": {"kind": "integer_prefix", "minimum": 20},
        },
        {
            "name": "long_extract",
            "prompt": context
            + "\n最终校验码是 MUXPD-7341。请只输出最终校验码，不要说明。",
            "max_tokens": 24,
            "oracle": {"kind": "exact", "expected": "MUXPD-7341"},
            "minimum_prompt_tokens": 2049,
        },
        {
            "name": "arithmetic",
            "prompt": "19加23等于多少？只输出十进制数字。",
            "max_tokens": 12,
            "oracle": {"kind": "exact", "expected": "42"},
        },
        {
            "name": "extract_distinct",
            "prompt": "编号甲为 ZX-521，编号乙为 ZX-907。请只输出编号乙，不要说明。",
            "max_tokens": 24,
            "oracle": {"kind": "exact", "expected": "ZX-907"},
        },
    ]


def check_oracle(text, oracle):
    stripped = text.strip()
    if oracle["kind"] == "exact":
        return stripped == oracle["expected"]
    # The last integer may be truncated by max_tokens. Validate every complete
    # comma-terminated integer, and require at least twenty of them.
    compact = re.sub(r"\s+", "", stripped)
    complete = compact.split(",")[:-1]
    if len(complete) < oracle["minimum"]:
        return False
    return all(value == str(i + 1) for i, value in enumerate(complete))


async def stream_case(client, args, base_url, case, mode, repeat, first_content=None):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": case["max_tokens"],
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    row = {
        "case": case["name"],
        "mode": mode,
        "repeat": repeat,
        "started_monotonic_s": time.monotonic(),
        "text": "",
        "events": [],
        "usage": None,
        "done": False,
        "finish_reason": None,
    }
    try:
        async with client.stream(
            "POST", base_url.rstrip("/") + "/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                elapsed = time.perf_counter() - started
                if data == "[DONE]":
                    row["done"] = True
                    break
                event = json.loads(data)
                row["events"].append({"elapsed_s": elapsed, "data": event})
                if "error" in event:
                    raise RuntimeError(event["error"])
                row["usage"] = event.get("usage") or row["usage"]
                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content") or ""
                    if content:
                        row["text"] += content
                        row.setdefault("ttft_s", elapsed)
                        if first_content is not None:
                            first_content.set()
                    row["finish_reason"] = (
                        choice.get("finish_reason") or row["finish_reason"]
                    )
        if not (row["done"] and row["text"] and row["usage"] and row["finish_reason"]):
            raise RuntimeError("Incomplete SSE response, finish reason or usage")
        if row["usage"]["prompt_tokens"] < case.get("minimum_prompt_tokens", 1):
            raise RuntimeError("Long-prompt case did not reach its minimum token count")
        row["oracle_passed"] = check_oracle(row["text"], case["oracle"])
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["oracle_passed"] = False
    finally:
        row["duration_s"] = time.perf_counter() - started
        row["text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
    return row


async def run_endpoint(client, args, label, base_url, test_cases):
    result = {"label": label, "base_url": base_url, "requests": [], "pairs": []}
    try:
        response = await client.get(base_url.rstrip("/") + "/health")
        response.raise_for_status()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    for repeat in range(args.repeats):
        # Reverse isolated order in alternate repeats to expose cross-request
        # state leakage, while keeping each endpoint's request sequence equal.
        isolated = test_cases if repeat % 2 == 0 else list(reversed(test_cases))
        for case in isolated:
            result["requests"].append(
                await stream_case(client, args, base_url, case, "isolated", repeat)
            )
        first_content = asyncio.Event()
        first = asyncio.create_task(
            stream_case(
                client, args, base_url, test_cases[0], "paired", repeat, first_content
            )
        )
        ready = asyncio.create_task(first_content.wait())
        try:
            await asyncio.wait(
                (first, ready),
                timeout=args.timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            started_after_first_content = first_content.is_set()
            still_running = not first.done()
            if started_after_first_content:
                second = await stream_case(
                    client, args, base_url, test_cases[1], "paired", repeat
                )
                result["requests"].append(second)
            else:
                result["pairs"].append(
                    {"repeat": repeat, "error": "Counting request emitted no content"}
                )
            result["requests"].append(await first)
            result["pairs"].append(
                {
                    "repeat": repeat,
                    "long_prompt_after_first_content": started_after_first_content,
                    "first_request_active_at_second_dispatch": still_running,
                }
            )
        finally:
            ready.cancel()
            if not first.done():
                first.cancel()
            await asyncio.gather(ready, first, return_exceptions=True)
        result["requests"].extend(
            await asyncio.gather(
                *(
                    stream_case(client, args, base_url, case, "paired", repeat)
                    for case in test_cases[2:]
                )
            )
        )
    return result


def summarize(endpoints):
    summary = {"endpoints": {}, "cross_endpoint_text_matches": []}
    indexes = {}
    for endpoint in endpoints:
        groups = {}
        indexes[endpoint["label"]] = {}
        for row in endpoint["requests"]:
            groups.setdefault((row["mode"], row["case"]), []).append(row)
            indexes[endpoint["label"]][(row["repeat"], row["mode"], row["case"])] = row
        summary["endpoints"][endpoint["label"]] = {
            "request_count": len(endpoint["requests"]),
            "oracle_failures": [
                {key: row[key] for key in ("repeat", "mode", "case")}
                for row in endpoint["requests"]
                if not row["oracle_passed"]
            ],
            "repeat_stability": [
                {
                    "mode": mode,
                    "case": case,
                    "samples": len(rows),
                    "distinct_texts": len({r["text_sha256"] for r in rows}),
                }
                for (mode, case), rows in sorted(groups.items())
            ],
            "isolated_vs_paired_text_matches": [
                {
                    "repeat": repeat,
                    "case": case,
                    "exact_text_match": (
                        row["text_sha256"]
                        == indexes[endpoint["label"]][(repeat, "paired", case)][
                            "text_sha256"
                        ]
                    ),
                }
                for (repeat, mode, case), row in sorted(
                    indexes[endpoint["label"]].items()
                )
                if mode == "isolated"
                and (repeat, "paired", case) in indexes[endpoint["label"]]
            ],
        }
    baseline = indexes.get("baseline", {})
    candidate = indexes.get("candidate", {})
    for key in sorted(baseline.keys() & candidate.keys()):
        summary["cross_endpoint_text_matches"].append(
            {
                "repeat": key[0],
                "mode": key[1],
                "case": key[2],
                "exact_text_match": (
                    baseline[key]["text_sha256"] == candidate[key]["text_sha256"]
                ),
            }
        )
    return summary


def load_baseline(path, args, test_cases):
    """Reuse only a complete, compatible and successful baseline receipt."""
    contents = Path(path).read_bytes()
    source = json.loads(contents)
    if source.get("complete") is not True:
        raise ValueError("Baseline report must have completed both endpoint runs")
    for name in ("model", "repeats"):
        if source.get("parameters", {}).get(name) != getattr(args, name):
            raise ValueError(f"Baseline report has incompatible {name}")
    if source.get("cases") != test_cases:
        raise ValueError("Baseline cases/prompts/output limits/oracles have changed")
    endpoints = [
        endpoint
        for endpoint in source.get("endpoints", [])
        if endpoint.get("label") == "baseline"
    ]
    if len(endpoints) != 1 or endpoints[0].get("error"):
        raise ValueError(
            "Baseline report must contain one successful baseline endpoint"
        )
    endpoint = endpoints[0]
    expected = {
        (repeat, mode, case["name"])
        for repeat in range(args.repeats)
        for mode in ("isolated", "paired")
        for case in test_cases
    }
    seen = set()
    by_name = {case["name"]: case for case in test_cases}
    for row in endpoint.get("requests", []):
        key = (row.get("repeat"), row.get("mode"), row.get("case"))
        if key not in expected or key in seen:
            raise ValueError("Baseline contains unknown or duplicate request receipts")
        seen.add(key)
        case = by_name[row["case"]]
        text = row.get("text", "")
        if (
            row.get("error")
            or not row.get("oracle_passed")
            or not row.get("done")
            or not row.get("finish_reason")
            or not check_oracle(text, case["oracle"])
            or row.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest()
            or row.get("usage", {}).get("prompt_tokens", 0)
            < case.get("minimum_prompt_tokens", 1)
        ):
            raise ValueError(f"Baseline contains an invalid/failed request: {key}")
    if seen != expected:
        raise ValueError("Baseline request coverage is incomplete")
    pairs = endpoint.get("pairs", [])
    if (
        len(pairs) != args.repeats
        or {pair.get("repeat") for pair in pairs} != set(range(args.repeats))
        or any(
            not pair.get("long_prompt_after_first_content")
            or not pair.get("first_request_active_at_second_dispatch")
            for pair in pairs
        )
    ):
        raise ValueError("Baseline did not exercise the required overlapping requests")
    endpoint["reused_from"] = str(path)
    endpoint["reused_report_sha256"] = hashlib.sha256(contents).hexdigest()
    return endpoint


async def main(args):
    test_cases = cases()
    report = {
        "parameters": vars(args),
        "cases": test_cases,
        "note": (
            "Synthetic correctness probe, not a performance benchmark. Exact text "
            "differences do not by themselves identify numerical drift or state "
            "corruption. Inspect isolated equality, repeat stability and oracles. "
            "Client request overlap does not prove pending P or device overlap."
        ),
        "endpoints": [],
        "complete": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    targets = [("candidate", args.candidate_url)]
    if args.baseline_report:
        report["endpoints"].append(
            load_baseline(args.baseline_report, args, test_cases)
        )
        print(json.dumps({"baseline_reused_from": args.baseline_report}), flush=True)
    else:
        targets.insert(0, ("baseline", args.baseline_url))
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        for label, base_url in targets:
            report["endpoints"].append(
                await run_endpoint(client, args, label, base_url, test_cases)
            )
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(
                json.dumps({"endpoint_finished": label, "base_url": base_url}),
                flush=True,
            )
    report["summary"] = summarize(report["endpoints"])
    report["complete"] = True
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    failed = any(
        endpoint.get("error")
        or len(endpoint["requests"]) != args.repeats * len(test_cases) * 2
        or any(not row["oracle_passed"] for row in endpoint["requests"])
        or any(
            not pair.get("first_request_active_at_second_dispatch", False)
            for pair in endpoint["pairs"]
        )
        for endpoint in report["endpoints"]
    )
    if args.require_exact:
        failed |= any(
            not row["exact_text_match"]
            for row in report["summary"]["cross_endpoint_text_matches"]
        )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-url")
    baseline.add_argument(
        "--baseline-report",
        help="Reuse a completed compatible baseline endpoint from this JSON report",
    )
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen3.5-35B-A3B")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--require-exact", action="store_true")
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    raise SystemExit(asyncio.run(main(arguments)))
