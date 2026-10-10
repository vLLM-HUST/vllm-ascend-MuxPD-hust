"""Reproducible staggered P/D workload with raw SSE timing receipts.

Uses synthetic token IDs; no user traffic or prompt contents are collected.
Inter-event gaps are explicitly distinguished from per-token latency.
"""

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path

import httpx


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)]


async def run_request(client, args, index, input_tokens, delay):
    await asyncio.sleep(delay)
    # Distinct prefixes prevent accidental cross-request cache reuse.
    prompt = [
        100 + (index * 97 + offset * 31) % 20000 for offset in range(input_tokens)
    ]
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "seed": 42,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    event_times, texts, usage = [], [], None
    done = False
    async with client.stream(
        "POST", args.base_url + "/v1/completions", json=payload
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                done = True
                break
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            choices = event.get("choices", [])
            if choices and (text := choices[0].get("text")):
                event_times.append(time.perf_counter() - started)
                texts.append(text)
    elapsed = time.perf_counter() - started
    if not done or not event_times or usage is None:
        raise RuntimeError("Incomplete SSE response or missing usage")
    if usage["completion_tokens"] != args.output_tokens:
        raise RuntimeError(f"Unexpected completion length: {usage}")
    ttft = event_times[0]
    # TPOT is request-level average, not an individual token latency sample.
    tpot = (
        (event_times[-1] - ttft) / (usage["completion_tokens"] - 1)
        if usage["completion_tokens"] > 1
        else 0
    )
    return {
        "index": index,
        "input_tokens": input_tokens,
        "arrival_delay_s": delay,
        "usage": usage,
        "duration_s": elapsed,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "sse_content_event_times_s": event_times,
        "max_content_event_gap_s": max(
            (b - a for a, b in zip(event_times, event_times[1:])), default=0
        ),
        "text_sha256": hashlib.sha256("".join(texts).encode()).hexdigest(),
        "slo_met": ttft <= args.ttft_slo and tpot <= args.tpot_slo,
    }


async def main(args):
    lengths = [int(v) for v in args.input_lengths.split(",")]
    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
        health = await client.get(args.base_url + "/health")
        health.raise_for_status()
        # Warm up the same shapes; exclude initialization/JIT from receipts.
        warmup_args = argparse.Namespace(
            **(vars(args) | {"output_tokens": args.warmup_output_tokens})
        )
        for index, length in enumerate(sorted(set(lengths))):
            await run_request(client, warmup_args, index + 1000, length, 0)
        all_results = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            rows = await asyncio.gather(
                *(
                    run_request(
                        client,
                        args,
                        i,
                        lengths[i % len(lengths)],
                        i * args.arrival_interval,
                    )
                    for i in range(args.requests)
                )
            )
            elapsed = time.perf_counter() - started
            all_results.append(
                {
                    "repeat": repeat,
                    "wall_s": elapsed,
                    "output_tokens_per_s": sum(
                        r["usage"]["completion_tokens"] for r in rows
                    )
                    / elapsed,
                    "request_goodput_per_s": sum(r["slo_met"] for r in rows) / elapsed,
                    "slo_attainment": sum(r["slo_met"] for r in rows) / len(rows),
                    "ttft_p50_s": percentile([r["ttft_s"] for r in rows], 0.5),
                    "ttft_p95_s": percentile([r["ttft_s"] for r in rows], 0.95),
                    "tpot_p50_s": percentile([r["tpot_s"] for r in rows], 0.5),
                    "tpot_p95_s": percentile([r["tpot_s"] for r in rows], 0.95),
                    "requests": rows,
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"label": args.label, "parameters": vars(args), "runs": all_results}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k != "requests"} for r in all_results],
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="Qwen3.5-35B-A3B")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--input-lengths", default="128,2048,512,4096")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--arrival-interval", type=float, default=0.12)
    parser.add_argument("--ttft-slo", type=float, default=2.0)
    parser.add_argument("--tpot-slo", type=float, default=0.1)
    arguments = parser.parse_args()
    if arguments.requests < 1 or arguments.repeats < 1:
        parser.error("requests and repeats must be positive")
    if min(arguments.output_tokens, arguments.warmup_output_tokens) < 2:
        parser.error("output-tokens and warmup-output-tokens must be >=2")
    if arguments.arrival_interval < 0:
        parser.error("arrival-interval must be >=0")
    asyncio.run(main(arguments))
