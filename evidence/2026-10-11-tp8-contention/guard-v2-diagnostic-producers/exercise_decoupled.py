"""Start a long prefill during decode; optionally cancel it before first token.

Pair this client receipt with a decoupled scheduler trace to prove progress and
deferred cancellation. HTTP timings alone cannot prove internal KV ownership.
"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx


def positive_finite_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


async def generate(client, base, model, rid, tokens, output_tokens, first_token):
    started = time.monotonic()
    events = []
    usage = None
    done = False
    async with client.stream(
        "POST",
        base + "/v1/completions",
        json={
            "model": model,
            "prompt": tokens,
            "max_tokens": output_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        headers={"X-Request-Id": rid},
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
            if event.get("choices") and event["choices"][0].get("text"):
                events.append(time.monotonic())
                first_token.set()
    if not done or usage is None or usage["completion_tokens"] != output_tokens:
        raise RuntimeError(f"Incomplete request {rid}: {usage}")
    return {
        "request_id": rid,
        "start_monotonic": started,
        "content_events_monotonic": events,
        "usage": usage,
        "duration_s": time.monotonic() - started,
    }


async def main(args):
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        first_decode = asyncio.Event()
        decode_task = asyncio.create_task(
            generate(
                client,
                args.base_url,
                args.model,
                "muxpd-decode-probe",
                [100 + i % 200 for i in range(128)],
                64,
                first_decode,
            )
        )
        await asyncio.wait_for(first_decode.wait(), 120)
        prefill_started = time.monotonic()
        prefill_task = asyncio.create_task(
            generate(
                client,
                args.base_url,
                args.model,
                "muxpd-prefill-probe",
                [200 + i % 500 for i in range(12000)],
                8,
                asyncio.Event(),
            )
        )
        cancelled_at = None
        if args.cancel_prefill:
            await asyncio.sleep(args.cancel_after)
            cancelled_at = time.monotonic()
            prefill_task.cancel()
            try:
                await prefill_task
            except asyncio.CancelledError:
                pass
            prefill = None
        else:
            prefill = await prefill_task
        decode = await decode_task
        health = await client.get(args.base_url + "/health")
        health.raise_for_status()
    report = {
        "passed": True,
        "cancel_delay_s": args.cancel_after if args.cancel_prefill else None,
        "cancelled_at_monotonic": cancelled_at,
        "prefill_started_monotonic": prefill_started,
        "decode": decode,
        "prefill": prefill,
        "decode_content_events_before_prefill_first_token": (
            sum(
                prefill_started < timestamp < prefill["content_events_monotonic"][0]
                for timestamp in decode["content_events_monotonic"]
            )
            if prefill is not None
            else None
        ),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("decode", "prefill")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="Qwen3.5-35B-A3B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cancel-prefill", action="store_true")
    parser.add_argument(
        "--cancel-after",
        type=positive_finite_float,
        default=0.4,
        help=(
            "seconds after starting prefill to cancel with --cancel-prefill "
            "(default: 0.4)"
        ),
    )
    asyncio.run(main(parser.parse_args()))
