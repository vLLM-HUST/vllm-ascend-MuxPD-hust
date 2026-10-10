"""Exercise a serving endpoint using only fixed, synthetic requests.

Save response contents, usage, SSE receipts, and timings even when checks fail.
Client disconnect recovery checks service liveness, not internal cache reclamation.
Requires httpx from the serving environment.
"""

import argparse
import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def chat_payload(model, prompt, *, stream=False, max_tokens=64):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def validate_arithmetic(response):
    content = response["choices"][0]["message"].get("content") or ""
    if not re.fullmatch(r"\s*42[.。!]?\s*", content):
        raise ValueError(f"Expected integer 42, received {content!r}")
    if response["choices"][0].get("finish_reason") != "stop":
        raise ValueError("Arithmetic response did not finish normally")
    if not response.get("usage", {}).get("completion_tokens", 0):
        raise ValueError("Arithmetic response is missing completion usage")


class SmokeClient:
    def __init__(self, client, base_url, model):
        self.client = client
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model

    async def json_request(self, name, path, payload=None, timeout=None):
        record = {"name": name, "started_at_utc": timestamp(), "passed": False}
        started = time.perf_counter()
        try:
            kwargs = {} if timeout is None else {"timeout": timeout}
            response = await self.client.request(
                "GET" if payload is None else "POST",
                self.base_url + path,
                json=payload,
                **kwargs,
            )
            record["http_status"] = response.status_code
            try:
                record["response"] = response.json()
            except ValueError:
                record["response"] = response.text
            response.raise_for_status()
            if payload is not None:
                validate_arithmetic(record["response"])
                record["usage"] = record["response"].get("usage")
            record["passed"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["duration_s"] = time.perf_counter() - started
        return record

    async def stream_request(
        self, name, path, payload, *, cancel_after_events=None,
        expect_chinese=False, expected_prompt_tokens=None,
    ):
        record = {
            "name": name, "started_at_utc": timestamp(), "passed": False,
            "events": [], "content_event_times_s": [], "done": False,
            "finish_reasons": [], "usage": None,
        }
        started = time.perf_counter()
        content = []
        try:
            async with self.client.stream(
                "POST", self.base_url + path, json=payload,
            ) as response:
                record["http_status"] = response.status_code
                if response.is_error:
                    record["error_response"] = (
                        await response.aread()
                    ).decode(errors="replace")
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    elapsed = time.perf_counter() - started
                    if data == "[DONE]":
                        record["done"] = True
                        record["done_at_s"] = elapsed
                        break
                    if not data:
                        continue
                    event = json.loads(data)
                    record["events"].append({"elapsed_s": elapsed, "data": event})
                    if event.get("error"):
                        raise ValueError(f"SSE error: {event['error']}")
                    record["usage"] = event.get("usage") or record["usage"]
                    for choice in event.get("choices", []):
                        text = choice.get("text")
                        if text is None:
                            text = choice.get("delta", {}).get("content")
                        if text:
                            content.append(text)
                            record["content_event_times_s"].append(elapsed)
                        reason = choice.get("finish_reason")
                        if reason:
                            record["finish_reasons"].append(reason)
                    if (
                        cancel_after_events is not None
                        and len(content) >= cancel_after_events
                    ):
                        if record["finish_reasons"]:
                            raise ValueError("Generation finished before cancellation")
                        record["cancellation_requested_at_s"] = elapsed
                        break
            record["response_text"] = "".join(content)
            if cancel_after_events is not None:
                if "cancellation_requested_at_s" not in record:
                    raise ValueError("No content arrived before cancellation")
                record["stream_closed_at_s"] = time.perf_counter() - started
            else:
                if not record["done"] or not record["finish_reasons"]:
                    raise ValueError("SSE response was not completed")
                if not record["usage"] or not content:
                    raise ValueError("SSE response is missing content or usage")
                if expect_chinese and not re.search(
                    r"[\u3400-\u9fff]", record["response_text"],
                ):
                    raise ValueError("Expected Chinese content in response")
                if expected_prompt_tokens is not None:
                    actual = record["usage"]["prompt_tokens"]
                    if actual != expected_prompt_tokens:
                        raise ValueError(
                            f"Expected {expected_prompt_tokens} prompt tokens, "
                            f"received {actual}"
                        )
                    if record["usage"]["completion_tokens"] != payload["max_tokens"]:
                        raise ValueError("Forced-length completion was incomplete")
            record["passed"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["response_text"] = "".join(content)
        record["duration_s"] = time.perf_counter() - started
        times = record["content_event_times_s"]
        record["ttft_s"] = times[0] if times else None
        record["max_content_event_gap_s"] = max(
            (later - earlier for earlier, later in zip(times, times[1:])),
            default=None,
        )
        return record


async def run_suite(args, *, transport=None):
    report = {
        "schema_version": 1, "started_at_utc": timestamp(),
        "model": args.model, "base_url": args.base_url,
        "synthetic_inputs_only": True,
        "timing_note": "SSE event gaps are not individual-token TBT samples.",
        "tests": [],
    }
    started = time.perf_counter()
    async with httpx.AsyncClient(
        timeout=args.timeout, trust_env=False, transport=transport,
    ) as client:
        api = SmokeClient(client, args.base_url, args.model)
        health = await api.json_request("initial_health", "/health")
        report["tests"].append(health)
        if health["passed"]:
            arithmetic = chat_payload(
                args.model, "What is 19 + 23? Reply with only the integer.",
            )
            report["tests"].append(await api.json_request(
                "greedy_arithmetic", "/v1/chat/completions", arithmetic,
            ))
            prompts = [
                "请用中文一句话解释什么是计算机缓存。",
                "请用中文一句话说明为什么天空通常是蓝色。",
                "请用中文一句话介绍长江。",
            ]
            concurrent_started = time.perf_counter()
            rows = await asyncio.gather(*(
                api.stream_request(
                    f"concurrent_chinese_{index}", "/v1/chat/completions",
                    chat_payload(args.model, prompt, stream=True, max_tokens=96),
                    expect_chinese=True,
                ) for index, prompt in enumerate(prompts)
            ))
            report["concurrent_chinese_wall_s"] = (
                time.perf_counter() - concurrent_started
            )
            report["tests"].extend(rows)

            # Exact token IDs avoid confusing character count with token count.
            token_ids = [100 + (index * 31) % 1000 for index in range(2305)]
            long_payload = {
                "model": args.model, "prompt": token_ids,
                "temperature": 0, "seed": 42, "max_tokens": 32,
                "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True},
            }
            long_result = await api.stream_request(
                "long_prompt_2305_tokens", "/v1/completions", long_payload,
                expected_prompt_tokens=len(token_ids),
            )
            long_result["input_token_count"] = len(token_ids)
            long_result["input_sha256"] = hashlib.sha256(
                json.dumps(token_ids).encode()
            ).hexdigest()
            report["tests"].append(long_result)

            cancel_payload = chat_payload(
                args.model, "请连续写一篇很长的中文科幻故事。",
                stream=True, max_tokens=2048,
            )
            cancel_payload["ignore_eos"] = True
            report["tests"].append(await api.stream_request(
                "cancel_long_generation", "/v1/chat/completions",
                cancel_payload, cancel_after_events=1,
            ))
            report["tests"].append(await api.json_request(
                "health_after_cancel", "/health", timeout=args.recovery_timeout,
            ))
            report["tests"].append(await api.json_request(
                "greedy_after_cancel", "/v1/chat/completions", arithmetic,
                timeout=args.recovery_timeout,
            ))
        else:
            report["skipped_reason"] = "Initial health check failed"
    report["duration_s"] = time.perf_counter() - started
    report["passed"] = all(row["passed"] for row in report["tests"])
    report["finished_at_utc"] = timestamp()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen3.5-35B-A3B")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--recovery-timeout", type=float, default=60)
    args = parser.parse_args()
    if args.timeout <= 0 or args.recovery_timeout <= 0:
        parser.error("timeouts must be positive")
    report = asyncio.run(run_suite(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "passed": report["passed"], "output": str(output),
        "tests": [{"name": row["name"], "passed": row["passed"]}
                  for row in report["tests"]],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
