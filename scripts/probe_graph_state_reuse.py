"""Serial token-ID diagnostics before and after a synthetic long prefill.

Run independently on matched native eager, native graph, and plugin graph
services. This is a correctness diagnostic, not a benchmark or proof that a
device graph ran. It does not alter the existing graph-serving baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

GENERATION = {
    "temperature": 0,
    "seed": 42,
    "ignore_eos": True,
    "add_special_tokens": False,
    "skip_special_tokens": False,
    "stream": False,
    "return_token_ids": True,
}


def cases():
    # IDs are from the local Qwen3.5 tokenizer: Hello=9419, Hi=12675.
    # Keep the original synthetic ID 100 to distinguish text-decoding effects.
    return [
        {"name": f"{name}_max{length}", "prompt": prompt, "max_tokens": length}
        for name, prompt in (
            ("original_one_token", [100]),
            ("hello_one_token", [9419]),
            ("hello_hi_two_tokens", [9419, 12675]),
        )
        for length in (1, 16)
    ]


def request_plan(args):
    """Every target case gets its own before -> background -> after bracket."""
    long_prompt = [9419, 12675] * ((args.long_prompt_tokens + 1) // 2)
    long_prompt = long_prompt[: args.long_prompt_tokens]
    for case in cases():
        for stage in ("before", "background", "after"):
            for repeat in range(1 if stage == "background" else args.repeats):
                background = stage == "background"
                yield {
                    "case": case["name"],
                    "stage": stage,
                    "repeat": repeat,
                    "payload": {
                        "model": args.model,
                        "prompt": long_prompt if background else case["prompt"],
                        "max_tokens": (
                            args.long_max_tokens if background else case["max_tokens"]
                        ),
                        **GENERATION,
                    },
                }


def validate_response(data, payload):
    """Require token IDs so invalid UTF-8 text cannot mask a token difference."""
    if not isinstance(data, dict) or "error" in data:
        raise ValueError("Completion returned an error or non-object response")
    if not isinstance(data.get("id"), str) or not data["id"]:
        raise ValueError("Missing response id")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one completion choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("index") != 0:
        raise ValueError("Invalid completion choice")
    tokens = choice.get("token_ids")
    if (
        not isinstance(tokens, list)
        or len(tokens) != payload["max_tokens"]
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError("Missing or invalid generated token_ids")
    if choice.get("prompt_token_ids") != payload["prompt"]:
        raise ValueError("Returned prompt_token_ids differ from the requested IDs")
    if choice.get("finish_reason") != "length":
        raise ValueError("ignore_eos probe must finish with reason=length")
    if not isinstance(choice.get("text"), str):
        raise ValueError("Missing completion text")
    usage = data.get("usage")
    expected = {
        "prompt_tokens": len(payload["prompt"]),
        "completion_tokens": payload["max_tokens"],
        "total_tokens": len(payload["prompt"]) + payload["max_tokens"],
    }
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("Completion usage differs from the requested token counts")
    return {
        "token_ids": tokens,
        "text": choice["text"],
        "usage": usage,
        "response_id": data["id"],
        "finish_reason": choice["finish_reason"],
    }


def first_difference(left, right):
    """Indices refer to generated tokens, starting at zero; None means equal."""
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a != b:
            return {"token_index": index, "left_token_id": a, "right_token_id": b}
    return None


def repeat_summary(rows, expected):
    sequences = [row["token_ids"] for row in rows if row["transport_passed"]]
    complete = len(sequences) == expected
    difference = next(
        (
            {"reference_repeat": 0, "repeat": index, **delta}
            for index, tokens in enumerate(sequences[1:], start=1)
            if (delta := first_difference(sequences[0], tokens)) is not None
        ),
        None,
    )
    return {
        "samples": len(sequences),
        "complete": complete,
        "distinct_token_sequences": len({tuple(tokens) for tokens in sequences}),
        "stable": difference is None if complete else None,
        "first_difference_from_repeat_zero": difference,
    }


def summarize(rows, repeats):
    result = []
    for case in cases():
        selected = [row for row in rows if row["case"] == case["name"]]
        before = [row for row in selected if row["stage"] == "before"]
        after = [row for row in selected if row["stage"] == "after"]
        first_before = next((r for r in before if r["transport_passed"]), None)
        first_after = next((r for r in after if r["transport_passed"]), None)
        comparable = first_before is not None and first_after is not None
        delta = (
            first_difference(first_before["token_ids"], first_after["token_ids"])
            if comparable else None
        )
        result.append({
            "case": case["name"],
            "before": repeat_summary(before, repeats),
            "after": repeat_summary(after, repeats),
            "background_completed": any(
                row["stage"] == "background" and row["transport_passed"]
                for row in selected
            ),
            "first_before_matches_first_after": delta is None if comparable else None,
            "first_before_vs_first_after_difference": delta,
        })
    return result


def run_probe(args, client, checkpoint):
    report = {
        "schema_version": 1,
        "parameters": vars(args).copy(),
        "generation": GENERATION.copy(),
        "cases": cases(),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "transport_passed": False,
        "expected_requests": len(cases()) * (2 * args.repeats + 1),
        "requests": [],
        "note": (
            "Synthetic prompts only; requests are strictly serial. Each case has "
            "before repetitions, one long P followed by generated D tokens, and "
            "after repetitions. Before is not a fresh-process/cold-cache claim. "
            "Token differences do not establish a graph or cache bug. Confirm "
            "actual dispatch and compare matched eager/graph services. On any "
            "failed request the probe stops, so a timed-out server request cannot "
            "overlap a later request from this client."
        ),
    }
    checkpoint(report)
    for index, planned in enumerate(request_plan(args)):
        row = {
            **planned,
            "sequence_index": index,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "started_monotonic_s": time.perf_counter(),
            "transport_passed": False,
        }
        try:
            response = client.post(
                args.base_url.rstrip("/") + "/v1/completions",
                json=row["payload"],
                timeout=args.timeout,
            )
            row["http_status"] = response.status_code
            try:
                row["response"] = response.json()
            except ValueError:
                row["response_text"] = response.text
                raise ValueError("Completion response is not JSON") from None
            response.raise_for_status()
            row.update(validate_response(row["response"], row["payload"]))
            row["transport_passed"] = True
        except (httpx.HTTPError, ValueError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        finally:
            row["finished_monotonic_s"] = time.perf_counter()
            row["elapsed_s"] = (
                row["finished_monotonic_s"] - row["started_monotonic_s"]
            )
            report["requests"].append(row)
            report["summary"] = summarize(report["requests"], args.repeats)
            checkpoint(report)
        if not row["transport_passed"]:
            break
    report["complete"] = (
        len(report["requests"]) == report["expected_requests"]
        and all(row["transport_passed"] for row in report["requests"])
    )
    report["transport_passed"] = report["complete"]
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint(report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen3.5-35B-A3B")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--long-prompt-tokens", type=int, default=4096)
    parser.add_argument("--long-max-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 for repeat stability")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.long_prompt_tokens < 2 or args.long_max_tokens < 1:
        parser.error("long prompt needs at least 2 tokens and generation at least 1")
    return args


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(report):
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    with httpx.Client(timeout=args.timeout, trust_env=False) as client:
        report = run_probe(args, client, checkpoint)
    print(json.dumps({
        "output": str(output),
        "complete": report["complete"],
        "transport_passed": report["transport_passed"],
        "summary": report["summary"],
    }, indent=2))
    return int(not report["transport_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
