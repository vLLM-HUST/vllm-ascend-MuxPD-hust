"""Explicit, bounded sampler observations; never replaces a native sample.

This diagnostic adds cloning, D2H and file-writing overhead. Its timings are
unsuitable for performance comparisons. No hook is installed by importing it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def schedule_snapshot(output: Any) -> dict[str, Any]:
    cached = output.scheduled_cached_reqs
    return {
        "num_scheduled_tokens": dict(output.num_scheduled_tokens),
        "total_num_scheduled_tokens": output.total_num_scheduled_tokens,
        "new_request_ids": [r.req_id for r in output.scheduled_new_reqs],
        "cached_request_ids": list(cached.req_ids),
        "cached_num_computed_tokens": list(cached.num_computed_tokens),
        "cached_num_output_tokens": list(cached.num_output_tokens),
        "finished_request_ids": sorted(output.finished_req_ids),
        "preempted_request_ids": sorted(getattr(output, "preempted_req_ids", ()) or ()),
        "muxpd_prefill_cohort_id": getattr(output, "muxpd_prefill_cohort_id", None),
    }


def batch_snapshot(runner: Any) -> list[dict[str, Any]]:
    batch = runner.input_batch
    rows = []
    for index, request_id in enumerate(batch.req_ids):
        request = runner.requests[request_id]
        prompt = list(request.prompt_token_ids)
        prefix = list(request.output_token_ids)
        count = int(batch.num_tokens_no_spec[index])
        rows.append(
            {
                "row": index,
                "request_id": request_id,
                "prompt_token_ids": prompt,
                "prompt_sha256": token_digest(prompt),
                "output_prefix_token_ids": prefix,
                "output_prefix_sha256": token_digest(prefix),
                "output_token_index": len(prefix),
                "num_computed_tokens": int(batch.num_computed_tokens_cpu[index]),
                "num_tokens_no_spec": count,
                "input_token_ids": [int(x) for x in batch.token_ids_cpu[index, :count]],
                "kv_block_ids": [
                    [int(x) for x in blocks] for blocks in request.block_ids
                ],
            }
        )
    return rows


class NativeSamplerObserver:
    """Observe one runner through original execute/sample/bookkeeping methods."""

    def __init__(
        self,
        runner: Any,
        directory: Path,
        lane: str,
        rank: int,
        *,
        capture_steps: int = 8,
        capture_ranks: tuple[int, ...] = (0,),
    ):
        if not directory.is_absolute() or not 1 <= capture_steps <= 128:
            raise ValueError("Absolute output directory and bounded capture required")
        if getattr(runner, "use_async_scheduling", False):
            raise ValueError("Numerical observer requires synchronous scheduling")
        if getattr(runner, "speculative_config", None) is not None:
            raise ValueError("Numerical observer requires non-speculative sampling")
        if getattr(runner, "_muxpd_numerical_observer", None) is not None:
            raise ValueError("Runner already has a numerical observer")
        self.runner, self.directory, self.lane, self.rank = (
            runner,
            directory,
            lane,
            rank,
        )
        self.capture_steps, self.capture_ranks = capture_steps, capture_ranks
        self.sequence = 0
        self.pending = None
        self.schedule = None
        self.batch_descriptor = None
        directory.mkdir(parents=True, exist_ok=True)
        self.log = (directory / f"rank-{rank}-{lane}.jsonl").open("x")
        self.emit(
            {
                "event": "observer-installed",
                "pid": os.getpid(),
                "capture_steps": capture_steps,
                "capture_ranks": capture_ranks,
                "scope": (
                    "Diagnostic only: native greedy call and native bookkeeping "
                    "remain intact; added clone/copy/I/O overhead."
                ),
            }
        )
        self.install()
        runner._muxpd_numerical_observer = self

    def emit(self, row: dict[str, Any]) -> None:
        self.log.write(
            json.dumps(
                {
                    "rank": self.rank,
                    "lane": self.lane,
                    "host_monotonic_ns": time.monotonic_ns(),
                    **row,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.log.flush()

    def install(self) -> None:
        runner = self.runner
        execute, sample = runner.execute_model, runner._sample
        bookkeeping, greedy = runner._bookkeeping_sync, runner.sampler.greedy_sample

        def observed_execute(output, *args, **kwargs):
            self.sequence += 1
            self.schedule = schedule_snapshot(output)
            self.emit(
                {
                    "event": "execute-enter",
                    "sequence": self.sequence,
                    "schedule": self.schedule,
                }
            )
            result = execute(output, *args, **kwargs)
            state = getattr(runner, "execute_model_state", None)
            self.batch_descriptor = None
            if state is not None:
                if len(state) != 12:
                    raise RuntimeError("Unknown native execute_model_state contract")
                descriptor = state[-1]
                self.batch_descriptor = (
                    asdict(descriptor)
                    if is_dataclass(descriptor)
                    else {
                        name: getattr(descriptor, name, None)
                        for name in (
                            "num_tokens",
                            "num_reqs",
                            "uniform",
                            "has_lora",
                            "num_active_loras",
                        )
                    }
                )
            self.emit(
                {
                    "event": "execute-return",
                    "sequence": self.sequence,
                    "deferred_native_state": state is not None,
                    "batch_descriptor": self.batch_descriptor,
                }
            )
            return result

        def observed_sample(logits, spec_decode_metadata, *args, **kwargs):
            if self.pending is not None:
                raise RuntimeError(
                    "Previous diagnostic sample has no native bookkeeping"
                )
            metadata = runner.input_batch.sampling_metadata
            if spec_decode_metadata is not None or not metadata.all_greedy:
                raise ValueError("Numerical observer only accepts plain greedy samples")
            rows = batch_snapshot(runner)
            self.pending = {
                "rows": rows,
                "shape": list(logits.shape),
                "dtype": str(logits.dtype),
                "logits": None,
                "native_greedy_calls": 0,
            }
            try:
                return sample(logits, spec_decode_metadata, *args, **kwargs)
            except BaseException:
                self.pending = None
                raise

        def observed_greedy(logits):
            # Original implementation is invoked exactly once, first. In
            # particular its collective/reduce-sample behavior is not replaced.
            result = greedy(logits)
            if self.pending is not None:
                self.pending["native_greedy_calls"] += 1
                if self.pending["native_greedy_calls"] != 1:
                    raise RuntimeError("Expected exactly one native greedy decision")
                selected = any(
                    row["output_token_index"] < self.capture_steps
                    for row in self.pending["rows"]
                )
                if self.rank in self.capture_ranks and selected:
                    self.pending["logits"] = logits.detach().clone()
            return result

        def observed_bookkeeping(*args, **kwargs):
            # Native synchronous D2H and bookkeeping happen before diagnostic
            # copies, numpy conversion, argmax inspection or file writes.
            result = bookkeeping(*args, **kwargs)
            pending, self.pending = self.pending, None
            if pending is not None:
                if pending["native_greedy_calls"] != 1:
                    raise RuntimeError("No single native greedy call was observed")
                self.flush(pending, result)
            return result

        runner.execute_model = observed_execute
        runner._sample = observed_sample
        runner.sampler.greedy_sample = observed_greedy
        runner._bookkeeping_sync = observed_bookkeeping

    def flush(self, pending: dict[str, Any], native: Any) -> None:
        import numpy as np

        valid_tokens, req_ids, invalid = native[1], native[3], native[5]
        if list(req_ids) != [row["request_id"] for row in pending["rows"]]:
            raise RuntimeError("Sampler rows differ from native bookkeeping rows")
        raw = pending.pop("logits")
        array = raw.float().cpu().numpy() if raw is not None else None
        for row in pending["rows"]:
            row["sampled_token_ids"] = list(valid_tokens[row["row"]])
            row["discarded_by_native"] = (
                row["row"] in invalid or not row["sampled_token_ids"]
            )
            if array is None or row["output_token_index"] >= self.capture_steps:
                continue
            values = array[row["row"]]
            filename = (
                f"rank-{self.rank}-{self.lane}-step-{self.sequence:05d}"
                f"-row-{row['row']}.npy"
            )
            path = self.directory / filename
            with path.open("xb") as stream:
                np.save(stream, values, allow_pickle=False)
            encoded = path.read_bytes()
            finite = bool(np.isfinite(values).all())
            top = np.argsort(values, kind="stable")[-8:][::-1]
            row["logits"] = {
                "file": filename,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "all_finite": finite,
                "top_token_ids": [int(x) for x in top],
                "top_values": [
                    float(values[x]) if np.isfinite(values[x]) else None for x in top
                ],
                "cpu_argmax": int(np.argmax(values)),
            }
        self.emit(
            {
                "event": "native-sample-observed",
                "sequence": self.sequence,
                "schedule": self.schedule,
                "batch_descriptor": self.batch_descriptor,
                **pending,
            }
        )


def attach_observer(worker: Any, runner: Any, lane: str) -> NativeSamplerObserver:
    directory = Path(os.environ["MUXPD_NUMERIC_OUTPUT"])
    steps = int(os.environ.get("MUXPD_NUMERIC_CAPTURE_STEPS", "8"))
    ranks = tuple(
        int(rank) for rank in os.environ.get("MUXPD_NUMERIC_RANKS", "0").split(",")
    )
    return NativeSamplerObserver(
        runner, directory, lane, worker.rank, capture_steps=steps, capture_ranks=ranks
    )


def bounded_constant(value: Any, *, depth: int = 0) -> Any:
    """JSON metadata for compiler constants; never read tensor contents."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return (
            value
            if value not in (float("inf"), float("-inf")) and value == value
            else str(value)
        )
    if depth >= 8:
        return {
            "type": type(value).__module__ + "." + type(value).__qualname__,
            "truncated": True,
        }
    if isinstance(value, (list, tuple)):
        return [bounded_constant(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            str(key): bounded_constant(item, depth=depth + 1)
            for key, item in value.items()
        }
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": type(value).__module__ + "." + type(value).__qualname__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(getattr(value, "device", "unknown")),
            "contents_observed": False,
        }
    return {
        "type": type(value).__module__ + "." + type(value).__qualname__,
        "name": getattr(value, "__qualname__", None),
        "repr": str(value) if type(value).__module__ == "torch" else None,
    }


def compiler_constants_snapshot(worker: Any) -> None:
    import torch
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    path = (
        Path(os.environ["MUXPD_NUMERIC_OUTPUT"])
        / f"rank-{worker.rank}-compiler-constants.json"
    )
    contents = {
        "rank": worker.rank,
        "pid": os.getpid(),
        "scope": (
            "Post-native-warmup Triton constant argument metadata; tensor "
            "contents are never read, and this alone does not establish "
            "compiler equivalence. Environment values are observed, not proof "
            "that every collective/operator is deterministic."
        ),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "HCCL_DETERMINISTIC",
                "VLLM_BATCH_INVARIANT",
                "ASCEND_LAUNCH_BLOCKING",
                "ASCEND_RT_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "constant_args": bounded_constant(kernel_side_table.constant_args),
    }
    with path.open("x") as stream:
        json.dump(contents, stream, indent=2, allow_nan=False)
        stream.write("\n")
