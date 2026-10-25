"""Bounded fixed-token diagnostic using a step-driven in-process coordinator.

Plan creation imports no inference runtime. The explicit run subcommand owns one
LLM instance and shuts it down. It does not start an HTTP service. Capture adds
overhead and supplies no latency/throughput evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    from .probe_prefill_group_outputs import SHORT_PROMPT
except ImportError:
    from probe_prefill_group_outputs import SHORT_PROMPT

MAX_ENGINE_STEPS = 500


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def plan():
    phases = []
    cohorts = {
        "single-p128": [SHORT_PROMPT],
        "four-p128": [
            [100 + (i * 31 + lane * 17) % 900 for i in range(128)] for lane in range(4)
        ],
    }
    for name, prompts in cohorts.items():
        for repeat in range(3):
            phases.append(
                {
                    "name": name,
                    "warm_repeat": repeat,
                    "prompts": prompts,
                    "max_tokens": 8,
                }
            )
    value = {
        "schema": "muxpd-fixed-token-numerics-plan-v1",
        "phases": phases,
        "requests": 15,
        "output_tokens_per_request": 8,
        "sampling": {
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
            "detokenize": False,
        },
        "admission": (
            "enqueue every phase request into InprocClient before any step; "
            "fully drain between phases"
        ),
        "scope": (
            "Fixed initial P128 and four-lane P128 cohorts; warm repeats within "
            "one process. Fresh process controls require separate invocations/"
            "output directories. Actual worker schedule and sampler row "
            "snapshots remain authoritative."
        ),
        "maximum_engine_steps": MAX_ENGINE_STEPS,
    }
    return {**value, "plan_sha256": digest(value)}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def runtime_profile(role, engine_args):
    """Bind the diagnostic label to the actual P and global compiler settings."""
    policy = engine_args.get("batch_admission_policy_config", {})
    grouped = policy.get("compile_prefill_groups", False)
    if type(grouped) is not bool:
        raise ValueError("compile_prefill_groups must be an explicit boolean")
    if grouped != (role == "group"):
        raise ValueError("Diagnostic role and compile_prefill_groups disagree")
    backend = engine_args.get("additional_config", {}).get(
        "ascend_compilation_config", {}
    )
    if role == "group" and any(
        backend.get(name) is not False
        for name in ("enable_npugraph_ex", "enable_static_kernel")
    ):
        raise ValueError("Group diagnostic requires explicit fusion backend settings")
    return {
        "role": role,
        "compile_prefill_groups": grouped,
        "ascend_compilation_config": backend,
        "comparison_scope": (
            "Compare raw/group only with identical source, observer, engine "
            "settings and explicit global compiler backend except the P-group flag."
        ),
    }


def run(args):
    frozen = json.loads(args.plan.read_text())
    if frozen != plan() or frozen["plan_sha256"] != args.plan_sha256:
        raise ValueError("Plan differs from the frozen 15-request contract")
    engine_args = json.loads(args.engine_args.read_text())
    profile = runtime_profile(args.role, engine_args)
    required = {
        "tensor_parallel_size": 4,
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "max_num_seqs": 8,
        "max_model_len": 16384,
        "max_num_batched_tokens": 16384,
    }
    for name, value in required.items():
        if engine_args.get(name) != value:
            raise ValueError(f"Engine argument {name} must be {value!r}")
    if engine_args.get("speculative_config") is not None:
        raise ValueError("Speculation is outside this diagnostic")
    worker = (
        "NumericalNativeNPUWorker"
        if args.role == "native"
        else "NumericalMuxPDNPUWorker"
    )
    if "worker_cls" in engine_args:
        raise ValueError("The diagnostic selects its worker class explicitly")
    engine_args["worker_cls"] = "vllm_muxpd.numerical_workers." + worker
    if (
        args.role in ("raw", "group")
        and engine_args.get("enable_chunked_prefill") is not False
    ):
        raise ValueError("MuxPD diagnostic requires chunked prefill disabled")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["MUXPD_NUMERIC_OUTPUT"] = str(args.output_dir / "worker-observations")
    os.environ["MUXPD_NUMERIC_CAPTURE_STEPS"] = "8"
    os.environ["MUXPD_NUMERIC_RANKS"] = "0"
    # The background EngineCore process may run while prompts are still being
    # added. This setting keeps coordinator admission under the explicit step
    # loop; TP workers themselves still use the configured MP executor.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    report = {
        "schema": "muxpd-fixed-token-numerics-run-v1",
        "role": args.role,
        "runtime_profile": profile,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "plan": frozen,
        "engine_args": engine_args,
        "phases": [],
        "errors": [],
        "passed": False,
        "scope": (
            "Collection/cleanup only; not numerical equivalence or "
            "performance acceptance."
        ),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(args.output_dir / "run.json", report)
    llm = None
    try:
        from vllm import LLM, SamplingParams
        from vllm.v1.engine.core_client import InprocClient

        llm = LLM(**engine_args)
        if not isinstance(llm.llm_engine.engine_core, InprocClient):
            raise RuntimeError(
                "Actual engine coordinator is not step-driven InprocClient"
            )
        report["engine_coordinator"] = type(llm.llm_engine.engine_core).__name__
        global_step = 0
        for phase in frozen["phases"]:
            inputs = [{"prompt_token_ids": tokens} for tokens in phase["prompts"]]
            params = SamplingParams(
                max_tokens=phase["max_tokens"], **frozen["sampling"]
            )
            ids = llm.enqueue(inputs, params, use_tqdm=False)
            # enqueue returns randomized INTERNAL IDs; RequestOutput exposes
            # the original EXTERNAL IDs. Bind through the real output states,
            # before completion removes them. Never infer a mapping by splitting
            # IDs or globally disable native request-ID randomization.
            external_ids = [
                llm.llm_engine.output_processor.request_states[item].external_req_id
                for item in ids
            ]
            observed = {
                "name": phase["name"],
                "warm_repeat": phase["warm_repeat"],
                "request_ids": ids,
                "external_request_ids": external_ids,
                "internal_to_external": dict(zip(ids, external_ids)),
                "admitted_before_step": global_step,
                "prompts": phase["prompts"],
                "steps": [],
                "outputs": [],
            }
            report["phases"].append(observed)
            while llm.llm_engine.has_unfinished_requests():
                if global_step >= MAX_ENGINE_STEPS:
                    raise RuntimeError("Diagnostic engine exceeded 500 total steps")
                global_step += 1
                before = time.monotonic_ns()
                outputs = llm.llm_engine.step()
                rows = []
                for output in outputs:
                    row = {
                        "request_id": output.request_id,
                        "finished": output.finished,
                        "prompt_token_ids": output.prompt_token_ids,
                        "outputs": [
                            {
                                "token_ids": list(item.token_ids),
                                "finish_reason": item.finish_reason,
                                "stop_reason": item.stop_reason,
                            }
                            for item in output.outputs
                        ],
                    }
                    rows.append(row)
                    if output.finished:
                        observed["outputs"].append(row)
                observed["steps"].append(
                    {
                        "global_step": global_step,
                        "before_monotonic_ns": before,
                        "after_monotonic_ns": time.monotonic_ns(),
                        "outputs": rows,
                    }
                )
            if {row["request_id"] for row in observed["outputs"]} != set(external_ids):
                raise RuntimeError("Finished request set differs from admitted phase")
            if any(
                len(row["outputs"]) != 1 or len(row["outputs"][0]["token_ids"]) != 8
                for row in observed["outputs"]
            ):
                raise RuntimeError("A request did not return exactly eight token IDs")
            write_json(args.output_dir / "run.json", report)
        report["collection_complete"] = True
    except BaseException:
        report["errors"].append(traceback.format_exc())
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
                report["shutdown_returned"] = True
            except BaseException:
                report["errors"].append(traceback.format_exc())
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["passed"] = bool(
            report.get("collection_complete")
            and report.get("shutdown_returned")
            and not report["errors"]
        )
        write_json(args.output_dir / "run.json", report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "output": str(args.output_dir),
                "errors": report["errors"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("plan")
    create.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--plan-sha256", required=True)
    execute.add_argument("--engine-args", type=Path, required=True)
    execute.add_argument("--role", choices=("native", "raw", "group"), required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        with args.output.open("x") as stream:
            json.dump(plan(), stream, indent=2)
            stream.write("\n")
        print(plan()["plan_sha256"])
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
