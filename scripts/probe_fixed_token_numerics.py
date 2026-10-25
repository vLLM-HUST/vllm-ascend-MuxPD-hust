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


def plan(staggered=False, *, composite=False):
    if composite:
        if staggered:
            raise ValueError("Choose composite or staggered, not both")
        normal = plan()
        seed_first = plan(staggered=True)
        value = {
            **{key: val for key, val in normal.items() if key != "plan_sha256"},
            "schema": "muxpd-composite-token-numerics-plan-v1",
            "requests": 30,
            "source_plan_sha256": [normal["plan_sha256"], seed_first["plan_sha256"]],
            "phases": [
                {**phase, "name": family + "-" + phase["name"]}
                for family, source in (("normal", normal), ("seed-first", seed_first))
                for phase in source["phases"]
            ],
            "admission": (
                "Run the complete normal family followed by the complete "
                "seed-first family in one process, fully draining each phase. "
                "Only seed-first four-lane phases inject peers after exactly "
                "one live scheduler output token."
            ),
            "scope": (
                "Thirty requests with three warm repeats per distinct family "
                "and shape. Native enables chunked prefill for the mixed path; "
                "raw/group disable it. Family names prevent pooling repetitions "
                "across arrival conditions. No fresh-process equivalence or "
                "performance claim is implied by sharing one startup."
            ),
        }
        return {**value, "plan_sha256": digest(value)}
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
    if staggered:
        value["schema"] = "muxpd-staggered-token-numerics-plan-v1"
        value["admission"] = (
            "single-P128 phases fully drain; four-lane phases first enqueue "
            "lane0, then enqueue lanes1-3 immediately after an explicit engine "
            "step leaves the live scheduler seed request with exactly one "
            "output token. Fully drain before the next phase."
        )
        value["scope"] = (
            "Bounded historical D1/P384 reproduction; identical token prompts "
            "and seed-output-count admission gate across roles. Native uses "
            "chunked prefill enabled for its mixed path; raw/group use complete "
            "P with chunked prefill disabled. Actual worker batches differ by "
            "design and must remain distinguished in logit comparisons."
        )
        for phase in value["phases"]:
            if phase["name"] == "four-p128":
                phase["name"] = "four-p128-staggered"
                phase["admit_remaining_after_seed_tokens"] = 1
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


def enqueue_bound(llm, inputs, params):
    """Capture authoritative internal/external IDs before native completion."""
    ids = llm.enqueue(inputs, params, use_tqdm=False)
    external = [
        llm.llm_engine.output_processor.request_states[item].external_req_id
        for item in ids
    ]
    return ids, external


def run(args):
    frozen = json.loads(args.plan.read_text())
    staggered = frozen.get("schema") == "muxpd-staggered-token-numerics-plan-v1"
    composite = frozen.get("schema") == "muxpd-composite-token-numerics-plan-v1"
    if (
        frozen != plan(staggered, composite=composite)
        or frozen["plan_sha256"] != args.plan_sha256
    ):
        raise ValueError("Plan differs from its frozen diagnostic contract")
    engine_args = json.loads(args.engine_args.read_text())
    profile = runtime_profile(args.role, engine_args)
    if (
        (staggered or composite)
        and args.role == "native"
        and engine_args.get("enable_chunked_prefill") is not True
    ):
        raise ValueError(
            "Staggered/composite native control requires chunked prefill enabled"
        )
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
        "vllm_ascend.worker.worker.NPUWorker"
        if args.role == "native"
        else "vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker"
    )
    if "worker_cls" in engine_args or "worker_extension_cls" in engine_args:
        raise ValueError("The diagnostic selects its worker and extension explicitly")
    engine_args["worker_cls"] = worker
    engine_args["worker_extension_cls"] = (
        "vllm_muxpd.numerical_workers.NumericalObserverExtension"
    )
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
        installed = llm.collective_rpc("muxpd_install_numerical_observers", timeout=30)
        report["observer_installation"] = installed
        if (
            len(installed) != 4
            or {item.get("rank") for item in installed} != set(range(4))
            or any(
                item.get("installed") is not True
                or item.get("worker_selector") != worker
                for item in installed
            )
        ):
            raise RuntimeError("Observer installation did not confirm all TP4 workers")
        write_json(args.output_dir / "run.json", report)
        global_step = 0
        for phase in frozen["phases"]:
            inputs = [{"prompt_token_ids": tokens} for tokens in phase["prompts"]]
            params = SamplingParams(
                max_tokens=phase["max_tokens"], **frozen["sampling"]
            )
            gate = phase.get("admit_remaining_after_seed_tokens")
            initial = inputs[:1] if gate is not None else inputs
            ids, external_ids = enqueue_bound(llm, initial, params)
            observed = {
                "name": phase["name"],
                "warm_repeat": phase["warm_repeat"],
                "request_ids": ids,
                "external_request_ids": external_ids,
                "internal_to_external": dict(zip(ids, external_ids)),
                "admissions": [
                    {
                        "after_step": global_step,
                        "trigger": "phase-start",
                        "request_ids": ids.copy(),
                        "external_request_ids": external_ids.copy(),
                    }
                ],
                "initial_admission_before_step": global_step,
                "initial_request_count": len(initial),
                "staggered_gate_output_tokens": gate,
                "prompts": phase["prompts"],
                "steps": [],
                "outputs": [],
            }
            report["phases"].append(observed)
            remaining = inputs[1:] if gate is not None else []
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
                if remaining:
                    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
                    seed = scheduler.requests.get(ids[0])
                    if seed is None:
                        raise RuntimeError(
                            "Seed retired before staggered admission gate"
                        )
                    count = int(seed.num_output_tokens)
                    if count > gate:
                        raise RuntimeError(
                            "Seed passed the exact one-token admission gate"
                        )
                    if count == gate:
                        more_ids, more_external = enqueue_bound(llm, remaining, params)
                        observed["admissions"].append(
                            {
                                "after_step": global_step,
                                "trigger": "live-seed-output-count",
                                "seed_request_id": ids[0],
                                "seed_num_output_tokens": count,
                                "seed_live_in_scheduler": True,
                                "request_ids": more_ids.copy(),
                                "external_request_ids": more_external.copy(),
                            }
                        )
                        ids.extend(more_ids)
                        external_ids.extend(more_external)
                        observed["internal_to_external"].update(
                            zip(more_ids, more_external)
                        )
                        remaining = []
            if remaining:
                raise RuntimeError("Phase ended before all requests were admitted")
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
    family = create.add_mutually_exclusive_group()
    family.add_argument("--staggered", action="store_true")
    family.add_argument("--composite", action="store_true")
    execute = commands.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--plan-sha256", required=True)
    execute.add_argument("--engine-args", type=Path, required=True)
    execute.add_argument("--role", choices=("native", "raw", "group"), required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        with args.output.open("x") as stream:
            json.dump(plan(args.staggered, composite=args.composite), stream, indent=2)
            stream.write("\n")
        print(plan(args.staggered, composite=args.composite)["plan_sha256"])
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
