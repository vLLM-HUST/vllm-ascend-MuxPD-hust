#!/usr/bin/env python3
"""Bounded graph-replay and compiled-region evidence; no torch/device access."""

import ast
import gc
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TARGET = "AscendCL@aclmdlRIExecuteAsync"
COMPILE_MARKER = re.compile(r"compil|torchdynamo|inductor|aot|(?:^|_)fx(?:_|$)", re.I)
FX_MARKER = "npu_fx_compiler inference"
COMPILER_SOURCE = Path("/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/dynamo/npugraph_ex/npu_fx_compiler.py")
MODEL_OPS = (
    "vllm::moe_forward_shared", "vllm::qwen_gdn_attention_core",
    "vllm::unified_attention_with_output", "vllm::unquantized_gemm",
)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ns(value):
    return int((Decimal(str(value)) * 1000).to_integral_value())


def extract(rank, scheduler_steps):
    paths = list((OUT / "graph-compiled-profile").glob(
        f"*rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json"
    ))
    assert len(paths) == 1, paths
    path = paths[0]
    with path.open() as handle:
        data = json.load(handle)
    events = data["traceEvents"] if isinstance(data, dict) else data
    processes = {
        event["pid"]: event.get("args", {}).get("name", "")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    labels = {
        event["pid"]: event.get("args", {}).get("labels", event.get("args", {}).get("name", ""))
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_labels"
    }
    selected, compilation, boundaries = [], [], []
    cpu_names = Counter()
    kernel_names = Counter()
    for event in events:
        if event.get("ph") != "X":
            continue
        name, process = event.get("name", ""), processes.get(event["pid"], "")
        if name == TARGET:
            assert ns(event.get("dur", 0)) > 0
            selected.append(event)
        if process == "Python":
            if event.get("cat") == "cpu_op":
                cpu_names[name] += 1
        # The real FX entry is a cpu_op range, not a Python-function category.
        # Inspect all host tracks and match compiler and underscore-delimited fx.
        if process not in ("Ascend Hardware", "Task Scheduler", "Communication", "Overlap Analysis") and COMPILE_MARKER.search(name):
            compilation.append(event)
        if process in ("Ascend Hardware", "Task Scheduler"):
            if name in ("PROFILING_ENABLE", "PROFILING_DISABLE"):
                boundaries.append(event)
            if event.get("args", {}).get("Task Type") in (
                "AI_CORE", "AI_VECTOR_CORE", "AI_VECTOR", "MIX_AIC", "MIX_AIV"
            ):
                kernel_names[name] += 1
    assert selected
    info = list(path.parent.parent.glob("profiler_info*.json"))
    assert len(info) == 1
    profiler_info = json.loads(info[0].read_text())
    end = profiler_info["end_info"]
    clock_offset = end["collectionTimeEnd"] - end["MonotonicTimeEnd"]
    fx_evidence = []
    for marker in compilation:
        if marker["name"] != FX_MARKER:
            continue
        begin, finish = ns(marker["ts"]), ns(marker["ts"]) + ns(marker["dur"])
        mono_begin, mono_end = begin - clock_offset, finish - clock_offset
        contained = [event for event in events if
                     event.get("ph") == "X" and event.get("pid") == marker["pid"]
                     and event.get("tid") == marker["tid"] and event is not marker
                     and begin <= ns(event["ts"])
                     and ns(event["ts"]) + ns(event.get("dur", 0)) <= finish]
        counts = Counter(event["name"] for event in contained)
        association = [step for step in scheduler_steps
                       if step["schedule_monotonic_ns"] <= mono_begin
                       and mono_end <= step["complete_monotonic_ns"]]
        fx_evidence.append({
            "event": marker,
            "process": processes[marker["pid"]],
            "label": labels.get(marker["pid"]),
            "estimated_monotonic_start_ns": mono_begin,
            "estimated_monotonic_end_ns": mono_end,
            "contained_same_thread_complete_range_counts": dict(sorted(counts.items())),
            "contained_model_op_counts": {name: counts[name] for name in MODEL_OPS},
            "contained_model_op_first_last_events": {
                name: ([matches[0], matches[-1]] if len(matches) > 1 else matches)
                for name in MODEL_OPS
                if (matches := sorted([event for event in contained if event["name"] == name], key=lambda event: ns(event["ts"])))
            },
            "overlapping_graph_execute_api_records": sum(
                ns(event["ts"]) < finish and begin < ns(event["ts"]) + ns(event["dur"])
                for event in selected),
            "ends_before_first_graph_replay": finish < min(ns(event["ts"]) for event in selected),
            "scheduler_step_association": association,
        })
    result = {
        "rank": rank,
        "trace": str(path.relative_to(ROOT)),
        "trace_sha256": digest(path),
        "trace_size_bytes": path.stat().st_size,
        "profiler_info": str(info[0].relative_to(ROOT)),
        "profiler_info_sha256": digest(info[0]),
        "event_name": TARGET,
        "complete_graph_execute_api_records": len(selected),
        "first_api_start_ns": min(ns(e["ts"]) for e in selected),
        "last_api_end_ns": max(ns(e["ts"]) + ns(e["dur"]) for e in selected),
        "sum_api_duration_us": sum(ns(e["dur"]) for e in selected) / 1000,
        "duration_meaning": "Host/CANN runtime API duration, not device execution duration",
        "processes": {str(pid): {"name": processes.get(pid), "label": labels.get(pid)} for pid in {e["pid"] for e in selected}},
        "events": selected,
        "compilation_related_host_range_names": dict(Counter(e["name"] for e in compilation)),
        "compilation_related_host_range_events": compilation,
        "fx_inference_records": len(fx_evidence),
        "fx_inference_evidence": fx_evidence,
        "clock_alignment": {
            "method": "trace timestamp ns minus (profiler_info end_info.collectionTimeEnd - end_info.MonotonicTimeEnd)",
            "realtime_minus_monotonic_ns": clock_offset,
            "limitation": "End-point clock calibration supports schedule-window association; range names do not themselves carry request IDs, and this is not a device timestamp latency measurement.",
        },
        "python_cpu_operation_counts": dict(sorted(cpu_names.items())),
        "device_compute_kernel_name_counts": dict(sorted(kernel_names.items())),
        "profiling_boundaries": boundaries,
    }
    del data, events
    gc.collect()
    print(f"rank {rank}: {len(selected)} replay APIs; compilation markers {result['compilation_related_host_range_names']}; FX model op counts {[item['contained_model_op_counts'] for item in fx_evidence]}", flush=True)
    return result


def main():
    manifest_path = OUT / "graph-compiled-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    argv = manifest["api_argv"]
    policy = json.loads(argv[argv.index("--batch-admission-policy-config") + 1])
    profiler = json.loads(argv[argv.index("--profiler-config") + 1])
    client_path = OUT / "graph-compiled-profile-client.json"
    client = json.loads(client_path.read_text())
    scheduler_path = OUT / "graph-compiled-trace-733372.jsonl"
    scheduler_events = [json.loads(line) for line in scheduler_path.read_text().splitlines()]
    scheduled = {event["step"]: event for event in scheduler_events if event["event"] == "schedule"}
    controlled = {event["step"]: event for event in scheduler_events if event["event"] == "decoupled_schedule"}
    completed = {event["step"]: event for event in scheduler_events if event["event"] == "decoupled_complete"}
    probe_steps = [step for step, event in scheduled.items()
                   if event["time"] >= client["decode"]["start_monotonic"] and step in completed]
    scheduler_steps = [{
        "step": step,
        "schedule_monotonic_ns": int(Decimal(str(scheduled[step]["time"])) * 1_000_000_000),
        "complete_monotonic_ns": int(Decimal(str(completed[step]["time"])) * 1_000_000_000),
        "scheduled_tokens": scheduled[step]["scheduled_tokens"],
        "phase": scheduled[step]["phase"],
        "cohort_id": controlled[step]["cohort_id"],
        "new_prefill_ids": controlled[step]["new_prefill_ids"],
        "finish_prefill": controlled[step]["finish_prefill"],
        "layers_completed": completed[step]["layers_completed"],
        "progress_source": completed[step]["progress_source"],
    } for step in probe_steps]
    sources = [extract(rank, scheduler_steps) for rank in range(4)]
    log = OUT / "graph-compiled-server.log"
    snapshots, excerpts = [], []
    for line_number, line in enumerate(log.read_text().splitlines(), 1):
        if any(key in line for key in (
            "MuxPD decode graph rank=", "MuxPD decode graph final rank=",
            "MuxPD stream rank=", "saved AOT compiled function", "current state is RECORD",
            "Max profiling iterations", "Profiler stopped successfully",
        )):
            excerpts.append({"line": line_number, "text": line})
        match = re.search(r"MuxPD decode graph (final )?rank=(\d+) (\{.*\})", line)
        if match:
            snapshots.append({"stage": "final" if match[1] else "startup", "rank": int(match[2]), "statistics": ast.literal_eval(match[3])})
    report = {
        "schema_version": 2,
        "scope": "TP4 fenced graph/compiled-full-prefill diagnostic, eight worker iterations",
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": digest(Path(__file__)),
        "reproduction": "python local-results/perf-tp4/graph-compiled-profile-proof-audit.py",
        "method": "Count exact positive-duration ph=X CANN aclmdlRIExecuteAsync records per trace. Inspect compilation-related ranges across host tracks including cpu_op; identify the exact npu_fx_compiler inference range, same-thread enclosed model operations, and scheduler-window association using profiler end clock calibration. No CPU wrapper duration is used as device timing.",
        "limitations": [
            "Prefill compiled_calls are permission/attempt counters, not direct AOT/FX entry hits; startup AOT artifact saving does not alone prove serving calls used that entry.",
            "Graph replay API events do not identify request IDs or prove output correctness.",
            "Four ranks execute one tensor-parallel model; seven API events on four ranks are not 28 independent model steps.",
            "Eight-step automatic stop emitted RECORD-state warnings, so this is a bounded exported window, not a complete server-lifetime census.",
        ],
        "policy": policy,
        "profile_max_iterations": profiler["max_iterations"],
        "compiler_marker_implementation": {
            "source": str(COMPILER_SOURCE), "sha256": digest(COMPILER_SOURCE),
            "lines": "503-517",
            "meaning": "_CompiledFxGraph.__call__ opens npu_fx_compiler inference around self.run_kernel(*args, **kwargs) and returns gm_result. It marks entry into an actual compiled FX runtime wrapper, not the plugin permission/attempt counter.",
            "excerpt": "\n".join(COMPILER_SOURCE.read_text().splitlines()[502:517]),
        },
        "manifest": {"source": str(manifest_path.relative_to(ROOT)), "sha256": digest(manifest_path)},
        "workload": {"source": str(client_path.relative_to(ROOT)), "sha256": digest(client_path), "decode_usage": client["decode"]["usage"], "prefill_usage": client["prefill"]["usage"]},
        "server_log": {"source": str(log.relative_to(ROOT)), "sha256_at_read": digest(log), "excerpts": excerpts},
        "controller_snapshots": snapshots,
        "scheduler_trace": {"source": str(scheduler_path.relative_to(ROOT)), "sha256": digest(scheduler_path), "profile_probe_first_eight_steps": scheduler_steps[:8]},
        "graph_execute_api_records_by_rank": {str(r["rank"]): r["complete_graph_execute_api_records"] for r in sources},
        "fx_inference_records_by_rank": {str(r["rank"]): r["fx_inference_records"] for r in sources},
        "findings": [
            "Each rank exports seven CANN graph-replay API records and one actual npu_fx_compiler inference cpu_op range.",
            "The single FX range contains 40 MoE, 30 GDN, and 10 full-attention model operations on each rank, ends before the first D replay, and aligns to scheduler step 1433: the 128-token initial prefill of the decode probe, with full_forward_return and 40 completed layers.",
            "The 12000-token concurrent prefill is cohort 99, progresses 8/16/24/32/40 through layer_hooks, and is not a full compiled-forward hit in this capture.",
            "The final lifetime prefill_compiled_calls=72 counter counts allowed attempts; this bounded trace proves one full compiled prefill invocation per TP rank, not 72 independent hits or 4 independent model requests.",
        ],
        "sources": sources,
    }
    assert all(source["complete_graph_execute_api_records"] == 7 for source in sources)
    for source in sources:
        assert source["fx_inference_records"] == 1
        fx = source["fx_inference_evidence"][0]
        assert fx["contained_model_op_counts"][MODEL_OPS[0]] == 40
        assert fx["contained_model_op_counts"][MODEL_OPS[1]] == 30
        assert fx["contained_model_op_counts"][MODEL_OPS[2]] == 10
        assert fx["ends_before_first_graph_replay"]
        assert [step["step"] for step in fx["scheduler_step_association"]] == [1433]
    (OUT / "graph-compiled-profile-proof.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
