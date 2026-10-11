#!/usr/bin/env python3
"""Offline TP8 profile audit. Host scopes are inclusive and never summed together."""

import argparse
import collections
import hashlib
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_npu_trace as trace  # noqa: E402

SELECTED = {
    "npu_fx_compiler inference",
    "vllm::qwen_gdn_attention_core",
    "ChunkGatedDeltaRuleFunction",
    "vllm::moe_forward_shared",
    "Event::synchronize",
    "Event::record",
    "Event::wait",
    "aten::copy_",
    "aten::_to_copy",
    "aten::item",
    "aten::argmax",
    "vllm::all_gather",
    "AscendCL@aclmdlRIExecuteAsync",
    "AscendCL@aclrtSynchronizeEvent",
    "AscendCL@aclrtSynchronizeStreamWithTimeout",
    "AscendCL@aclrtSynchronizeDeviceWithTimeout",
    "AscendCL@aclrtRecordEvent",
    "AscendCL@aclrtStreamWaitEvent",
    "AscendCL@aclmdlRICaptureTaskUpdateBegin",
    "AscendCL@aclmdlRICaptureTaskUpdateEnd",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(intervals):
    metrics = trace.interval_metrics(intervals)
    if intervals:
        span = (
            max(row[1] for row in intervals) - min(row[0] for row in intervals)
        ) / 1000
    else:
        span = 0
    return {
        "count": len(intervals),
        **metrics,
        "first_last_kernel_span_us": span,
        "active_fraction_of_kernel_span": metrics["active_union_us"] / span
        if span
        else None,
    }


def audit(path):
    data = json.loads(path.read_text(), parse_float=Decimal)
    events = data["traceEvents"] if isinstance(data, dict) else data
    processes = {
        str(e["pid"]): e["args"]["name"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    durations = collections.defaultdict(list)
    intervals = collections.defaultdict(list)
    device_ops = collections.defaultdict(list)
    compute_by_stream = collections.defaultdict(list)
    fx_ranges = []
    for e in events:
        if e.get("ph") != "X" or Decimal(str(e.get("dur", 0))) <= 0:
            continue
        process = processes.get(str(e["pid"]), "unknown")
        name = str(e.get("name", ""))
        if process in {"Python", "CANN"}:
            if process == "CANN" or e.get("cat") == "cpu_op":
                durations[(process, name)].append(float(e["dur"]))
                if name == "npu_fx_compiler inference":
                    fx_ranges.append({"ts_us": str(e["ts"]), "dur_us": float(e["dur"])})
        if process not in trace.DEVICE_PROCESSES:
            continue
        args = e.get("args", {})
        kind = trace._kind(args.get("Task Type"), name)
        if kind is None:
            continue
        start, duration = trace._ns(e["ts"]), trace._ns(e["dur"])
        stream = trace._stream(args, e.get("tid"))
        intervals[kind].append((start, start + duration, stream))
        device_ops[(kind, name)].append(duration / 1000)
        if kind == "ai_core":
            compute_by_stream[stream].append((start, start + duration, stream))

    def stat(key, values):
        return {
            "process": key[0],
            "name": key[1],
            "count": len(values),
            "inclusive_sum_us": sum(values),
            "max_us": max(values),
        }

    hosts = [stat(key, values) for key, values in durations.items()]
    hosts.sort(key=lambda r: r["inclusive_sum_us"], reverse=True)
    return {
        "rank": int(re.search(r"_rank(\d+)_", str(path)).group(1)),
        "source": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "trace_event_count": len(events),
        "selected_host_scopes": [row for row in hosts if row["name"] in SELECTED],
        "top_host_inclusive_scopes": hosts[:25],
        "fx_inference_ranges": fx_ranges,
        "device": {kind: summary(rows) for kind, rows in intervals.items()},
        "compute_by_physical_stream": {
            stream: summary(rows) for stream, rows in compute_by_stream.items()
        },
        "top_device_compute_operations": sorted(
            [
                {"name": name, "count": len(values), "duration_sum_us": sum(values)}
                for (kind, name), values in device_ops.items()
                if kind == "ai_core"
            ],
            key=lambda r: r["duration_sum_us"],
            reverse=True,
        )[:20],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages", nargs="+", default=["old-profile", "native-profile-profile"]
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("profile-hotspots.json")
    )
    args = parser.parse_args()
    base = Path(__file__).parent
    report = {
        "audit_script_sha256": sha(Path(__file__)),
        "classifier_script_sha256": sha(ROOT / "scripts/analyze_npu_trace.py"),
        "limits": [
            "Diagnostic profiled windows; not unprofiled latency/throughput "
            "measurements.",
            "Eight automatic-stop worker iterations; RECORD warning means only "
            "complete exported X ranges are analyzed.",
            "Host scopes are inclusive/nested; do not add them or treat Event "
            "synchronize as removable overhead.",
            "Device compute excludes Communication track and recognized HCCL; "
            "active fraction is window coverage, not core occupancy.",
            "Native and plugin can have different actual forward/replay counts "
            "despite the same client requests and iteration limit.",
            "Client tail stall across automatic profiler stop is excluded from "
            "device-kernel span and is not an inference regression.",
        ],
        "stages": {},
    }
    for stage in args.stages:
        paths = sorted(
            (base / stage).glob("*rank*/ASCEND_PROFILER_OUTPUT/trace_view.json")
        )
        if len(paths) != 8:
            raise ValueError(
                f"Expected 8 completed rank traces for {stage}, found {len(paths)}"
            )
        ranks = []
        for path in paths:
            row = audit(path)
            ranks.append(row)
            print(stage, row["rank"], "complete", flush=True)
        report["stages"][stage] = ranks
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
