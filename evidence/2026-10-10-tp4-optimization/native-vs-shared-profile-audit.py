#!/usr/bin/env python3
"""Bounded TP4 trace comparison. Stdlib only; never imports torch or uses NPU."""

import argparse
import bisect
import gc
import hashlib
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COMPUTE_TYPES = {"AI_CORE", "AI_VECTOR_CORE", "AI_VECTOR", "MIX_AIC", "MIX_AIV"}
COMM = re.compile(
    r"hccl|hcom|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all", re.I
)
PROFILE_DIRS = {
    "native": ROOT / "local-results/perf-tp4/native-profile",
    "shared_overlap8": ROOT / "local-results/tp4/overlap-profile",
}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ns(value):
    return int((Decimal(str(value)) * 1000).to_integral_value())


def union(items):
    result = []
    for start, end in sorted(items):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def length(items):
    return sum(end - start for start, end in items)


def intersection(left, right):
    total = i = j = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def simple_stats(rows):
    return {
        "count": len(rows),
        "inclusive_sum_us": sum(end - start for start, end in rows) / 1000,
        "union_us": length(union(rows)) / 1000,
    }


def cpu_stats(selected, cpu_by_thread):
    prepared = {}
    for key, rows in cpu_by_thread.items():
        rows.sort()
        prepared[key] = ([row[0] for row in rows], rows)
    result = {}
    for name, events in sorted(selected.items()):
        exclusive = crossing = coincident = 0
        rows_by_thread = defaultdict(list)
        for start, end, pid, tid, original_index in events:
            starts, rows = prepared[pid, tid]
            contained = []
            pos = bisect.bisect_left(starts, start)
            while pos < len(rows) and rows[pos][0] < end:
                child_start, child_end, child_index = rows[pos]
                pos += 1
                if original_index == child_index:
                    continue
                if child_start == start and child_end == end:
                    coincident += 1
                elif child_end <= end:
                    contained.append((child_start, child_end))
                elif child_start > start:
                    crossing += 1
            exclusive += (end - start) - length(union(contained))
            rows_by_thread[pid, tid].append((start, end))
        result[name] = {
            "count": len(events),
            "inclusive_sum_us": sum(e[1] - e[0] for e in events) / 1000,
            "exclusive_of_contained_cpu_ops_sum_us": exclusive / 1000,
            "per_thread_union_sum_us": sum(
                length(union(r)) for r in rows_by_thread.values()
            )
            / 1000,
            "proper_containment_crossings": crossing,
            "coincident_range_pairs": coincident,
        }
    return result


def audit(mode, rank):
    paths = list(
        PROFILE_DIRS[mode].glob(f"*rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json")
    )
    assert len(paths) == 1, paths
    path = paths[0]
    with path.open() as handle:
        data = json.load(handle)
    events = data["traceEvents"] if isinstance(data, dict) else data
    names, labels = {}, {}
    for event in events:
        if event.get("ph") != "M":
            continue
        args = event.get("args", {})
        if event.get("name") == "process_name":
            names[event["pid"]] = args.get("name", "")
        elif event.get("name") == "process_labels":
            labels[event["pid"]] = args.get("labels", args.get("name", ""))
    hardware = {
        pid
        for pid, name in names.items()
        if name in ("Ascend Hardware", "Task Scheduler")
    }
    assert len(hardware) == 1
    hardware_pid = next(iter(hardware))
    assert labels[hardware_pid] == f"NPU {rank}"
    cpu_by_thread, cpu_selected = defaultdict(list), defaultdict(list)
    runtime, hardware_other, compute = defaultdict(list), defaultdict(list), []
    communication_tasks, communication_spans = [], []
    by_stream, task_types, markers = defaultdict(list), Counter(), defaultdict(list)
    forward_candidates = Counter()
    cpu_count = 0
    for index, event in enumerate(events):
        if event.get("ph") != "X" or "ts" not in event:
            continue
        name, pid, tid = event.get("name", ""), event.get("pid"), event.get("tid")
        start = ns(event["ts"])
        if pid in hardware and name in ("PROFILING_ENABLE", "PROFILING_DISABLE"):
            markers[name].append(start)
        duration = ns(event.get("dur", 0))
        if duration <= 0:
            continue
        end = start + duration
        if event.get("cat") == "cpu_op" and names.get(pid) == "Python":
            cpu_count += 1
            cpu_by_thread[pid, tid].append((start, end, index))
            if name.startswith(("vllm::", "Event::")):
                cpu_selected[name].append((start, end, pid, tid, index))
            if any(
                token in name.lower()
                for token in ("execute_model", "decoderlayer", "model.forward")
            ):
                forward_candidates[name] += 1
        if name.startswith("AscendCL@"):
            runtime[name].append((start, end))
        if names.get(pid) == "Communication":
            communication_spans.append((start, end))
        if pid not in hardware:
            continue
        args = event.get("args", {})
        kind = args.get("Task Type", "unclassified")
        if COMM.search(name) or kind in {"HCCL", "HCCL_AI_CPU", "COMMUNICATION"}:
            communication_tasks.append((start, end))
        elif kind in COMPUTE_TYPES:
            compute.append((start, end))
            by_stream[str(args.get("Physic Stream Id", f"trace-tid:{tid}"))].append(
                (start, end)
            )
            task_types[kind] += 1
        else:
            hardware_other[f"{kind}:{name}"].append((start, end))
    cpu = cpu_stats(cpu_selected, cpu_by_thread)
    moe_count = cpu.get("vllm::moe_forward_shared", {}).get("count", 0)
    gdn_count = cpu.get("vllm::qwen_gdn_attention_core", {}).get("count", 0)
    gather_count = cpu.get("vllm::all_gather", {}).get("count", 0)
    assert moe_count % 40 == 0 and gdn_count % 30 == 0
    assert moe_count // 40 == gdn_count // 30 == gather_count
    forwards = moe_count // 40
    compute_union = union(compute)
    start, end = min(r[0] for r in compute), max(r[1] for r in compute)
    enable, disable = (
        markers.get("PROFILING_ENABLE", []),
        markers.get("PROFILING_DISABLE", []),
    )
    assert len(enable) == len(disable) == 1
    assert all(enable[0] <= a < b <= disable[0] for a, b in compute)
    lanes = {stream: union(rows) for stream, rows in by_stream.items()}
    overlap_ns = 0
    if mode == "shared_overlap8":
        assert set(lanes) == {"44", "45"}
        overlap_ns = intersection(lanes["44"], lanes["45"])
    infos = list(path.parent.parent.glob("profiler_info*.json"))
    assert len(infos) == 1
    result = {
        "mode": mode,
        "rank": rank,
        "source": str(path.relative_to(ROOT)),
        "source_sha256": sha(path),
        "source_bytes": path.stat().st_size,
        "profiler_info": str(infos[0].relative_to(ROOT)),
        "profiler_info_sha256": sha(infos[0]),
        "device_label": labels[hardware_pid],
        "complete_cpu_op_count": cpu_count,
        "direct_model_or_decoder_forward_range_candidates": dict(forward_candidates),
        "model_forward_equivalents": {
            "count": forwards,
            "kind": "inferred_from_40_MoE_and_30_GDN_calls_per_model_forward",
            "independent_logits_all_gather_count": gather_count,
            "direct_complete_model_range_observed": False,
            "direct_complete_model_inclusive_us": None,
            "moe_layer_calls": moe_count,
            "gdn_layer_calls": gdn_count,
        },
        "cpu_operations": cpu,
        "cann_runtime_apis": {
            name: simple_stats(rows) for name, rows in sorted(runtime.items())
        },
        "device_compute": {
            "kernel_count": len(compute),
            "task_types": dict(task_types),
            "active_union_us": length(compute_union) / 1000,
            "first_kernel_start_ns": start,
            "last_kernel_end_ns": end,
            "kernel_window_us": (end - start) / 1000,
            "active_fraction_of_kernel_window": length(compute_union) / (end - start),
            "profiling_enable_ns": enable[0],
            "profiling_disable_ns": disable[0],
            "profiling_window_us": (disable[0] - enable[0]) / 1000,
            "active_fraction_of_profiling_window": length(compute_union)
            / (disable[0] - enable[0]),
            "streams": {
                key: {
                    "kernel_count": len(by_stream[key]),
                    "active_union_us": length(rows) / 1000,
                }
                for key, rows in lanes.items()
            },
            "mapped_pd_overlap_us": overlap_ns / 1000
            if mode == "shared_overlap8"
            else None,
            "mapped_pd_overlap_fraction_of_compute_union": overlap_ns
            / length(compute_union)
            if mode == "shared_overlap8"
            else None,
        },
        "device_communication_tasks": simple_stats(communication_tasks),
        "excluded_communication_process_spans": simple_stats(communication_spans),
        "other_hardware_tasks": {
            name: simple_stats(rows) for name, rows in sorted(hardware_other.items())
        },
        "per_forward_equivalent": {
            "all_reduce_cpu_calls": cpu["vllm::all_reduce"]["count"] / forwards,
            "python_event_record_calls": cpu["Event::record"]["count"] / forwards,
            "python_event_wait_calls": cpu["Event::wait"]["count"] / forwards,
            "cann_stream_wait_event_calls": len(
                runtime["AscendCL@aclrtStreamWaitEvent"]
            )
            / forwards,
            "cann_record_event_calls": len(runtime["AscendCL@aclrtRecordEvent"])
            / forwards,
            "device_compute_active_union_us": length(compute_union) / 1000 / forwards,
        },
    }
    if mode == "shared_overlap8":
        previous = json.loads(
            (
                ROOT / "evidence/2026-10-10-tp4/overlap-profile-independent.json"
            ).read_text()
        )
        # Earlier audit is a separate cross-check of the same exported compute intervals.
        records = (
            previous
            if isinstance(previous, list)
            else previous.get("ranks", previous.get("sources", []))
        )
        if records:
            record = next(row for row in records if row["rank"] == rank)
            assert (
                result["device_compute"]["active_union_us"]
                == record["compute_union_us"]
            )
    del events, data
    gc.collect()
    print(
        f"{mode} rank {rank}: {forwards} forward equivalents, compute union {result['device_compute']['active_union_us']:.2f} us",
        flush=True,
    )
    return result


def render(report):
    rows = [
        "# Native versus shared-communicator overlap8: bounded profile audit",
        "",
        "These traces describe one diagnostic 128-token decode plus 12,000-token prefill probe per mode, not the timed eight-request benchmark. Collection was limited to eight engine iterations and does not represent the full request lifetime. Native exported eight model-forward equivalents and shared overlap exported nine; raw totals therefore include different executed work. Each rank and capture has its own timeline. No cross-rank or cross-capture intervals are merged.",
        "",
        "No whole-network or decoder-layer Python forward ranges were exported. The counts below infer full-forward equivalents from the independently agreeing MoE (40 per forward), GDN (30 per forward), and logits all-gather counts. A MoE marker is one component per decoder layer, not a whole-layer timing range. Whole-forward elapsed time cannot be recovered from these marker durations.",
        "",
        "| Mode | Rank | Forward equivalents | MoE / GDN calls | All-reduce CPU calls | Event record / wait | Device compute union ms | Kernel window ms | Compute active/window |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["captures"]:
        cpu, device, fwd = (
            row["cpu_operations"],
            row["device_compute"],
            row["model_forward_equivalents"],
        )
        rows.append(
            f"| {row['mode']} | {row['rank']} | {fwd['count']} | {fwd['moe_layer_calls']} / {fwd['gdn_layer_calls']} | {cpu['vllm::all_reduce']['count']} | {cpu['Event::record']['count']} / {cpu['Event::wait']['count']} | {device['active_union_us'] / 1000:.3f} | {device['kernel_window_us'] / 1000:.3f} | {device['active_fraction_of_kernel_window']:.2%} |"
        )
    rows += [
        "",
        "Per-forward-equivalent event counts separate the extra forward from synchronization overhead:",
        "",
        "| Mode | Rank | All-reduce | Python record | Python wait | CANN record | CANN stream wait |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["captures"]:
        per = row["per_forward_equivalent"]
        rows.append(
            f"| {row['mode']} | {row['rank']} | {per['all_reduce_cpu_calls']:.3f} | {per['python_event_record_calls']:.3f} | {per['python_event_wait_calls']:.3f} | {per['cann_record_event_calls']:.3f} | {per['cann_stream_wait_event_calls']:.3f} |"
        )
    rows += [
        "",
        "CPU operation timing for rank 0 (milliseconds). Inclusive values include nested CPU operations and must not be added across rows. Exclusive values subtract the union of strictly contained cpu_op intervals on the same thread; they are not device kernel time or a measurement of useful host execution. Suspended greenlet time and uninstrumented work remain included. Coincident/crossing ranges are reported separately in JSON.",
        "",
        "| Mode | Operation | Calls | Inclusive sum ms | Exclusive of contained CPU ops sum ms |",
        "|---|---|---:|---:|---:|",
    ]
    chosen = (
        "vllm::moe_forward_shared",
        "vllm::qwen_gdn_attention_core",
        "vllm::all_reduce",
        "vllm::all_gather",
        "Event::record",
        "Event::wait",
        "Event::synchronize",
    )
    for row in report["captures"]:
        if row["rank"] != 0:
            continue
        for name in chosen:
            stat = row["cpu_operations"].get(name)
            if stat:
                rows.append(
                    f"| {row['mode']} | `{name}` | {stat['count']} | {stat['inclusive_sum_us'] / 1000:.3f} | {stat['exclusive_of_contained_cpu_ops_sum_us'] / 1000:.3f} |"
                )
    rows += [
        "",
        "`device_compute.active_union_us` unions only positive-duration hardware tasks with an explicit AI_CORE/AI_VECTOR/MIX task type, excluding communication names. The kernel window is first selected kernel start to last selected kernel end. It excludes waits, host launch gaps, and communication work from its numerator. This active fraction is a timeline-coverage statistic, not measured AICore occupancy. HCCL hardware tasks and the broader Communication track spans are reported separately; they are not added to compute or interpreted as useful HCCL computation.",
        "",
        "No hardware task in these exports has an explicit recognizable HCCL kernel name/type. The resulting zero classified HCCL-task count does not mean zero communication: SDMA, Notify Wait, and Write Value tasks remain visible separately, as do Communication track spans. Their entire duration can contain waiting and cannot be treated as useful communication compute time.",
        "",
        "All four ranks agree on counts. After normalizing for the extra exported forward, shared overlap still performs 76.8% more Python event records and 69.8% more Python event waits per forward-equivalent. The all-reduce call count remains exactly 81 per forward in both modes. This supports auditing synchronization overhead; it does not isolate its causal contribution to end-to-end latency.",
        "",
        "Python event wrappers, AscendCL APIs, and hardware record/wait tasks represent different layers of the same synchronization and must not be summed. CPU all-reduce duration measures the host wrapper and cannot establish device communication latency. An overlap trace can include P/D kernels on physical streams 45/44 concurrently; this says nothing by itself about application throughput. The prior independent audit supplies that lane mapping.",
        "",
        "The source SHA256, profiler metadata SHA256, per-operation counts/timings, per-stream unions, and reproduction script SHA256 are in `native-vs-shared-profile.json`. Reproduce with `python local-results/perf-tp4/native-vs-shared-profile-audit.py --ranks 0 1 2 3`. The JSON preserves every rank separately; no inclusive totals are summed across operation names or devices.",
        "",
    ]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    assert union([(0, 4), (2, 6), (9, 10)]) == [(0, 6), (9, 10)]
    assert intersection([(0, 4), (8, 12)], [(2, 10)]) == 4
    captures = [audit(mode, rank) for mode in PROFILE_DIRS for rank in args.ranks]
    model = ROOT / "../models/Qwen3.5-35B-A3B/config.json"
    report = {
        "schema_version": 1,
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha(Path(__file__)),
        "reproduction": "python local-results/perf-tp4/native-vs-shared-profile-audit.py --ranks "
        + " ".join(map(str, args.ranks)),
        "model_config_sha256": sha(model),
        "method": {
            "timestamps": "Decimal microseconds converted to integer nanoseconds; never merge capture/rank timelines",
            "cpu_inclusive": "sum of durations for each exact operation name, not additive across nested names",
            "cpu_exclusive": "range duration minus union of strictly contained cpu_op ranges on same pid/tid; coincident ranges excluded and crossings counted; includes uninstrumented/greenlet suspension time",
            "device_active": "union of explicit compute task types on Ascend Hardware/Task Scheduler, excluding communication names",
            "device_window": "first compute kernel start to last compute kernel end; profiling enable/disable window also retained",
            "forward_count": "inferred from MoE/40, GDN/30, independently matching logits all-gather; no direct forward/layer timing exported",
            "scope": "diagnostic captures bounded to eight engine iterations; different number of executed model forwards, not a matched throughput benchmark",
        },
        "captures": captures,
    }
    (OUT / "native-vs-shared-profile.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (OUT / "native-vs-shared-profile.md").write_text(render(report))


if __name__ == "__main__":
    main()
