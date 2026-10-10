#!/usr/bin/env python3
"""Compare new isolated traces with the existing shared summary, never reparsing it."""

import argparse
import csv
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HELPER = OUT / "native-vs-shared-profile-audit.py"
spec = importlib.util.spec_from_file_location("bounded_profile_audit", HELPER)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
MODE = "isolated_overlap8"
base.PROFILE_DIRS[MODE] = OUT / "isolated8-profile"


def csv_check(capture):
    path = ROOT / capture["source"]
    path = path.with_name("kernel_details.csv")
    selected, streams = [], {"44": [], "45": []}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Accelerator Core"] not in base.COMPUTE_TYPES:
                continue
            if base.COMM.search(row["Name"]):
                continue
            start, duration = base.ns(row["Start Time(us)"]), base.ns(row["Duration(us)"])
            if duration <= 0:
                continue
            assert str(row["Device_id"]) == str(capture["rank"])
            interval = (start, start + duration)
            selected.append(interval)
            streams[row["Stream ID"]].append(interval)
    device = capture["device_compute"]
    combined = base.union(selected)
    p, d = (base.union(streams[key]) for key in ("45", "44"))
    active = base.length(combined)
    overlap = base.intersection(p, d)
    assert len(selected) == device["kernel_count"]
    assert active == int(Decimal(str(device["active_union_us"])) * 1000)
    assert selected and combined[0][0] == device["first_kernel_start_ns"]
    assert combined[-1][1] == device["last_kernel_end_ns"]
    for stream, intervals in streams.items():
        assert len(intervals) == device["streams"][stream]["kernel_count"]
        assert base.length(base.union(intervals)) == int(
            Decimal(str(device["streams"][stream]["active_union_us"])) * 1000
        )
    assert base.length(p) + base.length(d) - active == overlap
    device["mapped_pd_overlap_us"] = overlap / 1000
    device["mapped_pd_overlap_fraction_of_compute_union"] = overlap / active
    return {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": base.sha(path),
        "compute_kernel_count": len(selected),
        "compute_union_us": active / 1000,
        "pd_intersection_us": overlap / 1000,
        "trace_csv_exact_match": True,
        "stream_id_mapping": "CSV Stream ID agrees exactly with trace physical streams 45=P, 44=D",
    }


def normalized(capture):
    forwards = capture["model_forward_equivalents"]["count"]
    result = dict(capture["per_forward_equivalent"])
    for name in ("vllm::all_reduce", "vllm::all_gather", "Event::record", "Event::wait"):
        stats = capture["cpu_operations"][name]
        result[f"{name}_cpu_inclusive_us"] = stats["inclusive_sum_us"] / forwards
        result[f"{name}_cpu_exclusive_us"] = (
            stats["exclusive_of_contained_cpu_ops_sum_us"] / forwards
        )
    for name in ("AscendCL@hcom_allReduce_", "AscendCL@hcom_allGather_"):
        stats = capture["cann_runtime_apis"][name]
        result[f"{name}_count"] = stats["count"] / forwards
        result[f"{name}_inclusive_us"] = stats["inclusive_sum_us"] / forwards
    result["communication_track_union_us"] = (
        capture["excluded_communication_process_spans"]["union_us"] / forwards
    )
    return result


def render(report):
    rows = [
        "# Isolated versus shared HCCL groups: bounded TP4 profile audit",
        "",
        "Both profiles are diagnostic 128-token decode + 12,000-token prefill probes with eight-step automatic profiler stopping; they are not the timed throughput benchmark. The isolated profile used fixed quota 8, separate P/D groups, and a contiguous P slice before complete D submission. It predates the D-only fast path and compiled-prefill experiments. The old shared profile also contains the old event gates and per-layer host alternation, so this comparison does not isolate communicator identity as the only changed variable.",
        "",
        "Each rank is analysed independently. Forward equivalents are inferred from matching MoE/40, GDN/30, and logits all-gather counts, not direct whole-model timing ranges. Raw work counts are retained below. The shared values are read from the existing hashed summary without reparsing its large traces; isolated trace intervals are checked independently against kernel_details.csv, including stream assignment and exact nanosecond unions.",
        "",
        "| Mode | Rank | Forwards | MoE / GDN | Compute kernels | Compute union ms | Kernel window ms | Active/window | P/D overlap ms | Overlap/compute union |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["captures"]:
        d, f = row["device_compute"], row["model_forward_equivalents"]
        rows.append(
            f"| {row['mode']} | {row['rank']} | {f['count']} | {f['moe_layer_calls']} / {f['gdn_layer_calls']} | {d['kernel_count']} | {d['active_union_us']/1000:.3f} | {d['kernel_window_us']/1000:.3f} | {d['active_fraction_of_kernel_window']:.2%} | {d['mapped_pd_overlap_us']/1000:.3f} | {d['mapped_pd_overlap_fraction_of_compute_union']:.2%} |"
        )
    rows += [
        "", "Normalized counts per forward equivalent:", "",
        "| Mode | Rank | All-reduce | Python record / wait | CANN record / wait |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["captures"]:
        p = row["normalized"]
        rows.append(
            f"| {row['mode']} | {row['rank']} | {p['all_reduce_cpu_calls']:.3f} | {p['python_event_record_calls']:.3f} / {p['python_event_wait_calls']:.3f} | {p['cann_record_event_calls']:.3f} / {p['cann_stream_wait_event_calls']:.3f} |"
        )
    rows += [
        "", "Communication-related intervals per forward equivalent (microseconds):", "",
        "| Mode | Rank | All-reduce CPU inclusive / exclusive | HCOM all-reduce host API inclusive | Communication-track union | Hardware HCCL-classified tasks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["captures"]:
        p = row["normalized"]
        rows.append(
            f"| {row['mode']} | {row['rank']} | {p['vllm::all_reduce_cpu_inclusive_us']:.1f} / {p['vllm::all_reduce_cpu_exclusive_us']:.1f} | {p['AscendCL@hcom_allReduce__inclusive_us']:.1f} | {p['communication_track_union_us']:.1f} | {row['device_communication_tasks']['count']} |"
        )
    shared = [r for r in report["captures"] if r["mode"] == "shared_overlap8"]
    isolated = [r for r in report["captures"] if r["mode"] == MODE]
    old, new = shared[0]["normalized"], isolated[0]["normalized"]
    rows += [
        "",
        f"The exported all-reduce count remains 81 per forward. Isolated execution reduces Python event records by {1-new['python_event_record_calls']/old['python_event_record_calls']:.1%} and waits by {1-new['python_event_wait_calls']/old['python_event_wait_calls']:.1%} per forward relative to the old shared implementation. This is consistent with removing per-collective P/D gates and per-layer host switching; it does not measure the end-to-end benefit of either change in isolation.",
        "",
        "The same 16,086 selected kernels per rank cover essentially unchanged model-compute time (about 368–373 ms). The isolated kernel windows are about 2.657 s versus 2.916 s for shared, while isolated overlap is 2.63–3.30% versus shared 2.86–4.49%. Thus this capture shows fewer synchronization calls and a shorter submission window, not an across-rank increase in compute overlap. Rank 0 all-reduce CPU inclusive time falls from 34.36 to 13.15 ms per forward, whereas its HCOM host API inclusive time remains about 26.62 versus 27.04 ms. These differently nested/threaded intervals suggest less wrapper/gate work; they do not establish lower physical link latency or a quantitative causal attribution of throughput.",
        "",
        "The compute numerator includes only explicit AI_CORE/AI_VECTOR/MIX hardware tasks, excluding recognizable HCCL/communication names. P/D overlap is the intersection of each lane's union on verified physical streams 45/44. Communication-track spans, HCCL host wrappers, event waits and DMA are excluded from model compute. Compute active/window is interval coverage under profiler overhead, not AICore occupancy or hardware resource partitioning.",
        "",
        "CPU wrapper inclusive times contain nested work and cannot be summed across operations; exclusive times subtract contained cpu_op intervals on the same thread and still include uninstrumented time. HCOM API duration is host time. Communication-track union includes waiting and transport spans, not useful communication compute. Zero HCCL-classified hardware tasks does not mean no communication: SDMA, Notify Wait and Write Value tasks are retained separately in JSON. The different profile windows and workload timing prohibit interpreting communication-track union as a matched latency improvement.",
        "",
        "All four isolated ranks emitted a RECORD-state warning on automatic stop. Every selected compute kernel lies between one exported PROFILING_ENABLE and PROFILING_DISABLE marker, and layer/logits work counts agree, but these checks do not prove the entire request lifetime or every profiler record was captured. Offline parsing completed after the worker-daemon parser refusal. Absolute timings are perturbed by instrumentation and are not substituted for the separate unprofiled benchmarks.",
        "",
        f"Reproduce: `{report['reproduction']}`. JSON includes trace/CSV/metadata/helper/summary hashes, the isolated frozen source manifest hash, server-log excerpts with line numbers, every rank's raw counts and normalized metrics. No raw trace is copied into the report.",
        "",
    ]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    previous = OUT / "native-vs-shared-profile.json"
    captures = [
        r for r in json.loads(previous.read_text())["captures"]
        if r["mode"] == "shared_overlap8" and r["rank"] in args.ranks
    ]
    assert len(captures) == len(args.ranks)
    for rank in args.ranks:
        row = base.audit(MODE, rank)
        assert set(row["device_compute"]["streams"]) == {"44", "45"}
        row["independent_csv_check"] = csv_check(row)
        captures.append(row)
    for row in captures:
        row["normalized"] = normalized(row)
    log = OUT / "isolated8-server.log"
    excerpts = [
        {"line": i, "text": line}
        for i, line in enumerate(log.read_text().splitlines(), 1)
        if any(s in line for s in (
            "MuxPD stream rank=", "MuxPD isolated group rank=", "Max profiling iterations",
            "current state is RECORD", "Profiler stopped successfully",
        ))
    ]
    artifacts = [previous, HELPER, log, OUT / "isolated8-source/manifest.json",
                 OUT / "isolated8-profile-client.json", OUT / "isolated8-profile-analysis.log"]
    report = {
        "schema_version": 1,
        "script_sha256": base.sha(Path(__file__)),
        "reproduction": "python local-results/perf-tp4/isolated-vs-shared-profile-audit.py --ranks "
        + " ".join(map(str, args.ranks)),
        "source_artifacts": {str(p.relative_to(ROOT)): base.sha(p) for p in artifacts},
        "lane_map": {str(rank): {"prefill_physical": 45, "decode_physical": 44} for rank in args.ranks},
        "server_log_excerpts": excerpts,
        "captures": captures,
    }
    (OUT / "isolated-vs-shared-profile.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "isolated-vs-shared-profile.md").write_text(render(report))


if __name__ == "__main__":
    main()
