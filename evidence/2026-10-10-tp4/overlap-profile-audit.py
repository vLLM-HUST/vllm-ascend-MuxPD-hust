#!/usr/bin/env python3
"""Reproduce the TP4 audit from local CANN exports; CPU only, no torch imports."""

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PROFILE = ROOT / "local-results/tp4/overlap-profile"
TYPES = {"AI_CORE", "AI_VECTOR_CORE", "AI_VECTOR", "MIX_AIC", "MIX_AIV"}
COMM = re.compile(
    r"hccl|hcom|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all", re.I
)


def ns(value):
    return int((Decimal(str(value).strip()) * 1000).to_integral_value())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=float) + "\n")


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def length(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def metrics(rows):
    lanes = {
        lane: union((row[2], row[3]) for row in rows if row[4] == stream)
        for lane, stream in (("prefill", "45"), ("decode", "44"))
    }
    active = length(union(lanes["prefill"] + lanes["decode"]))
    overlap = intersection(lanes["prefill"], lanes["decode"])
    return {
        "prefill_active_union_us": length(lanes["prefill"]) / 1000,
        "decode_active_union_us": length(lanes["decode"]) / 1000,
        "compute_union_us": active / 1000,
        "prefill_decode_overlap_us": overlap / 1000,
        "overlap_fraction_of_compute_union": overlap / active if active else None,
    }


def audit_rank(rank):
    paths = list(PROFILE.glob(f"*rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    assert len(paths) == 1, paths
    trace = paths[0]
    csv_path = trace.with_name("kernel_details.csv")
    mapping = {
        f"device:{rank}": {"prefill": ["physical:45"], "decode": ["physical:44"]}
    }
    map_name = f"overlap-profile-lane-map-rank{rank}.json"
    analyzer_name = f"overlap-profile-analyzer-rank{rank}.json"
    save(map_name, mapping)
    command = [
        sys.executable,
        "scripts/analyze_npu_trace.py",
        str(trace.relative_to(ROOT)),
        "--lane-map",
        str((OUT / map_name).relative_to(ROOT)),
        "--output",
        str((OUT / analyzer_name).relative_to(ROOT)),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    with trace.open() as handle:
        data = json.load(handle, parse_float=Decimal)
    events = data["traceEvents"] if isinstance(data, dict) else data
    meta = [e for e in events if e.get("ph") == "M"]
    names = {
        e["pid"]: e.get("args", {}).get("name")
        for e in meta
        if e.get("name") == "process_name"
    }
    labels = {
        e["pid"]: e.get("args", {}).get("labels")
        for e in meta
        if e.get("name") == "process_labels"
    }
    hardware = {
        pid
        for pid, name in names.items()
        if name in ("Ascend Hardware", "Task Scheduler")
    }
    assert len(hardware) == 1 and {labels[pid] for pid in hardware} == {f"NPU {rank}"}
    hardware_pid = next(iter(hardware))
    selected, rows, markers = [], [], {}
    excluded_communication = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        if names.get(event.get("pid")) == "Communication":
            excluded_communication += 1
            continue
        if event.get("pid") not in hardware:
            continue
        name, args = event.get("name", ""), event.get("args", {})
        if name in ("PROFILING_ENABLE", "PROFILING_DISABLE"):
            assert name not in markers
            markers[name] = ns(event["ts"])
        kind = args.get("Task Type")
        if kind not in TYPES or COMM.search(name) or ns(event.get("dur", 0)) <= 0:
            continue
        start = ns(event["ts"])
        rows.append(
            (name, kind, start, start + ns(event["dur"]), str(args["Physic Stream Id"]))
        )
        selected.append(event)
    assert {row[4] for row in rows} == {"44", "45"}
    csv_rows = []
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["Accelerator Core"] not in TYPES or COMM.search(row["Name"]):
                continue
            start, duration = ns(row["Start Time(us)"]), ns(row["Duration(us)"])
            if duration > 0:
                assert int(row["Device_id"]) == rank
                csv_rows.append(
                    (
                        row["Name"],
                        row["Accelerator Core"],
                        start,
                        start + duration,
                        row["Stream ID"],
                    )
                )
    trace_only = Counter(rows) - Counter(csv_rows)
    csv_only = Counter(csv_rows) - Counter(rows)
    assert not trace_only and not csv_only, (trace_only, csv_only)
    measured = metrics(csv_rows)
    assert metrics(rows) == measured
    analyzer = json.loads((OUT / analyzer_name).read_text())["sources"][0]
    analyzer_metric = analyzer["devices"][f"device:{rank}"][
        "mapped_prefill_decode_compute"
    ]
    assert analyzer_metric["active_union_us"] == measured["compute_union_us"]
    assert (
        analyzer_metric["cross_stream_overlap_us"]
        == measured["prefill_decode_overlap_us"]
    )
    assert (
        analyzer_metric["overlap_fraction_of_active_union"]
        == measured["overlap_fraction_of_compute_union"]
    )
    first, last = min(row[2] for row in rows), max(row[3] for row in rows)
    outside = sum(
        row[2] < markers["PROFILING_ENABLE"] or row[3] > markers["PROFILING_DISABLE"]
        for row in rows
    )
    assert outside == 0
    pairs = []
    prefills = sorted(
        (row[2], row[3], i) for i, row in enumerate(rows) if row[4] == "45"
    )
    decodes = sorted(
        (row[2], row[3], i) for i, row in enumerate(rows) if row[4] == "44"
    )
    cursor = 0
    for start, end, pindex in prefills:
        while cursor < len(decodes) and decodes[cursor][1] <= start:
            cursor += 1
        for dstart, dend, dindex in decodes[cursor:]:
            if dstart >= end:
                break
            overlap = min(end, dend) - max(start, dstart)
            if overlap > 0:
                pairs.append((overlap, pindex, dindex))
    pairs = sorted(pairs, reverse=True)[:6]
    indexes = sorted(
        {index for _, pindex, dindex in pairs for index in (pindex, dindex)}
    )
    digest = sha(trace)
    excerpt = {
        "rank": rank,
        "source": str(trace.relative_to(ROOT)),
        "source_sha256": digest,
        "meaning": (
            "Six largest observed cross-lane pairs, selected for inspection, "
            "not a representative utilization sample."
        ),
        "pair_overlap_us": [overlap / 1000 for overlap, _, _ in pairs],
        "traceEvents": [e for e in meta if e.get("pid") == hardware_pid]
        + [selected[i] for i in indexes],
    }
    source = {
        "rank": rank,
        "trace": str(trace.relative_to(ROOT)),
        "trace_bytes": trace.stat().st_size,
        "trace_sha256": digest,
        "kernel_csv": str(csv_path.relative_to(ROOT)),
        "kernel_csv_sha256": sha(csv_path),
        "device_label": labels[hardware_pid],
        "hardware_trace_pid": hardware_pid,
        "lane_mapping": mapping[f"device:{rank}"],
        "kernel_count": len(rows),
        "task_types": dict(Counter(row[1] for row in rows)),
        "prefill_kernel_count": sum(row[4] == "45" for row in rows),
        "decode_kernel_count": sum(row[4] == "44" for row in rows),
        "excluded_communication_track_ranges": excluded_communication,
        **measured,
        "first_kernel_start_ns": first,
        "last_kernel_end_ns": last,
        "kernel_window_span_us": (last - first) / 1000,
        "compute_active_fraction_of_kernel_window": measured["compute_union_us"]
        * 1000
        / (last - first),
        "profiling_enable_ns": markers["PROFILING_ENABLE"],
        "profiling_disable_ns": markers["PROFILING_DISABLE"],
        "compute_intervals_outside_profiling_markers": outside,
        "csv_crosscheck": {
            "kernel_count": len(csv_rows),
            "trace_only_rows": 0,
            "csv_only_rows": 0,
            "independent_interval_metrics_match_trace_and_analyzer": True,
        },
        "compute_subtype_overlap_us": {
            label: metrics([r for r in rows if r[1] in types])[
                "prefill_decode_overlap_us"
            ]
            for label, types in (
                ("cube_or_mixed_aic", {"AI_CORE", "MIX_AIC"}),
                ("vector_or_mixed_aiv", {"AI_VECTOR_CORE", "AI_VECTOR", "MIX_AIV"}),
            )
        },
    }
    print(
        json.dumps(
            {
                "rank": rank,
                **measured,
                "active_fraction": source["compute_active_fraction_of_kernel_window"],
            }
        ),
        flush=True,
    )
    return source, excerpt, " ".join(command)


def main():
    sources, excerpts, commands = [], [], []
    server = ROOT / "local-results/tp4/overlap-server.log"
    lines = server.read_text().splitlines()
    for rank in range(4):
        for lane, stream in (("prefill", 45), ("decode", 44), ("caller", 47)):
            assert any(
                (
                    f"rank={rank} device=npu:{rank} lane={lane} "
                    f"physical_stream_id={stream} "
                )
                in line
                for line in lines
            )
        source, excerpt, command = audit_rank(rank)
        sources.append(source)
        excerpts.append(excerpt)
        commands.append(command)
    save("overlap-profile-excerpt.json", {"sources": excerpts})
    save(
        "overlap-profile-independent.json",
        {
            "schema_version": 1,
            "scope": (
                "Qwen3.5-35B-A3B BF16, Ascend 910B2/A2, TP4+EP4, eager V1, "
                "decoupled overlap, layers_per_slice=8"
            ),
            "workload": (
                "128-token prompt + 64-token decode, then 12000-token prefill "
                "+ 8 tokens after decode first content event"
            ),
            "method": (
                "Independent CSV Decimal-to-integer-nanosecond per-lane interval "
                "union and two-pointer intersection. Every classified CSV row "
                "must match a hardware trace row by name/type/start/end/physical "
                "stream, with multiplicity. Results must equal the separate "
                "analyzer CLI. Only AI_CORE/AI_VECTOR_CORE/AI_VECTOR/MIX_AIC/"
                "MIX_AIV; Communication and recognizable HCCL names excluded."
            ),
            "claim": (
                "Observed cross-lane device compute interval overlap. Not "
                "simultaneous per-core occupancy proof, hardware resource "
                "partitioning, or a speedup claim."
            ),
            "capture_limit": (
                "max_iterations=8 auto-stop during RECORD emitted a possible-"
                "incomplete-data warning. Complete exported intervals are inside "
                "enable/disable markers, but this cannot establish capture "
                "completeness. Bounded sample, not the entire 64-token workload. "
                "Active/span is only profile-window activity with profiler "
                "overhead, not production utilization. Subtype overlaps are "
                "separate measurements and are not additive."
            ),
            "sources": sources,
            "worker_stream_log": [
                line for line in lines if "MuxPD stream rank=" in line
            ],
            "profiler_stop_log": [
                line
                for line in lines
                if any(
                    word in line
                    for word in (
                        "GPU profiling will stop",
                        "Max profiling iterations",
                        "Stop profiler while current state",
                        "Profiler stopped successfully",
                    )
                )
            ],
            "server_log": str(server.relative_to(ROOT)),
            "server_log_sha256": sha(server),
            "offline_parse_log": "local-results/tp4/overlap-profile-analysis.log",
            "commands": [
                "source /root/workspace/runtime/env.sh && "
                '/root/workspace/.venv/bin/python -c "'
                "from torch_npu.profiler.profiler import analyse; "
                "analyse('local-results/tp4/overlap-profile')\"",
                *commands,
                "/root/workspace/.venv/bin/python "
                "evidence/2026-10-10-tp4/overlap-profile-audit.py",
            ],
        },
    )


if __name__ == "__main__":
    main()
