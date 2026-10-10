#!/usr/bin/env python3
"""Extract exact CANN graph execute API records; stdlib only, no device access."""

import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TARGET = "AscendCL@aclmdlRIExecuteAsync"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ns(value):
    return int((Decimal(str(value)) * 1000).to_integral_value())


def extract(rank):
    paths = list(
        (OUT / "graph8-profile").glob(
            f"*rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json"
        )
    )
    assert len(paths) == 1, paths
    path = paths[0]
    with path.open() as handle:
        data = json.load(handle)
    events = data["traceEvents"] if isinstance(data, dict) else data
    process_names = {
        event["pid"]: event.get("args", {}).get("name")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    labels = {
        event["pid"]: event.get("args", {}).get(
            "labels", event.get("args", {}).get("name")
        )
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_labels"
    }
    selected = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("name") == TARGET
    ]
    assert selected and all(ns(event.get("dur", 0)) > 0 for event in selected)
    relevant_pids = {event["pid"] for event in selected}
    first = min(ns(event["ts"]) for event in selected)
    last = max(ns(event["ts"]) + ns(event["dur"]) for event in selected)
    info = list(path.parent.parent.glob("profiler_info*.json"))
    assert len(info) == 1
    result = {
        "rank": rank,
        "trace": str(path.relative_to(ROOT)),
        "trace_sha256": digest(path),
        "trace_size_bytes": path.stat().st_size,
        "profiler_info": str(info[0].relative_to(ROOT)),
        "profiler_info_sha256": digest(info[0]),
        "event_name": TARGET,
        "complete_graph_execute_api_records": len(selected),
        "categories": dict(Counter(str(event.get("cat")) for event in selected)),
        "processes": {
            str(pid): {"name": process_names.get(pid), "label": labels.get(pid)}
            for pid in relevant_pids
        },
        "first_api_start_ns": first,
        "last_api_end_ns": last,
        "sum_api_duration_us": sum(ns(event["dur"]) for event in selected) / 1000,
        "duration_meaning": "Host/CANN runtime API duration, not device graph/kernel execution duration.",
        "events": selected,
    }
    print(f"rank {rank}: {len(selected)} exact graph execute API records", flush=True)
    return result


def main():
    sources = [extract(rank) for rank in range(4)]
    client_path = OUT / "graph8-profile-client.json"
    client = json.loads(client_path.read_text())
    manifest_path = OUT / "graph8-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    argv = manifest["api_argv"]
    profiler = json.loads(argv[argv.index("--profiler-config") + 1])
    startup_path = OUT / "graph8-server.log"
    startup_text = startup_path.read_text()
    startup = [
        line for line in startup_text.splitlines() if "MuxPD decode graph rank=" in line
    ]
    assert len(startup) == 4 and all(
        "'captured_graph_entries': 4" in line for line in startup
    )
    report = {
        "schema_version": 1,
        "scope": "TP4 graph8 fenced diagnostic capture, Qwen3.5-35B-A3B BF16",
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": digest(Path(__file__)),
        "reproduction": "python local-results/perf-tp4/graph8-profile-proof-audit.py",
        "method": "Count exact positive-duration ph=X CANN runtime events named AscendCL@aclmdlRIExecuteAsync in each independent trace. Exclude similarly named CPU wrappers, graph-capture counts and client request counts.",
        "claim": "The profiler recorded graph-execute submissions through the CANN device runtime on all four ranks; this does not rely on worker shutdown snapshots.",
        "limitations": [
            "API records do not measure device kernel duration or per-core occupancy.",
            "A runtime API record alone does not identify the request IDs, decode batch size, or correctness of generated values.",
            "The profile is bounded to eight engine iterations; its seven records per rank do not count all replays during the server lifetime.",
            "Each rank is part of the same tensor-parallel execution. Do not interpret four times seven as 28 independent model decode steps.",
            "Capture startup snapshots prove four entries per rank, not runtime replay counts; no worker shutdown snapshot is required for this bounded evidence.",
        ],
        "workload": {
            "client_report": str(client_path.relative_to(ROOT)),
            "client_report_sha256": digest(client_path),
            "decode_usage": client["decode"]["usage"],
            "prefill_usage": client["prefill"]["usage"],
            "profile_max_iterations": profiler["max_iterations"],
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": digest(manifest_path),
        },
        "startup_capture_snapshot": {
            "source": str(startup_path.relative_to(ROOT)),
            "source_sha256_at_read": hashlib.sha256(startup_text.encode()).hexdigest(),
            "source_may_grow": True,
            "rank_count": 4,
            "entries_per_rank": 4,
            "captured_log_lines": startup,
        },
        "all_ranks_have_execute_records": all(
            s["complete_graph_execute_api_records"] > 0 for s in sources
        ),
        "graph_execute_api_records_by_rank": {
            str(s["rank"]): s["complete_graph_execute_api_records"] for s in sources
        },
        "sources": sources,
    }
    (OUT / "graph8-profile-proof.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
