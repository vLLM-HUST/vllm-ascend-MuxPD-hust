"""Audit the eight planned TP8 performance files; no service or device access.

Run only after all four stages finish. Missing/invalid input is fatal and no
summary is written. Correctness, calibration, trace and profile files are never
discovered or included. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STAGES = {
    "native-initial": "native",
    "native-bracket": "native",
    "work256-stage1": "work256",
    "work256-confirm": "work256",
}
WORKLOADS = {
    "mixed": {"lengths": [128, 2048, 512, 4096], "arrival_interval": 0.12},
    "decode-heavy": {"lengths": [128], "arrival_interval": 0.0},
}
REQUESTS = 8
REPEATS = 3
OUTPUT_TOKENS = 64
TTFT_SLO = 2.0
TPOT_SLO = 0.1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, *, positive=False):
    require(type(value) in (int, float), f"{label}: expected a number")
    require(math.isfinite(value), f"{label}: nonfinite value")
    require(value > 0 if positive else value >= 0, f"{label}: invalid sign")
    return value


def integer(value, expected, label):
    require(type(value) is int and value == expected, f"{label}: expected {expected}")


def equal_number(actual, expected, label):
    number(actual, label)
    require(
        math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
        f"{label}: recorded {actual!r}, recomputed {expected!r}",
    )


def percentile(values, quantile):
    ordered = sorted(values)
    require(bool(ordered), "cannot compute a percentile of no samples")
    return ordered[math.ceil(len(ordered) * quantile) - 1]


def file_metadata(path):
    content = path.read_bytes()
    return {
        "file": path.name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def audit_request(row, index, workload, label):
    shape = WORKLOADS[workload]
    length = shape["lengths"][index % len(shape["lengths"])]
    integer(row["index"], index, label + ".index")
    integer(row["input_tokens"], length, label + ".input_tokens")
    equal_number(
        row["arrival_delay_s"], index * shape["arrival_interval"], label + ".arrival"
    )
    for field, expected in {
        "prompt_tokens": length,
        "completion_tokens": OUTPUT_TOKENS,
        "total_tokens": length + OUTPUT_TOKENS,
    }.items():
        integer(row["usage"][field], expected, label + ".usage." + field)
    duration = number(row["duration_s"], label + ".duration", positive=True)
    times = row["sse_content_event_times_s"]
    require(isinstance(times, list) and bool(times), label + ": no content SSE times")
    for value in times:
        number(value, label + ".sse_content_event_times_s", positive=True)
    require(times == sorted(times), label + ": decreasing content SSE timestamps")
    require(times[-1] <= duration + 1e-9, label + ": last SSE after request duration")
    equal_number(row["ttft_s"], times[0], label + ".ttft")
    equal_number(
        row["tpot_s"], (times[-1] - times[0]) / (OUTPUT_TOKENS - 1), label + ".tpot"
    )
    equal_number(
        row["max_content_event_gap_s"],
        max((b - a for a, b in zip(times, times[1:])), default=0),
        label + ".max_content_event_gap",
    )
    require(
        isinstance(row["text_sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", row["text_sha256"]) is not None,
        label + ": malformed text SHA256",
    )
    expected_slo = row["ttft_s"] <= TTFT_SLO and row["tpot_s"] <= TPOT_SLO
    require(type(row["slo_met"]) is bool, label + ": slo_met must be boolean")
    require(row["slo_met"] == expected_slo, label + ": incorrect SLO flag")
    # Current benchmark files do not persist these strict SSE receipts. If a
    # future compatible producer includes one, never ignore an explicit failure.
    for field in ("done_received", "transport_passed"):
        if field in row:
            require(row[field] is True, label + ": " + field + " is not true")
    if "finish_reason" in row:
        require(row["finish_reason"] == "length", label + ": unexpected finish reason")


def aggregate(runs):
    rows = [row for run in runs for row in run["requests"]]
    tokens = sum(row["usage"]["completion_tokens"] for row in rows)
    wall = sum(run["wall_s"] for run in runs)
    slo_count = sum(row["slo_met"] for row in rows)
    result = {
        "run_count": len(runs),
        "request_count": len(rows),
        "completion_tokens": tokens,
        "sum_wall_s": wall,
        "output_tokens_per_s": tokens / wall,
        "slo_met_requests": slo_count,
        "slo_attainment": slo_count / len(rows),
        "request_goodput_per_s": slo_count / wall,
        "nearest_rank_p95_index_1_based": math.ceil(0.95 * len(rows)),
        "content_sse_events": sum(len(r["sse_content_event_times_s"]) for r in rows),
    }
    for field in ("ttft_s", "tpot_s", "max_content_event_gap_s"):
        result[field.removesuffix("_s") + "_p95_s"] = percentile(
            [row[field] for row in rows], 0.95
        )
    result["text_hash_stability_by_request_index"] = []
    for index in range(REQUESTS):
        selected = [row for row in rows if row["index"] == index]
        counts = Counter(row["text_sha256"] for row in selected)
        result["text_hash_stability_by_request_index"].append(
            {
                "index": index,
                "input_tokens": selected[0]["input_tokens"],
                "samples": len(selected),
                "stable": len(counts) == 1,
                "distinct_hashes": len(counts),
                "hashes_in_stage_then_repeat_order": [
                    r["text_sha256"] for r in selected
                ],
                "hash_counts": dict(sorted(counts.items())),
            }
        )
    result["unstable_request_indexes"] = [
        item["index"]
        for item in result["text_hash_stability_by_request_index"]
        if not item["stable"]
    ]
    return result


def audit_file(path, stage, workload):
    report = json.loads(path.read_text())
    params = report["parameters"]
    for field, expected in {
        "requests": REQUESTS,
        "repeats": REPEATS,
        "output_tokens": OUTPUT_TOKENS,
        "warmup_output_tokens": 8,
    }.items():
        integer(params[field], expected, f"{path.name}.parameters.{field}")
    lengths = [int(value) for value in params["input_lengths"].split(",")]
    require(
        lengths == WORKLOADS[workload]["lengths"], path.name + ": input lengths differ"
    )
    equal_number(
        params["arrival_interval"],
        WORKLOADS[workload]["arrival_interval"],
        path.name + ".arrival_interval",
    )
    equal_number(params["ttft_slo"], TTFT_SLO, path.name + ".ttft_slo")
    equal_number(params["tpot_slo"], TPOT_SLO, path.name + ".tpot_slo")
    require(report["label"] == params["label"], path.name + ": inconsistent label")
    require(isinstance(params["model"], str) and bool(params["model"]), "missing model")
    runs = report["runs"]
    require(
        isinstance(runs, list) and len(runs) == REPEATS, path.name + ": need 3 runs"
    )
    receipts = []
    for repeat, run in enumerate(runs):
        label = f"{path.name}.repeat{repeat}"
        integer(run["repeat"], repeat, label)
        number(run["wall_s"], label + ".wall_s", positive=True)
        require(len(run["requests"]) == REQUESTS, label + ": need 8 requests")
        for index, row in enumerate(run["requests"]):
            audit_request(row, index, workload, label + f".request{index}")
        computed = aggregate([run])
        for field in (
            "output_tokens_per_s",
            "request_goodput_per_s",
            "slo_attainment",
            "ttft_p95_s",
            "tpot_p95_s",
        ):
            equal_number(run[field], computed[field], label + "." + field)
        for field in ("ttft", "tpot"):
            equal_number(
                run[field + "_p50_s"],
                percentile([row[field + "_s"] for row in run["requests"]], 0.5),
                label + "." + field + "_p50_s",
            )
        receipts.append({"repeat": repeat, **computed})
    return report, {
        "source": file_metadata(path),
        "stage": stage,
        "role": STAGES[stage],
        "workload": workload,
        "label": report["label"],
        "parameters": params,
        "aggregate": aggregate(runs),
        "runs": receipts,
    }


def percent_change(candidate, baseline):
    require(baseline > 0, "cannot calculate percentage change from a zero baseline")
    return (candidate / baseline - 1) * 100


def comparison(candidate, baseline):
    return {
        field + "_change_percent": percent_change(candidate[field], baseline[field])
        for field in (
            "output_tokens_per_s",
            "ttft_p95_s",
            "tpot_p95_s",
            "max_content_event_gap_p95_s",
        )
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    expected = [
        args.input_dir / f"{stage}-{workload}.json"
        for stage in STAGES
        for workload in WORKLOADS
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    require(not missing, "Missing planned performance inputs: " + ", ".join(missing))
    reports, stages = {}, {}
    for stage in STAGES:
        stages[stage] = {}
        for workload in WORKLOADS:
            report, result = audit_file(
                args.input_dir / f"{stage}-{workload}.json", stage, workload
            )
            reports[stage, workload] = report
            stages[stage][workload] = result
    require(
        len({r["parameters"]["model"] for r in reports.values()}) == 1,
        "Model names differ between inputs",
    )
    pooled = {}
    for role in ("native", "work256"):
        pooled[role] = {}
        for workload in WORKLOADS:
            runs = [
                run
                for stage, stage_role in STAGES.items()
                if role == stage_role
                for run in reports[stage, workload]["runs"]
            ]
            require(len(runs) == 6, "Pooled result must contain all six runs")
            pooled[role][workload] = aggregate(runs)
    comparisons = {}
    for workload in WORKLOADS:
        comparisons[workload] = {}
        for candidate_name in ("work256-stage1", "work256-confirm", "work256-all-six"):
            candidate = (
                pooled["work256"][workload]
                if candidate_name == "work256-all-six"
                else stages[candidate_name][workload]["aggregate"]
            )
            comparisons[workload][candidate_name] = {}
            for baseline_name in ("native-initial", "native-bracket", "native-all-six"):
                baseline = (
                    pooled["native"][workload]
                    if baseline_name == "native-all-six"
                    else stages[baseline_name][workload]["aggregate"]
                )
                comparisons[workload][candidate_name][baseline_name] = comparison(
                    candidate, baseline
                )
    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete: all eight planned performance inputs available and audited",
        "audit_script": file_metadata(Path(__file__)),
        "scope": "TP8 graph finite-burst performance. No correctness, calibration, trace or profiler samples are included.",
        "formulas": {
            "pooled_throughput": "sum(completion_tokens) / sum(run.wall_s)",
            "ttft": "first nonempty text SSE time since request dispatch",
            "request_average_tpot": "(last nonempty text SSE - first nonempty text SSE) / (completion_tokens - 1)",
            "percentile": "nearest rank over all request values: sorted[ceil(q*n)-1], no averaging per-run percentiles",
            "goodput": "count(TTFT <= 2.0s and request-average TPOT <= 0.1s) / sum(run.wall_s)",
            "slo_attainment": "SLO-qualified requests / all requests",
            "percent_change": "100 * (candidate / baseline - 1)",
            "text_stability": "distinct text SHA256 values for each fixed request index across all selected repeats",
        },
        "validation": {
            "expected_inputs": 8,
            "audited_inputs": 8,
            "requests_per_run": REQUESTS,
            "runs_per_stage_workload": REPEATS,
            "output_tokens_per_request": OUTPUT_TOKENS,
            "usage_prompt_completion_total_exact": True,
            "content_sse_times_monotonic_and_within_duration": True,
            "stored_timing_and_slo_aggregates_recomputed": True,
            "limitations": [
                "benchmark_serving.py does not persist raw SSE frames, finish_reason, DONE flag, response IDs, absolute request starts or all warmup receipts. This audit cannot independently prove those transport properties; use the separate strict serving reports.",
                "Content SSE events are not individual token observations. Request-average TPOT and maximum content event gap are not token TBT percentiles.",
                "TP8, actual visible devices, model files, graph replay, source pins and absence of concurrent load require manifests/environment/device evidence; benchmark JSON alone cannot prove them.",
                "Stage ordering and adaptive-state carry-over must be documented from run manifests. Six sequential repeats are not six independent cold starts.",
                "Text hash instability is reported without assigning a numerical or cache-state cause. Small performance changes do not establish statistical significance.",
            ],
        },
        "stages": stages,
        "all_six_runs": pooled,
        "comparisons": comparisons,
    }
    destination = args.output or args.input_dir / "benchmark-summary.json"
    require(
        destination.resolve() not in {p.resolve() for p in expected},
        "output overwrites an input",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    print(
        json.dumps({"output": str(destination), "status": output["status"]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
