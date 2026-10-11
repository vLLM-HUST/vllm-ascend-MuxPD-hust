#!/usr/bin/env python3
"""Reproduce contention SVGs from independently audited, complete stage grids.

No benchmark or runtime module is imported. Every role must have all four
workloads and every registered repeat. Matplotlib is imported only after input
validation and is unnecessary for --validate-only. Example final invocation:

  python plot_contention.py \
    --native16k native16k-initial,native16k-confirm \
    --native2k native2k-initial,native2k-confirm \
    --old old-work256-initial,old-work256-confirm-clean \
    --guard guard-v2-initial,guard-v2-confirm \
    --exclude-stage 'old-work256-confirm=CPU-test overlap; all four rerun clean' \
    --exclude-stage 'guard-initial=v1 implementation; retained separately'

Stage names are explicit, not guessed. For a provisional initial-only figure,
pass one stage per role and --stage-labels initial; a missing guard is an error.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROLES = ("native16k", "native2k", "old", "guard")
LABELS = ("Native 16K", "Native 2K", "Old work256", "Decode-budget guard")
COLORS = ("#355C8C", "#CA782C", "#6D5B97", "#267B5B")
WORKLOADS = ("decode-control", "p8k", "p12k", "p12k-burst")
ROLE_CONTRACT = {
    "native16k": ("native", 16384), "native2k": ("native", 2048),
    "old": ("old", 16384), "guard": ("candidate", 16384),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_receipt(directory, audited):
    candidates = [directory / Path(audited["file"]).name]
    recorded = Path(audited.get("recorded_output", audited["file"])).name
    candidates.extend((directory / recorded, directory / (recorded + ".gz")))
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    require(path is not None, f"Missing audited report: {recorded}")
    stored = path.read_bytes()
    raw = gzip.decompress(stored) if path.suffix == ".gz" else stored
    require(sha(raw) == audited["sha256"], f"Report SHA changed: {path.name}")
    return json.loads(raw), {
        "file": path.name, "sha256": sha(raw), "stored_file_sha256": sha(stored),
    }


def stage_list(text):
    result = text.split(",")
    require(all(result) and len(result) == len(set(result)), "Empty/duplicate stages")
    require(all(Path(name).name == name for name in result), "Stages must be names")
    return result


def compact_request(row):
    return {
        "request_id": row["request_id"], "phase": row["phase"], "index": row["index"],
        "started_monotonic_s": row["started_monotonic_s"],
        "finished_monotonic_s": row["finished_monotonic_s"],
        "first_token_monotonic_s": row["first_token_monotonic_s"],
        "token_events": [
            {"time": event["received_monotonic_s"], "tokens": event["token_count"]}
            for event in row["events"] if event.get("token_count", 0)
        ],
        "metrics": row["metrics"], "payload_sha256": row["payload_sha256"],
        "raw_sse_sha256": row["raw_sse_sha256"],
    }


def load_inputs(args):
    directory = args.input_dir
    audit_path = args.audit or directory / "contention-audit.json"
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    plan_path = args.plan or directory / "experiment-plan-v2.json"
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    registered_workloads = {item["name"]: item for item in plan["workloads"]}
    selections = {role: stage_list(getattr(args, role)) for role in ROLES}
    labels = stage_list(args.stage_labels)
    require(all(len(stages) == len(labels) for stages in selections.values()),
            "Every role needs exactly one stage for each stage label")
    selected_names = [stage for stages in selections.values() for stage in stages]
    require(len(selected_names) == len(set(selected_names)),
            "One stage cannot stand in for multiple roles or periods")
    exclusions = {}
    for entry in args.exclude_stage:
        name, separator, reason = entry.partition("=")
        require(separator and name and reason.strip(), "Use --exclude-stage NAME=WHY")
        require(name not in selected_names and name not in exclusions,
                "Excluded stage is selected or repeated")
        exclusions[name] = reason
    require(args.repeats_per_stage > 0, "Repeat count must be positive")
    require(args.repeats_per_stage == plan["repeats"],
            "Repeat count differs from the registered plan")
    index = {}
    omitted = []
    diagnostics = []
    for audited in audit["reports"]:
        stage = audited["stage"]
        parameters = audited["manifest_role"]
        if parameters.get("profile") or parameters.get("trace"):
            diagnostics.append(audited["label"])
            continue
        if stage not in selected_names:
            require(stage in exclusions,
                    f"Completed stage {stage} omitted without an explicit reason")
            omitted.append({"label": audited["label"], "reason": exclusions[stage]})
            continue
        require(audited["passed"] is True, f"Audit failed: {audited['label']}")
        workload = next((w for w in WORKLOADS if audited["label"] == f"{stage}-{w}"),
                        None)
        require(workload is not None, f"Unrecognized workload: {audited['label']}")
        key = (stage, workload)
        require(key not in index, f"Duplicate audited report: {key}")
        index[key] = audited

    missing = [(stage, workload) for stage in selected_names for workload in WORKLOADS
               if (stage, workload) not in index]
    require(not missing, f"Incomplete figure grid: {missing}")

    result = {
        "schema_version": 1,
        "audit_file": audit_path.name, "audit_sha256": sha(audit_bytes),
        "experiment_plan_file": plan_path.name,
        "experiment_plan_sha256": sha(plan_bytes),
        "plot_script_sha256": sha(Path(__file__).read_bytes()),
        "role_order": list(ROLES), "stage_labels": labels,
        "stages": selections, "repeats_per_stage": args.repeats_per_stage,
        "timeline_workload": "p12k-burst", "workloads": list(WORKLOADS),
        "explicitly_excluded_reports": omitted,
        "diagnostic_reports_excluded_from_timing": diagnostics,
        "source_reports": [], "runs": [],
        "contract": {
            "selection": "Every registered repeat from every explicitly listed stage",
            "timeline": "Entire p12k-burst windows; all roles share x limits",
            "zero": "Each run's all-D-first-token gate; pre-gate data retained",
            "gaps": "Intact adjacent client token-event intervals, never clipped",
            "chunks": "One point per SSE token event, never one point per token",
            "ecdf": "Empirical cumulative fraction, all measured requests pooled",
            "scoped": (
                "Diagnostic ECDF compares per-request post-gate intact maxima "
                "with fully-contained maxima; it does not replace overall SLO"
            ),
            "warmup": "Audited but excluded from figures",
        },
    }
    plan_hashes = {workload: set() for workload in WORKLOADS}
    producer_hashes = set()
    measurement_contract = None
    reference_shapes = {}
    for role in ROLES:
        expected_role, expected_budget = ROLE_CONTRACT[role]
        for period, stage in enumerate(selections[role]):
            for workload in WORKLOADS:
                require((stage, workload) in index,
                        f"Incomplete figure grid: missing {stage}-{workload}")
                audited = index[stage, workload]
                parameters = audited["manifest_role"]
                require(parameters["role"] == expected_role
                        and parameters["token_budget"] == expected_budget,
                        f"Stage {stage} violates role/budget identity: {role}")
                manifest_path = directory / Path(audited["manifest"]).name
                manifest_bytes = manifest_path.read_bytes()
                require(sha(manifest_bytes) == audited["manifest_sha256"],
                        f"Manifest changed after audit: {manifest_path.name}")
                manifest = json.loads(manifest_bytes)
                argv = manifest["argv"]
                actual_budget = int(argv[argv.index("--max-num-batched-tokens") + 1])
                require(actual_budget == expected_budget,
                        "Manifest argv budget differs")
                if role == "guard":
                    policy = json.loads(argv[argv.index(
                        "--batch-admission-policy-config") + 1])
                    require(policy.get("decode_budget_guard") is True,
                            "Guard-labelled stage does not enable the guard")
                data, source = read_receipt(directory, audited)
                require(data["complete"] is True,
                        f"Incomplete report: {source['file']}")
                require(data["parameters"]["repeats"] == args.repeats_per_stage,
                        "Registered repeat count differs from selected count")
                require([run["repeat"] for run in data["repeats"]]
                        == list(range(args.repeats_per_stage)),
                        f"Missing/reordered repeats: {source['file']}")
                require(len(data["warmup"]) == data["parameters"]["warmup_repeats"]
                        and all(run["complete"] for run in data["warmup"]),
                        f"Warmup incomplete: {source['file']}")
                contract = data["measurement_contract"]
                if measurement_contract is None:
                    measurement_contract = contract
                require(contract == measurement_contract,
                        "Measurement contracts differ")
                plan_hashes[workload].add(data["plan_sha256"])
                producer_hashes.add(data["producer_sha256"])
                shape = {key: data["parameters"][key] for key in (
                    "decode_requests", "decode_input_tokens", "decode_output_tokens",
                    "prefill_requests", "prefill_input_tokens", "prefill_output_tokens",
                    "prefill_interval", "prefill_start_delay", "seed", "model",
                )}
                for key, value in registered_workloads[workload].items():
                    if key == "name":
                        continue
                    actual_key = key[:-2] if key.endswith("_s") else key
                    require(shape[actual_key] == value,
                            f"Registered workload changed: {workload}.{key}")
                require(shape == reference_shapes.setdefault(workload, shape),
                        f"Workload/arrival mismatch: {source['file']}")
                result["source_reports"].append({
                    **source, "role": role, "stage": stage, "workload": workload,
                    "manifest": manifest_path.name,
                    "manifest_sha256": sha(manifest_bytes),
                    "producer_sha256": data["producer_sha256"],
                    "plan_sha256": data["plan_sha256"],
                })
                for run in data["repeats"]:
                    require(run["complete"] and all(
                        row["transport_passed"] for row in run["requests"]
                    ), f"Failed repeat cannot be silently omitted: {source['file']}")
                    result["runs"].append({
                        "role": role, "stage": stage, "period": period,
                        "period_label": labels[period], "workload": workload,
                        "repeat": run["repeat"],
                        "started_monotonic_s": run["started_monotonic_s"],
                        "finished_monotonic_s": run["finished_monotonic_s"],
                        "gate_monotonic_s": run["decode_gate_monotonic_s"],
                        "requests": [compact_request(row) for row in run["requests"]],
                        "summary": run["summary"],
                    })
    require(all(len(values) == 1 for values in plan_hashes.values()),
            "Request plans differ across roles or periods")
    require(len(producer_hashes) == 1, "Producer hashes differ across plotted data")
    result["parameters_by_workload"] = reference_shapes
    result["producer_sha256"] = next(iter(producer_hashes))
    return result


def ecdf_points(values):
    values = sorted(values)
    return values, [(index + 1) / len(values) for index in range(len(values))]


def postgate_maximum(run, row, contained):
    times = [event["time"] for event in row["token_events"]]
    gate = run["gate_monotonic_s"]
    gaps = [right - left for left, right in zip(times, times[1:])
            if (left >= gate if contained else right > gate or left == gate)]
    return max(gaps) * 1000 if gaps else None


def draw(data, output, preview):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "svg.hashsalt": "muxpd-contention-v1", "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    generated = []
    preview.mkdir(parents=True, exist_ok=True)

    def save(fig, filename):
        path = output / filename
        fig.savefig(path, format="svg", metadata={
            "Date": None, "Creator": "plot_contention.py (audited client SSE)",
        })
        preview_path = preview / Path(filename).with_suffix(".png").name
        fig.savefig(preview_path, format="png", dpi=120)
        plt.close(fig)
        generated.append({
            "file": filename, "sha256": sha(path.read_bytes()),
            "local_preview": {"file": str(preview_path),
                              "sha256": sha(preview_path.read_bytes()),
                              "published": False,
                              "reason": "Local inspection; SVG is public"},
        })

    runs = [run for run in data["runs"] if run["workload"] == "p12k-burst"]
    rows = len(data["stage_labels"]) * data["repeats_per_stage"]
    fig, axes = plt.subplots(rows, 4, squeeze=False, figsize=(19, 2.25 * rows),
                             sharex=True, sharey=True)
    left = min(run["started_monotonic_s"] - run["gate_monotonic_s"] for run in runs)
    right = max(run["finished_monotonic_s"] - run["gate_monotonic_s"] for run in runs)
    for run in runs:
        col = ROLES.index(run["role"])
        index = run["period"] * data["repeats_per_stage"] + run["repeat"]
        ax = axes[index][col]
        gate = run["gate_monotonic_s"]
        for row in run["requests"]:
            if row["phase"] == "decode":
                y = row["index"]
                times = [event["time"] - gate for event in row["token_events"]]
                ax.vlines(times, y - 0.13, y + 0.13, color=COLORS[col],
                          linewidth=0.5, alpha=0.6)
                for event, timestamp in zip(row["token_events"], times):
                    if event["tokens"] > 1:
                        ax.plot(timestamp, y, marker="D", color="black", markersize=3)
                for a, b in zip(times, times[1:]):
                    if b - a > 0.1:
                        ax.plot([a, b], [y, y], linewidth=2.0,
                                color="#A42C35" if b - a > 0.2 else "#D4931B")
            else:
                y = 4 + row["index"]
                start = row["started_monotonic_s"] - gate
                first = row["first_token_monotonic_s"] - gate
                finish = row["finished_monotonic_s"] - gate
                ax.plot([start, first], [y, y], color="#7C4D9B", linewidth=3)
                ax.plot([first, finish], [y, y], color="#B9A5CB", linewidth=1)
                ax.plot(first, y, "o", color="#7C4D9B", markersize=3)
                ax.axvline(start, color="#7C4D9B", alpha=0.18, linewidth=0.6)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=0.7)
        ax.set_xlim(left, right)
        ax.set_ylim(6.6, -0.6)
        ax.set_yticks(range(7), ["D1", "D2", "D3", "D4", "P1", "P2", "P3"])
        ax.grid(axis="x", alpha=0.12)
        if index == 0:
            ax.set_title(LABELS[col], fontsize=11)
        if col == 0:
            ax.set_ylabel(f"{run['period_label']} / repeat {run['repeat'] + 1}")
        if index == rows - 1:
            ax.set_xlabel("Seconds relative to all-D-first-token gate")
    fig.suptitle("P12K burst: every measured repeat, complete client windows", y=0.995)
    fig.legend(handles=[
        Line2D([], [], color="#355C8C", marker="|", linestyle="None",
               label="Observed D token event"),
        Line2D([], [], color="#D4931B", linewidth=2, label="Gap > 100 ms"),
        Line2D([], [], color="#A42C35", linewidth=2, label="Gap > 200 ms"),
        Line2D([], [], color="#7C4D9B", linewidth=3, label="P dispatch to first token"),
        Line2D([], [], color="black", marker="D", linestyle="None",
               label="Multi-token event (not split)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "contention-timeline.svg")

    def distribution_figure(scoped):
        fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharey=True, squeeze=False)
        for col, workload in enumerate(WORKLOADS):
            for row_index in range(2):
                ax = axes[row_index][col]
                found = False
                eligible_total = 0
                for role, label, color in zip(ROLES, LABELS, COLORS):
                    values = []
                    eligible = 0
                    for run in data["runs"]:
                        if run["role"] != role or run["workload"] != workload:
                            continue
                        for request in run["requests"]:
                            phase = "decode" if scoped or row_index == 0 else "prefill"
                            if request["phase"] != phase:
                                continue
                            eligible += 1
                            if scoped:
                                value = postgate_maximum(run, request, row_index == 1)
                            else:
                                value = (request["metrics"]["token_event_gap_s"]["max"]
                                         if row_index == 0
                                         else request["metrics"]["ttft_s"])
                                value = value * 1000 if value is not None else None
                            if value is not None:
                                values.append(value)
                    if values:
                        x, y = ecdf_points(values)
                        ax.step([0, *x], [0, *y], where="post", color=color,
                                label=f"{label} (n={len(values)}/{eligible})",
                                linewidth=1.6)
                        ax.scatter(x, y, color=color, s=6)
                        found = True
                    elif eligible:
                        ax.plot([], [], color=color, label=f"{label} (n=0/{eligible})")
                    eligible_total += eligible
                if not found:
                    ax.text(0.5, 0.5, "No observable gaps" if eligible_total
                            else "No prefill requests", ha="center", va="center",
                            transform=ax.transAxes)
                else:
                    for limit in ((100, 200) if scoped or row_index == 0
                                  else (1000, 2000)):
                        ax.axvline(limit, linestyle="--", color="#777777",
                                   linewidth=0.6, alpha=0.6)
                    ax.legend(fontsize=6, frameon=False, loc="lower right")
                ax.set_ylim(0, 1.04)
                ax.set_xlim(left=0)
                ax.grid(alpha=0.12)
                if row_index == 0:
                    ax.set_title(workload)
                if scoped:
                    ax.set_xlabel("Post-gate intact max gap (ms)" if row_index == 0
                                  else "Post-gate contained max gap (ms)")
                else:
                    ax.set_xlabel("Whole-request max D event gap (ms)" if row_index == 0
                                  else "Injected P TTFT (ms)")
                if col == 0:
                    ax.set_ylabel("Empirical cumulative fraction")
        fig.suptitle(
            "Diagnostic scopes: intact boundary-crossing gaps; overall SLO unchanged"
            if scoped else "All measured requests: startup retained, warmup excluded",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save(fig, "contention-scoped-gaps.svg" if scoped else "contention-ecdf.svg")

    distribution_figure(False)
    distribution_figure(True)
    return {"matplotlib_version": matplotlib.__version__, "figures": generated}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=base)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preview-dir", type=Path,
                        help="Local PNG previews, excluded from public SVG evidence")
    for role in ROLES:
        parser.add_argument("--" + role, required=True,
                            help="Comma-separated stage names")
    parser.add_argument("--stage-labels", default="initial,confirmation")
    parser.add_argument("--repeats-per-stage", type=int, default=3)
    parser.add_argument("--exclude-stage", action="append", default=[],
                        metavar="NAME=WHY")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir or args.input_dir / "figures"
    preview = args.preview_dir or args.input_dir / "plot-previews"
    require(preview.resolve() != output.resolve(),
            "PNG previews must be separate from public figures")
    try:
        data = load_inputs(args)
    except (OSError, ValueError, KeyError) as error:
        print(f"No plots generated: {error}", file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "plot-inputs.json"
    inputs.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version, "argv": sys.argv,
        "input_validation_passed": True, "figures_generated": False,
        "plot_inputs": inputs.name, "plot_inputs_sha256": sha(inputs.read_bytes()),
        "plot_script_sha256": data["plot_script_sha256"],
        "audit_sha256": data["audit_sha256"], "source_reports": data["source_reports"],
    }
    manifest_path = output / "plot-manifest.json"
    try:
        if not args.validate_only:
            manifest.update(draw(data, output, preview))
            manifest["figures_generated"] = True
    except (ImportError, RuntimeError, ValueError) as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in ("source_reports", "argv")}, indent=2))
    return 1 if "error" in manifest else 0


if __name__ == "__main__":
    raise SystemExit(main())
