"""Extract TP8 lifecycle evidence from completed diagnostics, without device access.

Cohort IDs, step IDs, layer quotas, progress and debt-pair counts are discovered
from this run. Nothing is copied from a TP4 proof. Run after the driver finishes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata(path):
    data = path.read_bytes()
    return {
        "file": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument("--min-decode-steps", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    traces = (
        [args.trace]
        if args.trace
        else sorted(args.input_dir.glob("work256-traced-*.jsonl"))
    )
    require(
        len(traces) == 1, "Require exactly one engine trace, or select it with --trace"
    )
    trace = traces[0]
    trace_before = metadata(trace)
    names = ("traced-progress.json", "traced-cancel.json", "traced-check.json")
    reports = {name: json.loads((args.input_dir / name).read_text()) for name in names}
    require(
        all(report.get("passed") is True for report in reports.values()),
        "A diagnostic did not pass",
    )
    root = Path(__file__).resolve().parents[2]
    checker_path = root / "scripts/check_decoupled_trace.py"
    spec = importlib.util.spec_from_file_location("tp8_trace_checker", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checked = checker.check([trace], True, args.min_decode_steps, args.expected_layers)
    require(
        json.dumps(checked, sort_keys=True)
        == json.dumps(reports["traced-check.json"], sort_keys=True),
        "Stored checker report differs from a fresh check of the selected trace",
    )
    raw = trace.read_text().splitlines()
    events, ordered = {}, defaultdict(list)
    selected = set()
    for line_number, line in enumerate(raw, 1):
        row = json.loads(line)
        key = row["pid"], row["step"]
        group = events.setdefault(key, {})
        require(row["event"] not in group, f"Duplicate event at {key}")
        group[row["event"]] = (line_number, row)
    for key, group in events.items():
        require(
            set(group) == {"schedule", "decoupled_schedule", "decoupled_complete"},
            f"Incomplete step {key}",
        )
        ordered[key[0]].append(key)
    for keys in ordered.values():
        keys.sort(key=lambda key: key[1])

    def row(key, kind):
        return events[key][kind][1]

    def keep(keys):
        numbers = sorted(number for key in keys for number, _ in events[key].values())
        selected.update(numbers)
        return numbers

    def later(key):
        keys = ordered[key[0]]
        return keys[keys.index(key) + 1 :]

    def probe(name, cancelled):
        client = reports[name]
        start = client["prefill_started_monotonic"]
        end = client["decode"]["start_monotonic"] + client["decode"]["duration_s"]
        candidates = []
        for key in events:
            schedule = row(key, "decoupled_schedule")
            native = row(key, "schedule")
            if start <= schedule["time"] <= end:
                for rid in schedule["new_prefill_ids"]:
                    if native["scheduled_tokens"].get(rid) == 12000:
                        candidates.append((key, rid))
        require(
            len(candidates) == 1,
            f"{name}: expected one fresh 12K P in the client interval, found {len(candidates)}",
        )
        first_key, rid = candidates[0]
        first = row(first_key, "decoupled_schedule")
        native = row(first_key, "schedule")
        require(
            native["expected_tokens"][rid] == 12000
            and native["computed_before"][rid] == 0,
            f"{name}: not an entire fresh 12K prompt",
        )
        if not cancelled:
            require(
                client["prefill"]["usage"]["prompt_tokens"] == 12000,
                f"{name}: client prompt usage differs",
            )
            require(
                client["prefill"]["request_id"] in rid,
                f"{name}: client request identity differs",
            )
        keys = [
            key
            for key in ordered[first_key[0]]
            if row(key, "decoupled_schedule")["cohort_id"] == first["cohort_id"]
        ]
        previous = 0
        progress, decode_steps, retirement = [], [], None
        for key in keys:
            schedule, receipt = (
                row(key, "decoupled_schedule"),
                row(key, "decoupled_complete"),
            )
            require(
                schedule["layers_completed_before"] == previous,
                f"{key}: missing progress predecessor",
            )
            current = receipt["layers_completed"]
            require(
                previous <= current <= args.expected_layers,
                f"{key}: nonmonotonic progress",
            )
            if key != first_key:
                require(
                    rid not in row(key, "schedule")["scheduled_tokens"],
                    f"{key}: P tokens readmitted",
                )
            require(rid not in schedule["decode_ids"], f"{key}: pending P entered D")
            progress.append(
                {
                    "step": key[1],
                    "before": previous,
                    "quota": schedule["layers_per_slice"],
                    "completed": current,
                    "actual_delta": current - previous,
                    "source": receipt["progress_source"],
                }
            )
            if (
                receipt["decode_ids"]
                and rid not in receipt["completed_prefill_ids"]
                and rid not in receipt["cancelled_prefill_ids"]
            ):
                require(
                    set(receipt["decode_ids"]) == set(schedule["decode_ids"]),
                    f"{key}: D receipt differs",
                )
                for decode_id in receipt["decode_ids"]:
                    require(
                        row(key, "schedule")["request_phases"][decode_id] == "decode",
                        f"{key}: D phase mismatch",
                    )
                    require(
                        row(key, "schedule")["scheduled_tokens"][decode_id] == 1,
                        f"{key}: non-unit D advance",
                    )
                decode_steps.append(key[1])
            if (
                rid in receipt["completed_prefill_ids"]
                or rid in receipt["cancelled_prefill_ids"]
            ):
                require(retirement is None, f"{key}: duplicate P retirement")
                retirement = key
            previous = current
        require(retirement is not None, f"{name}: no P retirement")
        final = row(retirement, "decoupled_complete")
        if cancelled:
            require(
                rid in final["cancelled_prefill_ids"]
                and rid not in final["completed_prefill_ids"],
                f"{name}: no cancellation ACK",
            )
            require(
                rid in row(retirement, "decoupled_schedule")["cancelled_prefill_ids"],
                f"{name}: ACK lacks cancel control",
            )
        else:
            require(
                rid in final["completed_prefill_ids"]
                and previous == args.expected_layers,
                f"{name}: no completed full P",
            )
            require(
                len(decode_steps) >= args.min_decode_steps,
                f"{name}: no required D progress before P completion",
            )
        after = later(retirement)
        require(bool(after), f"{name}: no following step to verify ownership cleanup")
        following = row(after[0], "decoupled_schedule")
        require(
            rid not in following["pending_prefill_ids"]
            and following["cohort_id"] != first["cohort_id"],
            f"{name}: cohort ownership retained after ACK",
        )
        return {
            "client_report": name,
            "pid": first_key[0],
            "cohort_id": first["cohort_id"],
            "request_id": rid,
            "full_prompt_tokens": 12000,
            "progress": progress,
            "decode_receipt_steps_before_retirement": decode_steps,
            "retirement_step": retirement[1],
            "retirement": "cancelled" if cancelled else "completed",
            "next_step_ownership_cleared": after[0][1],
            "source_lines": keep(keys + [after[0]]),
        }

    completed = probe("traced-progress.json", False)
    cancelled = probe("traced-cancel.json", True)
    pairs = []
    for key in events:
        schedule = row(key, "decoupled_schedule")
        admission = schedule.get("admission") or {}
        if admission.get("reason") not in {
            "token_budget_prefill_first",
            "deadline_prefill_first",
            "full_budget_prefill_first",
        }:
            continue
        deferred = set(admission["deferred_decode_ids"])
        require(
            deferred and not schedule["decode_ids"], f"{key}: P-first did not defer D"
        )
        require(
            not deferred.intersection(row(key, "schedule")["scheduled_tokens"]),
            f"{key}: deferred D token count advanced",
        )
        following_keys = later(key)
        require(bool(following_keys), f"{key}: no step after P-first")
        following_key = following_keys[0]
        following = row(following_key, "decoupled_schedule")
        native_next = row(following_key, "schedule")
        terminal = deferred.intersection(native_next["finished_req_ids"])
        preempted = deferred.intersection(native_next["preempted_req_ids"])
        available = deferred - terminal - preempted
        require(
            available <= set(row(key, "decoupled_complete")["decode_debt_ids"]),
            f"{key}: available deferred D lacks debt",
        )
        if available:
            require(
                following["admission"]["reason"] == "repay_decode"
                and available <= set(following["decode_ids"]),
                f"{key}: next nonterminal/nonpreempted D not repaid; inspect eligibility before drawing a fairness conclusion",
            )
            require(
                not available.intersection(
                    row(following_key, "decoupled_complete")["decode_debt_ids"]
                ),
                f"{key}: serviced D debt remains",
            )
        # Check native computed-token continuity using the last actual schedule
        # of each deferred request. A previous step may itself be full prefill.
        checks, predecessors = {}, []
        for rid in sorted(available):
            earlier = [
                candidate
                for candidate in ordered[key[0]]
                if candidate[1] < key[1]
                and rid in row(candidate, "schedule")["scheduled_tokens"]
            ]
            require(bool(earlier), f"{key}: no previous token counter for {rid}")
            predecessor = earlier[-1]
            previous_native = row(predecessor, "schedule")
            expected = (
                previous_native["computed_before"][rid]
                + previous_native["scheduled_tokens"][rid]
            )
            require(
                native_next["computed_before"][rid] == expected,
                f"{key}: deferred D counter advanced without a scheduled token",
            )
            predecessors.append(predecessor)
            checks[rid] = {
                "previous_step": predecessor[1],
                "expected_computed_before_repayment": expected,
                "actual_computed_before_repayment": native_next["computed_before"][rid],
            }
        pairs.append(
            {
                "pid": key[0],
                "reason": admission["reason"],
                "prefill_first_step": key[1],
                "following_step": following_key[1],
                "repaid_ids": sorted(available),
                "finished_ids": sorted(terminal),
                "preempted_ids": sorted(preempted),
                "counter_checks": checks,
                "source_lines": keep(list(set(predecessors + [key, following_key]))),
            }
        )
    require(
        metadata(trace) == trace_before,
        "Trace changed during the audit; rerun after diagnostics finish",
    )
    result = {
        "schema_version": 1,
        "passed": True,
        "scope": "Host scheduling, full-token admission, layer receipts, cancellation ACK and recorded decode-debt repayment. Does not prove device overlap, graph/FX execution or KV tensor contents.",
        "selection": "Fresh 12000-token P matched to each client probe time window; cohort and steps discovered from this trace. Every P-first admission audited. Original JSONL text excludes terminal newline; line numbers are 1-based.",
        "trace": trace_before,
        "trace_lines": len(raw),
        "inputs": [metadata(args.input_dir / name) for name in names],
        "audit_script": metadata(Path(__file__)),
        "checker_source": metadata(checker_path),
        "checker_summary": checked,
        "completed_long_prefill": completed,
        "cancelled_long_prefill": cancelled,
        "prefill_first_pairs": pairs,
        "prefill_first_pair_count": len(pairs),
        "prefill_first_scope": "No P-first observation is not a proof of fairness. Terminal/preempted IDs are reported separately; successful immediate-next-step D receipts prove service for the remaining recorded IDs.",
        "performance_scope": "Trace-on diagnostics and prior pure-P calibration are excluded from throughput comparisons. Actual quotas are observed after calibration, not assumed from TP4 or guaranteed on a cold start.",
        "original_trace_lines": [
            {"line_number": number, "raw_line": raw[number - 1]}
            for number in sorted(selected)
        ],
    }
    output = args.output or args.input_dir / "lifecycle-proof.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": True,
                "completed_progress": completed["progress"],
                "cancelled_progress": cancelled["progress"],
                "prefill_first_pairs": len(pairs),
                "original_lines": len(selected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
