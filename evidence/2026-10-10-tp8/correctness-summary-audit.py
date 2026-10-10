#!/usr/bin/env python3
"""Read only TP8 client reports; write compact diagnostics, never call a server."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COMMIT = json.loads((OUT / "native-initial-manifest.json").read_text())["source_commit"]


def sha(content):
    return hashlib.sha256(content).hexdigest()


def helper(filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = subprocess.check_output(
        ["git", "show", f"{COMMIT}:scripts/{filename}"], cwd=ROOT
    )
    assert sha(path.read_bytes()) == sha(committed), filename
    return module, {
        "path": f"scripts/{filename}", "sha256": sha(committed),
        "source_commit": COMMIT,
    }


NATURAL, NATURAL_PRODUCER = helper("compare_outputs.py")
SYNTHETIC, SYNTHETIC_PRODUCER = helper("check_decode_graph_serving.py")
STATE, STATE_PRODUCER = helper("probe_graph_state_reuse.py")


def manifest_provenance(filename, output_sha):
    if filename.startswith("native-eager-"):
        manifest_name = "native-eager-manifest.json"
    elif filename == "native-initial-vs-bracket-correctness.json":
        manifest_name = "native-bracket-manifest.json"
    elif filename.startswith(("native-vs-work256", "work256-")):
        manifest_name = "work256-manifest.json"
    else:
        manifest_name = "native-initial-manifest.json"
    content = (OUT / manifest_name).read_bytes()
    manifest = json.loads(content)
    assert manifest["source_commit"] == COMMIT
    args = manifest["api_argv"]
    assert args[args.index("--tensor-parallel-size") + 1] == "8"
    assert "--enable-expert-parallel" in args
    matches = [row for row in manifest["client_runs"]
               if Path(row["output"]).name == filename]
    if not matches and "post_timing_correctness" in manifest:
        row = manifest["post_timing_correctness"]
        if Path(row["argv"][row["argv"].index("--output") + 1]).name == filename:
            matches.append(row)
    assert len(matches) == 1
    assert matches[0]["returncode"] == 0 and matches[0]["output_sha256"] == output_sha
    return {
        "path": str((OUT / manifest_name).relative_to(ROOT)),
        "sha256_at_read": sha(content), "source_commit": COMMIT,
        "api_pid": manifest["api_pid"], "tensor_parallel_size": 8,
        "expert_parallel": True, "enforce_eager": "--enforce-eager" in args,
        "muxpd_worker": any("vllm_muxpd" in value for value in args),
        "client_returncode": 0, "recorded_output_sha256_matches": True,
    }


def load(filename, producer):
    path = OUT / filename
    if not path.exists():
        return None
    content = path.read_bytes()
    report = json.loads(content)
    if not report.get("complete"):
        return None
    embedded = report.get("producer_sha256")
    if embedded is not None:
        assert embedded == producer["sha256"], filename
    return report, {
        "path": str(path.relative_to(ROOT)), "sha256": sha(content),
        "size_bytes": len(content), "producer": producer,
        "service_manifest": manifest_provenance(filename, sha(content)),
        "producer_sha_embedded": embedded is not None,
        "producer_provenance": (
            "embedded report hash matches immutable source commit and current helper"
            if embedded else
            "report has no producer hash; helper hash reconstructed from recorded service commit and checked against current helper"
        ),
    }


def endpoint(report, label):
    selected = [row for row in report["endpoints"] if row["label"] == label]
    assert len(selected) == 1 and not selected[0].get("error")
    return selected[0]


def verify_reused_baseline(report, source, filename):
    baseline = endpoint(report, "baseline")
    content = (OUT / filename).read_bytes()
    assert baseline["reused_report_sha256"] == sha(content)
    assert baseline["requests"] == endpoint(json.loads(content), "baseline")["requests"]
    source["baseline_reuse"] = {
        "path": str((OUT / filename).relative_to(ROOT)),
        "sha256": sha(content), "requests_identical_to_original_report": True,
    }


def natural(filename, label):
    item = load(filename, NATURAL_PRODUCER)
    if item is None:
        return None
    report, source = item
    if label == "candidate":
        verify_reused_baseline(report, source, "native-correctness.json")
    ep = endpoint(report, label)
    rows = ep["requests"]
    cases = {case["name"]: case for case in report["cases"]}
    expected = {(repeat, mode, case) for repeat in range(2)
                for mode in ("isolated", "paired") for case in cases}
    assert {(row["repeat"], row["mode"], row["case"]) for row in rows} == expected
    assert len(rows) == 16
    for row in rows:
        assert not row.get("error") and row["done"] and row["finish_reason"]
        assert row["usage"] and row["text_sha256"] == sha(row["text"].encode())
        assert all("error" not in event["data"] for event in row["events"])
        assembled = "".join(choice.get("delta", {}).get("content") or ""
                            for event in row["events"]
                            for choice in event["data"].get("choices", []))
        assert assembled == row["text"]
        assert NATURAL.check_oracle(row["text"], cases[row["case"]]["oracle"]) == row["oracle_passed"]
    summary = NATURAL.summarize(report["endpoints"])
    assert summary == report["summary"]
    own = summary["endpoints"][label]
    cross = summary["cross_endpoint_text_matches"]
    return {
        "source": source, "endpoint": label, "request_count": len(rows),
        "sse_completion_receipts": len(rows),
        "transport_scope": "Validated saved event/text hashes, no error, DONE/finish/usage flags; this older producer does not persist raw SSE bodies or HTTP status.",
        "oracle_passed": sum(row["oracle_passed"] for row in rows),
        "repeat_groups": len(own["repeat_stability"]),
        "unstable_repeat_groups": [row for row in own["repeat_stability"] if row["distinct_texts"] != 1],
        "isolated_paired_matches": sum(row["exact_text_match"] for row in own["isolated_vs_paired_text_matches"]),
        "isolated_paired_comparisons": len(own["isolated_vs_paired_text_matches"]),
        "cross_endpoint_matches": sum(row["exact_text_match"] for row in cross),
        "cross_endpoint_comparisons": len(cross),
        "cross_endpoint_mismatches": [row for row in cross if not row["exact_text_match"]],
    }


def synthetic(filename, label):
    item = load(filename, SYNTHETIC_PRODUCER)
    if item is None:
        return None
    report, source = item
    if label == "candidate":
        verify_reused_baseline(report, source, "native-serving.json")
    ep = endpoint(report, label)
    cases = {case["name"]: case for case in report["cases"]}
    rows = ep["requests"]
    expected = {(repeat, case["group"], case["name"])
                for repeat in range(2) for case in report["cases"]}
    assert len(rows) == 56 and {(r["repeat"], r["mode"], r["case"]) for r in rows} == expected
    for row in rows:
        assert row["transport_passed"] and row["http_status"] == 200 and not row.get("error")
        raw = row["sse_body"]
        assert sha(raw.encode()) == row["sse_body_sha256"]
        replay = SYNTHETIC.CompletionSSE()
        replay.feed(raw, 0)
        replay.finish(cases[row["case"]])
        rebuilt = replay.receipt()
        for field in ("text", "text_sha256", "usage", "finish_reason", "response_id", "done_received"):
            assert row[field] == rebuilt[field], (filename, row["case"], field)
    groups = report["summary"]["repeat_stability"][label]
    for group in groups:
        own = [row for row in rows if row["case"] == group["case"] and row["mode"] == group["mode"]]
        assert len(own) == group["samples"] == 2
        assert len({row["text"] for row in own}) == group["distinct_texts"]
    cross = report["summary"]["cross_endpoint_text_matches"]
    if cross:
        baseline = {(r["repeat"], r["mode"], r["case"]): r for r in endpoint(report, "baseline")["requests"]}
        for match in cross:
            key = (match["repeat"], match["mode"], match["case"])
            other = next(r for r in rows if (r["repeat"], r["mode"], r["case"]) == key)
            assert match["exact_text_match"] == (baseline[key]["text"] == other["text"])
    return {
        "source": source, "endpoint": label, "request_count": len(rows),
        "transport_passed": len(rows), "raw_sse_replayed": len(rows),
        "transport_scope": "HTTP 200, raw SSE SHA and replay, constant response identity, DONE, length finish, exact prompt/output usage, reconstructed text.",
        "repeat_groups": len(groups),
        "unstable_repeat_groups": [row for row in groups if row["distinct_texts"] != 1],
        "cross_endpoint_matches": sum(row["exact_text_match"] for row in cross),
        "cross_endpoint_comparisons": len(cross),
        "cross_endpoint_mismatches": [row for row in cross if not row["exact_text_match"]],
        "generated_token_identity_available": False,
        "device_graph_execution_proven": report["summary"]["device_graph_execution_proven"],
        "actual_decode_batch_sizes_proven": report["summary"]["actual_decode_batch_sizes_proven"],
    }


def state(filename):
    item = load(filename, STATE_PRODUCER)
    if item is None:
        return None
    report, source = item
    rows = report["requests"]
    assert report["transport_passed"] and len(rows) == report["expected_requests"] == 30
    for index, row in enumerate(rows):
        assert row["sequence_index"] == index and row["transport_passed"] and row["http_status"] == 200
        receipt = STATE.validate_response(row["response"], row["payload"])
        assert all(row[key] == value for key, value in receipt.items())
        if index:
            assert rows[index - 1]["finished_monotonic_s"] <= row["started_monotonic_s"]
    summary = STATE.summarize(rows, report["parameters"]["repeats"])
    assert summary == report["summary"]
    compact = []
    for case in report["cases"]:
        own = [row for row in rows if row["case"] == case["name"] and row["stage"] != "background"]
        compact.append({
            "case": case["name"], "prompt_token_ids": case["prompt"],
            "max_tokens": case["max_tokens"],
            "first_generated_token_stable_across_all_four": len({row["token_ids"][0] for row in own}) == 1,
            "full_generated_sequence_stable_across_all_four": len({tuple(row["token_ids"]) for row in own}) == 1,
            "samples": [{"stage": row["stage"], "repeat": row["repeat"], "token_ids": row["token_ids"]} for row in own],
            "bracket_summary": next(row for row in summary if row["case"] == case["name"]),
        })
    return {
        "source": source, "transport_passed": len(rows), "request_count": len(rows),
        "target_requests": 24, "long_prefill_decode_brackets": 6,
        "strict_serial_order_verified": True, "cases": compact,
        "background_prompt_tokens": report["parameters"]["long_prompt_tokens"],
        "background_output_tokens": report["parameters"]["long_max_tokens"],
        "prompt_length_group_stability": {
            str(length): {
                "case_count": len(selected),
                "first_token_stable_case_count": sum(case["first_generated_token_stable_across_all_four"] for case in selected),
                "full_sequence_stable_case_count": sum(case["full_generated_sequence_stable_across_all_four"] for case in selected),
            }
            for length in (1, 2)
            if (selected := [case for case in compact if len(case["prompt_token_ids"]) == length])
        },
    }


def main():
    results = {
        "native_graph": {
            "natural": natural("native-correctness.json", "baseline"),
            "synthetic": synthetic("native-serving.json", "baseline"),
            "state_reuse": state("native-state-reuse.json"),
        },
        "muxpd_graph_work256": {
            "natural": natural("native-vs-work256.json", "candidate"),
            "synthetic": synthetic("native-vs-work256-serving.json", "candidate"),
            "state_reuse": state("work256-state-reuse.json"),
        },
        "native_graph_bracket": {
            "natural": natural("native-initial-vs-bracket-correctness.json", "candidate"),
        },
        "native_eager": {"state_reuse": state("native-eager-state-reuse.json")},
    }
    pending = [f"{service}.{kind}" for service, tests in results.items()
               for kind, value in tests.items() if value is None]
    report = {
        "schema_version": 1, "scope": "Only the current TP8 experiment; no TP4 samples pooled",
        "complete": not pending, "pending_reports": pending,
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "reproduce": "../.venv/bin/python local-results/tp8/correctness-summary-audit.py (from repository root)",
        "results": results,
        "limitations": [
            "Natural and synthetic serving exactness compares decoded text, not generated token IDs; state-reuse diagnostics compare returned token IDs.",
            "Synthetic prompt IDs are diagnostic inputs, not a natural-language quality benchmark.",
            "One-token prompt first-token divergence in a native control prevents attribution to the plugin alone; it does not establish a unique root cause or rule out additional plugin defects.",
            "The serial state probe's before phase is not a fresh-process or cold-cache test.",
            "Transport success does not prove graph replay, actual device batch sizes, P/D overlap, or token-level equivalence.",
        ],
    }
    (OUT / "correctness-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# TP8 correctness diagnostics", "", "This report uses only this TP8 run. " + ("All planned reports are complete." if not pending else "Pending: " + ", ".join(pending) + "."), "",
             "| Service | Natural oracle / receipts | Natural exact vs native | Synthetic transport | Synthetic unstable repeat groups | Synthetic exact vs native | State transport |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for service, tests in results.items():
        n, s, t = tests.get("natural"), tests.get("synthetic"), tests.get("state_reuse")
        lines.append(f"| {service} | {str(n['oracle_passed']) + '/' + str(n['request_count']) if n else '—'} | {str(n['cross_endpoint_matches']) + '/' + str(n['cross_endpoint_comparisons']) if n and n['cross_endpoint_comparisons'] else '—'} | {str(s['transport_passed']) + '/' + str(s['request_count']) if s else '—'} | {str(len(s['unstable_repeat_groups'])) + '/' + str(s['repeat_groups']) if s else '—'} | {str(s['cross_endpoint_matches']) + '/' + str(s['cross_endpoint_comparisons']) if s and s['cross_endpoint_comparisons'] else '—'} | {str(t['transport_passed']) + '/' + str(t['request_count']) if t else '—'} |")
    lines += ["", "State-reuse rows use actual generated token IDs: four serial target samples per case (two before and two after a long P+D request). The six brackets add six background requests, for 30 requests per service.", "",
              "| Service | Case | First generated token stable across all four | Entire generated sequence stable across all four |", "| --- | --- | --- | --- |"]
    for service, tests in results.items():
        if tests.get("state_reuse"):
            for case in tests["state_reuse"]["cases"]:
                lines.append(f"| {service} | {case['case']} | {case['first_generated_token_stable_across_all_four']} | {case['full_generated_sequence_stable_across_all_four']} |")
    candidate_state = results["muxpd_graph_work256"]["state_reuse"]
    if candidate_state is not None:
        single = candidate_state["prompt_length_group_stability"]["1"]
        lines += ["", f"In the MuxPD sample, {single['full_sequence_stable_case_count']}/{single['case_count']} one-token-prompt cases keep the entire generated sequence identical across all four before/after samples. This differs from the native graph sample; it is an observation on these finite probes, not proof of a general fix or that the candidate tokens are uniquely correct."]
    eager_state = results["native_eager"]["state_reuse"]
    if eager_state is not None:
        single = eager_state["prompt_length_group_stability"]["1"]
        lines += ["", f"The native eager control keeps the first generated token stable in {single['first_token_stable_case_count']}/{single['case_count']} one-token-prompt cases; {single['full_sequence_stable_case_count']}/{single['case_count']} keep the entire output stable. `original_one_token_max16` differs in a later token during its before repeats. Thus first-token instability is observed in the native graph service in this sample, while later sequence divergence also occurs with native eager execution. This separates the observations without identifying a unique underlying cause."]
    lines += ["", "The native TP8 graph control has one-token-prompt first-token instability. For the two-token prompt, the first generated token and max1 output are stable in the completed graph reports, while the max16 output can diverge later in the sequence. Inspect the per-service table above; this report makes no claim that such later divergence is new at TP8. Native instability prevents a plugin-only attribution, but does not prove the root cause or excuse candidate-specific differences.", "",
              "All 56 synthetic SSE bodies per available endpoint are re-parsed, with saved-body SHA, HTTP 200, response identity, DONE, finish reason and exact usage checked. State responses are validated against their exact prompt IDs and requested output lengths, and strictly serial timestamps are checked. Natural reports retain events and completion flags but no raw SSE body or HTTP-status field; this narrower evidence is stated separately in JSON.", "",
              "Producer hashes: `compare_outputs.py` " + NATURAL_PRODUCER["sha256"] + " (reconstructed from service commit; older report has no embedded hash); `check_decode_graph_serving.py` " + SYNTHETIC_PRODUCER["sha256"] + "; `probe_graph_state_reuse.py` " + STATE_PRODUCER["sha256"] + ". The latter two are also embedded in their reports. Every report SHA is checked against its service manifest's successful client receipt, every manifest is pinned to commit `" + COMMIT + "`, and copied baseline requests/hashes are checked against the original report. JSON retains those manifest hashes, input-report SHA-256, byte sizes, all mismatch keys and compact state token sequences.", "",
              "Reproduce offline with the workspace Python: `python local-results/tp8/correctness-summary-audit.py`. This audit never sends requests or imports torch. These diagnostics do not establish actual device graph execution, hardware overlap, overall quality, or performance.", ""]
    (OUT / "correctness-summary.md").write_text("\n".join(lines))
    print(json.dumps({"complete": report["complete"], "pending": pending}, indent=2))


if __name__ == "__main__":
    main()
