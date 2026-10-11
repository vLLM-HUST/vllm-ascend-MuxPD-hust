"""Run an already-launched guard service's diagnostics, excluded from timing.

This controller never launches or stops a service. --plan-only performs no HTTP
requests and writes nothing. Producers are pinned and copied before execution;
all request/trace failures remain in a separate diagnostic receipt.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PYTHON = "/root/workspace/.venv/bin/python"
BENCHMARK_SHA = "6f373ff7383ccc61579d4e9406ee0c9b9523ee8a4ef0d5330d76471e421f6712"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def metadata(path):
    data = Path(path).read_bytes()
    return {
        "path": str(Path(path).resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def client_commands(args, producer_dir):
    def command(script, suffix, *extra):
        output = OUT / f"{args.stage}-{suffix}.json"
        return {
            "name": suffix,
            "output": str(output),
            "argv": [
                PYTHON,
                str(producer_dir / script),
                *extra,
                "--output",
                str(output),
            ],
        }

    return [
        command("exercise_decoupled.py", "progress", "--base-url", args.base_url),
        command(
            "exercise_decoupled.py",
            "cancel",
            "--base-url",
            args.base_url,
            "--cancel-prefill",
            "--cancel-after",
            "0.4",
        ),
        command(
            "benchmark_contention.py",
            "full-budget",
            "--base-url",
            args.base_url,
            "--label",
            args.stage + "-diagnostic-full-budget",
            "--decode-requests",
            "4",
            "--decode-input-tokens",
            "128",
            "--decode-output-tokens",
            "256",
            "--prefill-requests",
            "1",
            "--prefill-input-tokens",
            "16383",
            "--prefill-output-tokens",
            "1",
            "--warmup-repeats",
            "1",
            "--repeats",
            "1",
            "--prefill-start-delay",
            "0.2",
        ),
        command(
            "compare_outputs.py",
            "correctness",
            "--baseline-report",
            str(args.baseline_report),
            "--candidate-url",
            args.base_url,
            "--repeats",
            "2",
            "--require-exact",
        ),
    ]


def alive(pid):
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"


def wait_health(client, base, pid, timeout):
    started = time.monotonic()
    last_error = None
    while time.monotonic() - started < timeout:
        require(alive(pid), f"API {pid} exited before readiness")
        try:
            response = client.get(base + "/health")
            response.raise_for_status()
            return {"passed": True, "at": now(), "status_code": response.status_code}
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"Health did not become ready: {last_error}")


def run_command(command, report, receipt_path, *, stdout_json=False):
    output = Path(command["output"])
    log = output.with_suffix(".client.log")
    require(not output.exists() and not log.exists(), f"Refusing to overwrite {output}")
    producer = Path(command["argv"][1])
    row = {
        **command,
        "started_at": now(),
        "started_monotonic_s": time.monotonic(),
        "producer": metadata(producer),
        "diagnostic_only": True,
    }
    report["commands"].append(row)
    save(receipt_path, report)
    try:
        if stdout_json:
            with output.open("wb") as stdout, log.open("wb") as stderr:
                result = subprocess.run(
                    command["argv"],
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=900,
                    check=False,
                )
        else:
            with log.open("wb") as stream:
                result = subprocess.run(
                    command["argv"],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                    check=False,
                )
        row["returncode"] = result.returncode
        if result.returncode:
            row["error"] = f"Producer exited with code {result.returncode}"
    except Exception as exc:
        row["returncode"] = None
        row["error"] = f"{type(exc).__name__}: {exc}"
    row.update(finished_at=now(), finished_monotonic_s=time.monotonic())
    row["producer_unchanged"] = metadata(producer) == row["producer"]
    row["client_log"] = metadata(log) if log.exists() else None
    row["output_file"] = metadata(output) if output.exists() else None
    if row.get("returncode") == 0:
        try:
            json.loads(output.read_text())
        except Exception as exc:
            row["error"] = f"Missing/invalid JSON output: {type(exc).__name__}: {exc}"
    if not row["producer_unchanged"]:
        row["error"] = "Frozen producer changed while executing"
    save(receipt_path, report)
    print(
        command["name"], row.get("returncode"), row.get("error", "complete"), flush=True
    )
    return row


def stable_traces(pattern):
    paths = sorted(OUT.glob(pattern))
    require(paths, f"No diagnostic trace matches {pattern}")
    before = [metadata(path) for path in paths]
    for _ in range(10):
        time.sleep(1)
        after = [metadata(path) for path in paths]
        if after == before:
            return paths, after
        before = after
    raise RuntimeError("Diagnostic traces continued changing after clients completed")


def lifecycle_proof(
    paths, edge_report, progress_report, cancel_report, expected_layers
):
    """Discover actual cohorts/steps, retaining original JSONL lines and hashes."""
    sources, events, refs = [], {}, {}
    for path in paths:
        sources.append(metadata(path))
        for number, raw in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(raw)
            key = row["pid"], row["step"]
            name = row["event"]
            require(name not in events.setdefault(key, {}), f"Duplicate {key}/{name}")
            events[key][name] = row
            refs[(key, name)] = {"file": path.name, "line": number, "raw": raw}
    for key, group in events.items():
        require(
            set(group) == {"schedule", "decoupled_schedule", "decoupled_complete"},
            f"Incomplete trace transaction {key}",
        )
    keys = sorted(events)
    kept = set()

    def keep(key):
        kept.update((key, name) for name in events[key])

    def find_prefill(tokens, start, end, response_id=None):
        found = []
        for key in keys:
            native = events[key]["schedule"]
            schedule = events[key]["decoupled_schedule"]
            if not start <= schedule["time"] <= end:
                continue
            for rid in schedule["new_prefill_ids"]:
                if native["scheduled_tokens"].get(rid) == tokens and (
                    response_id is None
                    or rid == response_id
                    or rid.startswith(response_id + "-")
                ):
                    found.append((key, rid))
        require(len(found) == 1, f"Expected one full {tokens}-token P; found {found}")
        key, rid = found[0]
        native = events[key]["schedule"]
        require(
            native["expected_tokens"][rid] == tokens
            and native["computed_before"][rid] == 0,
            f"{key}: P was not a complete fresh prompt",
        )
        return key, rid

    def cohort_proof(first_key, rid, *, cancelled=False):
        first = events[first_key]["decoupled_schedule"]
        cohort_keys = [
            key
            for key in keys
            if key[0] == first_key[0]
            and events[key]["decoupled_schedule"]["cohort_id"] == first["cohort_id"]
        ]
        previous, retirement = 0, None
        progress, decode_steps = [], []
        for key in cohort_keys:
            native = events[key]["schedule"]
            schedule = events[key]["decoupled_schedule"]
            receipt = events[key]["decoupled_complete"]
            require(
                schedule["layers_completed_before"] == previous,
                f"{key}: missing layer predecessor",
            )
            current = receipt["layers_completed"]
            require(previous <= current <= expected_layers, f"{key}: invalid progress")
            require(rid not in schedule["decode_ids"], f"{key}: pending P entered D")
            if key != first_key:
                require(
                    rid not in native["scheduled_tokens"],
                    f"{key}: P tokens admitted twice",
                )
            progress.append(
                {
                    "step": key[1],
                    "before": previous,
                    "quota": schedule["layers_per_slice"],
                    "completed": current,
                    "delta": current - previous,
                    "source": receipt["progress_source"],
                }
            )
            retired = (
                rid
                in receipt["completed_prefill_ids"] + receipt["cancelled_prefill_ids"]
            )
            if receipt["decode_ids"] and not retired:
                decode_steps.append(key[1])
            if retired:
                require(retirement is None, f"{key}: duplicate retirement")
                retirement = key
            previous = current
            keep(key)
        require(retirement is not None, f"{rid}: no retirement receipt")
        final = events[retirement]["decoupled_complete"]
        target = "cancelled_prefill_ids" if cancelled else "completed_prefill_ids"
        require(rid in final[target], f"{rid}: expected {target}")
        if cancelled:
            require(
                rid
                in events[retirement]["decoupled_schedule"]["cancelled_prefill_ids"],
                f"{rid}: cancellation ACK lacks control",
            )
        else:
            require(
                previous == expected_layers, f"{rid}: completion before final layer"
            )
        later = [
            key for key in keys if key[0] == first_key[0] and key[1] > retirement[1]
        ]
        if later:
            require(
                rid
                not in events[later[0]]["decoupled_schedule"]["pending_prefill_ids"],
                f"{rid}: still pinned after ACK",
            )
            keep(later[0])
        if cancelled:
            require(later, "Cancellation proof needs a subsequent cleanup step")
            require(
                all(
                    rid not in events[key]["schedule"]["scheduled_tokens"]
                    for key in later
                ),
                f"{rid}: cancelled request rescheduled after ACK",
            )
        return {
            "request_id": rid,
            "pid": first_key[0],
            "cohort_id": first["cohort_id"],
            "first_step": first_key[1],
            "retirement_step": retirement[1],
            "progress": progress,
            "decode_steps_before_retirement": decode_steps,
            "cancelled": cancelled,
        }

    require(
        progress_report.get("passed") is True and cancel_report.get("passed") is True,
        "A progress/cancellation client failed",
    )
    probe_proofs = []
    for report, cancel in ((progress_report, False), (cancel_report, True)):
        start = report["prefill_started_monotonic"]
        end = report["decode"]["start_monotonic"] + report["decode"]["duration_s"]
        key, rid = find_prefill(12000, start, end)
        proof = cohort_proof(key, rid, cancelled=cancel)
        if not cancel:
            require(
                report["prefill"]["usage"]["prompt_tokens"] == 12000,
                "Progress client did not use 12K prompt",
            )
            require(
                proof["decode_steps_before_retirement"],
                "No D result while 12K P remained pending",
            )
        probe_proofs.append(proof)

    require(
        edge_report.get("complete") is True, "Full-budget benchmark transport failed"
    )
    edge_proofs = []
    for stage in ("warmup", "repeats"):
        require(len(edge_report[stage]) == 1, f"Expected one diagnostic {stage} run")
        run = edge_report[stage][0]
        require(run["complete"], f"Incomplete full-budget {stage}")
        p_rows = [row for row in run["requests"] if row["phase"] == "prefill"]
        d_rows = [row for row in run["requests"] if row["phase"] == "decode"]
        require(len(p_rows) == 1 and len(d_rows) == 4, "Wrong edge request coverage")
        p = p_rows[0]
        require(
            p["usage"]["prompt_tokens"] == 16383
            and p["usage"]["completion_tokens"] == 1,
            "Full-budget P usage mismatch",
        )
        require(
            all(
                row["transport_passed"]
                and row["usage"]["prompt_tokens"] == 128
                and row["usage"]["completion_tokens"] == 256
                for row in d_rows
            ),
            "Full-budget D transport/usage mismatch",
        )
        first_key, rid = find_prefill(
            16383, p["started_monotonic_s"], p["finished_monotonic_s"], p["response_id"]
        )
        first = events[first_key]["decoupled_schedule"]
        native = events[first_key]["schedule"]
        receipt = events[first_key]["decoupled_complete"]
        guard = first["quota"]["decode_budget"]
        active = set(guard["active_decode_ids"])
        require(
            len(active) == 4
            and not first["decode_ids"]
            and not guard["selected_decode_ids"],
            "Full-budget first slice lacks four live, unselected D requests",
        )
        require(
            native["scheduled_tokens"] == {rid: 16383},
            "Full-budget admission is not P-only",
        )
        require(
            first["admission"]["reason"] == "full_budget_prefill_first",
            "Missing full-budget admission reason",
        )
        require(
            not first["finish_prefill"]
            and receipt["layers_completed"] < expected_layers,
            "Full-budget first slice drained P despite live D",
        )
        if stage == "warmup":
            require(
                first["layers_per_slice"] == 1 and receipt["layers_completed"] == 1,
                "Fresh full-budget P did not use the expected cold one-layer slice",
            )
        require(
            active <= set(receipt["decode_debt_ids"]),
            "P-only step did not establish D debt",
        )
        later = [
            key for key in keys if key[0] == first_key[0] and key[1] > first_key[1]
        ]
        require(later, "Full-budget P lacks a subsequent D transaction")
        next_key = later[0]
        next_native = events[next_key]["schedule"]
        next_schedule = events[next_key]["decoupled_schedule"]
        next_receipt = events[next_key]["decoupled_complete"]
        require(
            active
            <= set(next_schedule["decode_ids"])
            == set(next_receipt["decode_ids"]),
            "Next transaction did not service the deferred D requests",
        )
        require(
            not active & set(next_receipt["decode_debt_ids"]),
            "D debt remained after D retirement",
        )
        counters = []
        for d in sorted(active):
            prior = [
                key
                for key in keys
                if key[0] == first_key[0]
                and key[1] < first_key[1]
                and d in events[key]["schedule"]["scheduled_tokens"]
            ]
            require(prior, f"Missing pre-P native counter for {d}")
            prior_key = prior[-1]
            previous = events[prior_key]["schedule"]
            expected = previous["computed_before"][d] + previous["scheduled_tokens"][d]
            require(
                next_native["computed_before"][d] == expected
                and next_native["scheduled_tokens"][d] == 1,
                f"D token count changed during P-only admission: {d}",
            )
            counters.append(
                {
                    "request_id": d,
                    "prior_step": prior_key[1],
                    "next_step": next_key[1],
                    "next_computed_before": expected,
                }
            )
            keep(prior_key)
        keep(next_key)
        proof = cohort_proof(first_key, rid)
        proof.update(
            client_stage=stage,
            first_quota=first["layers_per_slice"],
            active_decode_ids=sorted(active),
            repayment_step=next_key[1],
            decode_counter_evidence=counters,
        )
        edge_proofs.append(proof)
    require(
        [metadata(path) for path in paths] == sources,
        "Trace changed during lifecycle audit",
    )
    return {
        "passed": True,
        "diagnostic_only": True,
        "included_in_performance_summary": False,
        "trace_sources": sources,
        "expected_layers": expected_layers,
        "normal_and_cancel_proofs": probe_proofs,
        "full_budget_proofs": edge_proofs,
        "original_lines": [refs[key] for key in sorted(kept)],
        "scope": (
            "Native token admission/counters, exact progress and ACK evidence; "
            "not a device-overlap or hard-SLO proof"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="guard-diagnostic")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--producer-sha", default=BENCHMARK_SHA)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=OUT / "native16k-initial-correctness.json",
    )
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    require(
        args.stage and all(c.isalnum() or c in "-_" for c in args.stage),
        "Invalid stage name",
    )
    require(
        args.expected_layers > 0 and args.startup_timeout > 0,
        "Invalid layer count/timeout",
    )
    args.base_url = args.base_url.rstrip("/")
    args.baseline_report = args.baseline_report.resolve()
    producer_dir = OUT / (args.stage + "-producers")
    commands = client_commands(args, producer_dir)
    compatibility = {
        "requested_warmup_repeats": 0,
        "effective_warmup_repeats": 1,
        "reason": (
            "Frozen benchmark CLI requires warmup_repeats > 0; no producer changes"
        ),
        "cold_first_slice_expected_in": "warmup",
        "all_requests_diagnostic_only": True,
    }
    if args.plan_only:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "commands": commands,
                    "warmup_compatibility": compatibility,
                    "trace_glob": args.stage + "-trace-*.jsonl",
                },
                indent=2,
            )
        )
        return 0
    receipt_path = OUT / (args.stage + "-driver.json")
    require(not receipt_path.exists(), f"Refusing to overwrite {receipt_path}")
    report = {
        "stage": args.stage,
        "started_at": now(),
        "complete": False,
        "passed": False,
        "diagnostic_only": True,
        "included_in_performance_summary": False,
        "controller": metadata(__file__),
        "warmup_compatibility": compatibility,
        "commands": [],
        "health": [],
        "failures": [],
    }
    save(receipt_path, report)
    try:
        manifest_path = OUT / (args.stage + "-manifest.json")
        manifest = json.loads(manifest_path.read_text())
        report["server_manifest"] = metadata(manifest_path)
        report["baseline_report"] = metadata(args.baseline_report)
        api_pid = manifest["api_pid"]
        server_argv = manifest["argv"]
        for flag, expected in (
            ("--tensor-parallel-size", "8"),
            ("--max-num-batched-tokens", "16384"),
            ("--max-model-len", "16384"),
        ):
            require(
                server_argv[server_argv.index(flag) + 1] == expected,
                f"Diagnostic service {flag} does not equal {expected}",
            )
        policy = json.loads(
            server_argv[server_argv.index("--batch-admission-policy-config") + 1]
        )
        require(
            policy.get("decode_budget_guard") is True
            and policy.get("adaptive_layers") is True
            and policy.get("execution") == "decoupled"
            and policy.get("stream_mode") == "fenced",
            "Diagnostic service does not enable the expected guard policy",
        )
        require(
            Path(policy["trace_path"]).name == args.stage + "-trace-{pid}.jsonl",
            "Diagnostic service trace path differs from the stage",
        )
        require(
            metadata(ROOT / "scripts/benchmark_contention.py")["sha256"]
            == args.producer_sha,
            "Benchmark source differs from its registered SHA256",
        )
        require(not producer_dir.exists(), f"Refusing to reuse {producer_dir}")
        producer_dir.mkdir()
        report["producers"] = []
        for name in (
            "exercise_decoupled.py",
            "benchmark_contention.py",
            "compare_outputs.py",
            "check_decoupled_trace.py",
        ):
            source, frozen = ROOT / "scripts" / name, producer_dir / name
            frozen.write_bytes(source.read_bytes())
            report["producers"].append(
                {"original": metadata(source), "frozen": metadata(frozen)}
            )
        save(receipt_path, report)
        with httpx.Client(timeout=3, trust_env=False) as client:
            report["health"].append(
                wait_health(client, args.base_url, api_pid, args.startup_timeout)
            )
            save(receipt_path, report)
            for command in commands:
                row = run_command(command, report, receipt_path)
                if row.get("error"):
                    report["failures"].append(
                        {"name": command["name"], "error": row["error"]}
                    )
                try:
                    response = client.get(args.base_url + "/health")
                    response.raise_for_status()
                    report["health"].append(
                        {"passed": True, "after": command["name"], "at": now()}
                    )
                except Exception as exc:
                    report["health"].append(
                        {"passed": False, "after": command["name"], "error": str(exc)}
                    )
                    raise RuntimeError(
                        "Service unhealthy after diagnostic command"
                    ) from exc
                save(receipt_path, report)
        paths, trace_metadata = stable_traces(args.stage + "-trace-*.jsonl")
        checker_command = {
            "name": "trace-check",
            "output": str(OUT / (args.stage + "-check.json")),
            "argv": [
                PYTHON,
                str(producer_dir / "check_decoupled_trace.py"),
                *(str(path) for path in paths),
                "--require-cancellation",
                "--expected-layers",
                str(args.expected_layers),
                "--min-decode-steps",
                "1",
            ],
        }
        row = run_command(checker_command, report, receipt_path, stdout_json=True)
        report["trace_sources"] = trace_metadata
        if row.get("error"):
            report["failures"].append({"name": "trace-check", "error": row["error"]})
        try:
            inputs = {
                name: json.loads((OUT / f"{args.stage}-{name}.json").read_text())
                for name in ("full-budget", "progress", "cancel")
            }
            proof = lifecycle_proof(
                paths,
                inputs["full-budget"],
                inputs["progress"],
                inputs["cancel"],
                args.expected_layers,
            )
        except Exception as exc:
            proof = {
                "passed": False,
                "diagnostic_only": True,
                "error": f"{type(exc).__name__}: {exc}",
                "trace_sources": trace_metadata,
            }
            report["failures"].append(
                {"name": "lifecycle-proof", "error": proof["error"]}
            )
        proof_path = OUT / (args.stage + "-lifecycle-proof.json")
        save(proof_path, proof)
        report["lifecycle_proof"] = metadata(proof_path)
        require(
            [metadata(path) for path in paths] == trace_metadata,
            "Trace changed after clients finished",
        )
        report["complete"] = True
        report["passed"] = not report["failures"]
    except Exception as exc:
        report["failures"].append(
            {"name": "controller", "error": f"{type(exc).__name__}: {exc}"}
        )
    finally:
        report["finished_at"] = now()
        save(receipt_path, report)
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "passed": report["passed"],
                "receipt": str(receipt_path),
                "failures": report["failures"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
