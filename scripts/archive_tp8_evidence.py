#!/usr/bin/env python3
"""Plan, publish, verify and unpack the TP8 contention evidence (stdlib only).

No command imports torch, calls a service, or executes an archived producer.
`plan` only lists/stat's files. `pack`, `verify`, `unpack` and `self-test` do CPU
and disk work: run these only after hardware timing and CPU tests have stopped.

Examples, from the repository root:
  python local-results/tp8-contention/archive_evidence.py plan
  python local-results/tp8-contention/archive_evidence.py pack --quiescent
  python evidence/2026-10-11-tp8-contention/archive_evidence.py verify
  python evidence/2026-10-11-tp8-contention/archive_evidence.py unpack \
    --output local-results/tp8-contention-replay

Unpack restores the relative profiler directory structure and ordinary .json /
.jsonl / .csv names. Use a fresh repository/local-results/<name> directory so
the existing profile auditor's ROOT and scripts imports continue to resolve.
The public environment snapshot deliberately omits the unrelated system-wide
process listing; this transformation is explicit and not byte-identical to
its locally preserved original. All other selected artifacts are byte-exact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
SOURCE = REPO / "local-results/tp8-contention"
ARCHIVE = REPO / "evidence/2026-10-11-tp8-contention"
PROFILE_STAGES = ("old-profile", "native-profile-profile")
PROFILE_FILES = ("trace_view.json", "kernel_details.csv")
VALIDATION_LOGS = (
    "decode-budget-cache-tests.log",
    "decode-budget-v2-regression-tests.log",
    "decode-budget-v2-tests.log",
)
ALLOWED_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
    ".log",
    ".txt",
    ".md",
    ".py",
    ".sh",
    ".svg",
}
EXCLUDED_NAMES = {
    "pr-draft.md",
    "archive-plan.json",
    "archive-index.json",
    "ARCHIVE.md",
    "start_stage.py",
    "stop_service.py",
    "run_confirmations.py",
}
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
CHUNK = 1024 * 1024
INDEX = "archive-index.json"
GUIDE = "ARCHIVE.md"
SENSITIVE_FIELD = re.compile(
    rb"""["']((?:(?:access|refresh|auth|api|connection|bearer|github|hf|private|secret)[_-]?)?token|(?:api|private|secret|access)[_-]?key|password|passwd|credential[s]?|authorization)["']\s*[:=]""",
    re.IGNORECASE,
)
SENSITIVE_ARG = re.compile(
    rb"--((?:connection|access|auth|api|hf|github)[_-]?token|api[_-]?key|password|passwd|credentials?)\b",
    re.IGNORECASE,
)
URL_CREDENTIAL = re.compile(rb"https?://[^\s/\"'<>:@]+:[^\s/\"'<>@]+@", re.IGNORECASE)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid archive path")
    path = PurePosixPath(name)
    require(
        path.parts
        and not path.is_absolute()
        and all(p not in {".", ".."} for p in path.parts),
        "Unsafe archive path",
    )
    require(path.as_posix() == name, "Noncanonical archive path")
    return Path(*path.parts)


def checked_file(root, name):
    relative = safe_relative(name)
    current = root
    for part in relative.parts:
        current /= part
        require(not current.is_symlink(), f"Symlink is forbidden: {name}")
    require(current.is_file(), f"Missing ordinary file: {name}")
    return current


def signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def input_file(source, name):
    if name in {"validation/" + item for item in VALIDATION_LOGS}:
        return checked_file(source.parent, Path(name).name)
    return checked_file(source, name)


def file_sha(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while data := handle.read(CHUNK):
            digest.update(data)
            size += len(data)
    return digest.hexdigest(), size


def scan_sensitive(data):
    """Return names/presence only, never values; inference token fields are safe."""
    result = {
        "field:" + match.group(1).decode("ascii").lower()
        for match in SENSITIVE_FIELD.finditer(data)
    }
    result.update(
        "argument:--" + match.group(1).decode("ascii").lower()
        for match in SENSITIVE_ARG.finditer(data)
    )
    if URL_CREDENTIAL.search(data):
        result.add("URL:userinfo-credentials")
    if (
        b"-----BEGIN " + b"PRIVATE KEY-----" in data
        or b"-----BEGIN " + b"RSA PRIVATE KEY-----" in data
    ):
        result.add("PEM:private-key")
    return result


def select_files(source):
    """Shallow metadata-only inventory; profiler selection is an exact allowlist."""
    source = source.resolve()
    require(source.is_dir(), "Source directory is missing")
    selected, excluded = [], []
    for path in sorted(source.iterdir()):
        name = path.name
        if path.is_symlink():
            raise ValueError(f"Source symlink is forbidden: {name}")
        if path.is_file():
            if (
                name in EXCLUDED_NAMES
                or path.suffix not in ALLOWED_SUFFIXES
                or name.startswith(".")
            ):
                excluded.append(
                    {
                        "path": name,
                        "reason": "administrative, temporary, or unsupported file",
                    }
                )
            else:
                selected.append(name)
        elif path.is_dir() and (name.endswith("-producers") or name == "figures"):
            suffixes = (
                {".json", ".svg"} if name == "figures" else {".py", ".sh", ".json"}
            )
            for producer in sorted(path.rglob("*")):
                if any(
                    part in EXCLUDED_DIRS for part in producer.relative_to(path).parts
                ):
                    continue
                require(
                    not producer.is_symlink(),
                    f"Producer symlink: {producer.relative_to(source)}",
                )
                if producer.is_file():
                    require(
                        producer.suffix in suffixes,
                        f"Unexpected producer/figure artifact: {producer.name}",
                    )
                    selected.append(producer.relative_to(source).as_posix())
        elif path.is_dir() and name not in PROFILE_STAGES:
            excluded.append(
                {
                    "path": name + "/",
                    "reason": "not a published experiment artifact directory",
                }
            )
    profile_counts = {}
    for stage in PROFILE_STAGES:
        stage_dir = source / stage
        require(
            stage_dir.is_dir() and not stage_dir.is_symlink(),
            f"Missing profile directory: {stage}",
        )
        ranks = {}
        for rank_dir in sorted(stage_dir.glob("*rank*")):
            match = re.search(r"_rank([0-9]+)_", rank_dir.name)
            require(
                match is not None and rank_dir.is_dir(),
                f"Invalid rank path: {stage}/{rank_dir.name}",
            )
            rank = int(match.group(1))
            require(rank not in ranks, f"Duplicate profiler rank: {stage}/{rank}")
            ranks[rank] = rank_dir
            for filename in PROFILE_FILES:
                relative = (
                    (rank_dir / "ASCEND_PROFILER_OUTPUT" / filename)
                    .relative_to(source)
                    .as_posix()
                )
                checked_file(source, relative)
                selected.append(relative)
        require(
            set(ranks) == set(range(8)), f"Profile must have exactly ranks 0–7: {stage}"
        )
        profile_counts[stage] = {
            "ranks": sorted(ranks),
            "files": len(ranks) * len(PROFILE_FILES),
        }
        excluded.append(
            {
                "path": stage + "/<other-profiler-files>",
                "reason": (
                    "raw profiler databases/binaries and unused exports; "
                    "trace_view.json and kernel_details.csv retained for every rank"
                ),
            }
        )
    selected.extend("validation/" + name for name in VALIDATION_LOGS)
    rows = []
    for name in sorted(selected):
        path = input_file(source, name)
        rows.append({"path": name, "size_bytes": path.stat().st_size})
    require(len(rows) == len({row["path"] for row in rows}), "Duplicate selected path")
    return {
        "files": rows,
        "excluded": excluded,
        "profiles": profile_counts,
        "source_bytes": sum(row["size_bytes"] for row in rows),
    }


def environment_payload(raw):
    value = json.loads(raw)
    require(isinstance(value, dict), "Environment snapshot must be an object")
    if "processes" not in value:
        return raw, []
    value.pop("processes")
    return json_bytes(value), [
        {
            "operation": "remove_json_field",
            "field": "processes",
            "reason": (
                "system-wide administrative process arguments can contain "
                "credentials; unrelated to measured inference"
            ),
        }
    ]


def write_artifact(source, name, destination, compress):
    path = input_file(source, name)
    before = signature(path)
    source_digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    payload_size = 0
    findings = set()
    trailing = b""
    transformations = []
    stored_name = name + ".gz" if compress else name
    output = destination / safe_relative(stored_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    if name == "environment-initial.json":
        raw = path.read_bytes()
        source_digest.update(raw)
        payload, transformations = environment_payload(raw)
        reader = io.BytesIO(payload)
    else:
        reader = path.open("rb")
    with reader, output.open("xb") as raw_output:
        writer = (
            gzip.GzipFile(
                filename="", mode="wb", compresslevel=6, fileobj=raw_output, mtime=0
            )
            if compress
            else raw_output
        )
        try:
            while data := reader.read(CHUNK):
                if name != "environment-initial.json":
                    source_digest.update(data)
                payload_digest.update(data)
                payload_size += len(data)
                findings.update(scan_sensitive(trailing + data))
                trailing = data[-512:]
                writer.write(data)
        finally:
            if compress:
                writer.close()
    require(signature(path) == before, f"Source changed while copying: {name}")
    require(
        not findings,
        f"Sensitive fields/parameters require review in {name}: "
        + ", ".join(sorted(findings)),
    )
    stored_digest, stored_size = file_sha(output)
    origin = "../" + Path(name).name if name.startswith("validation/") else name
    return {
        "source_path": name,
        "original_source_relative_path": origin,
        "source_size_bytes": before[2],
        "source_sha256": source_digest.hexdigest(),
        "stored_path": stored_name,
        "stored_size_bytes": stored_size,
        "stored_sha256": stored_digest,
        "uncompressed_size_bytes": payload_size,
        "uncompressed_sha256": payload_digest.hexdigest(),
        "compression": "gzip-mtime0-level6" if compress else "none",
        "transformations": transformations,
    }


def guide_text():
    return """# Replaying the TP8 contention archive

`archive-index.json` records source, stored and decompressed SHA-256 and byte
counts. Unchanged artifacts have identical source/decompressed SHA. Gzip uses
an empty filename, mtime=0 and level 6; the Python/zlib versions are in the index.
The index is not a cryptographic signature; pin the Git commit when citing it.

All measured stages, including guard v1 and the excluded CPU-overlap old stage,
are retained. Selection/exclusion decisions belong to the plans and final audit;
publication is not evidence that every included measurement is a final candidate.
Client warmups, failures and raw SSE are not removed. All 16 rank trace exports
and kernel CSVs are included with their original relative profiler directories.
No profiler database/binary is needed by the archived offline classifier.

`environment-initial.json` omits the original `processes` field, which contained
unrelated system-management commands and connection-token arguments. The local
original is retained. The index explicitly identifies this transformation and
both hashes: the public snapshot is not presented as its byte-identical original.
Administrative PID/control files, PR drafts, caches and temporary files are
excluded. Other selected raw data is byte-exact; inferred token fields such as
`token_ids` and `max_tokens` are never treated as credentials or removed.

The three external CPU logs referenced by source-validation.json are copied
from the parent local-results directory into validation/. Their exact origin
mapping is indexed. The early decode-budget-v2-tests.log is retained alongside
the later regression/full-suite passes; a failed test invocation is not removed
because a subsequent corrected fixture passes. Consult source-validation.json
and the logs for the command, failure and later validation scope.

From the repository root, after timing/CPU tests have finished:

```sh
python evidence/2026-10-11-tp8-contention/archive_evidence.py verify
python evidence/2026-10-11-tp8-contention/archive_evidence.py unpack \\
  --output local-results/tp8-contention-replay
python local-results/tp8-contention-replay/profile-hotspots-audit.py \\
  --output local-results/tp8-contention-replay/replayed-profile-hotspots.json
```

The unpack destination must not exist. Unpack verifies stored and decompressed
hashes before atomically publishing the restored tree. Use a directory immediately
inside this repository's `local-results` so the profile auditor's repo-root import
of `scripts/analyze_npu_trace.py` resolves. Its historical report source paths can
change after replay; compare metrics and source hashes, not absolute path strings.
Check the indexed classifier SHA against the checked-out script before replay.
Auditing/profiling can be CPU intensive and must not overlap inference timing.
No archived launcher is executed by verification or unpacking.
"""


def validate_index(index):
    require(
        index.get("schema") == "tp8-contention-archive-v1", "Unsupported archive schema"
    )
    source_names, stored_names = set(), set()
    for row in index["artifacts"]:
        safe_relative(row["source_path"])
        safe_relative(row["stored_path"])
        expected_origin = (
            "../" + Path(row["source_path"]).name
            if row["source_path"] in {"validation/" + name for name in VALIDATION_LOGS}
            else row["source_path"]
        )
        require(
            row["original_source_relative_path"] == expected_origin,
            "Original source mapping mismatch",
        )
        require(
            row["source_path"] not in source_names
            and row["stored_path"] not in stored_names,
            "Duplicate indexed path",
        )
        source_names.add(row["source_path"])
        stored_names.add(row["stored_path"])
        require(
            row["compression"] in {"none", "gzip-mtime0-level6"}, "Unknown compression"
        )
        suffix = ".gz" if row["compression"] != "none" else ""
        require(
            row["stored_path"] == row["source_path"] + suffix, "Stored layout mismatch"
        )
        for key in ("source_sha256", "stored_sha256", "uncompressed_sha256"):
            require(
                re.fullmatch(r"[0-9a-f]{64}", row[key]) is not None,
                "Invalid indexed hash",
            )
        for key in (
            "source_size_bytes",
            "stored_size_bytes",
            "uncompressed_size_bytes",
        ):
            require(type(row[key]) is int and row[key] >= 0, "Invalid indexed size")
        if row["transformations"]:
            require(
                row["source_path"] == "environment-initial.json"
                and len(row["transformations"]) == 1
                and row["transformations"][0].get("operation") == "remove_json_field"
                and row["transformations"][0].get("field") == "processes",
                "Unexpected public data transformation",
            )
        else:
            require(
                row["source_sha256"] == row["uncompressed_sha256"]
                and row["source_size_bytes"] == row["uncompressed_size_bytes"],
                "Untransformed payload differs from source",
            )
    require(not ({INDEX, GUIDE} & stored_names), "Reserved path collision")


def verify(archive, output=None, original_source=None):
    archive = archive.resolve()
    index = json.loads(checked_file(archive, INDEX).read_bytes())
    validate_index(index)
    expected = {INDEX, GUIDE} | {row["stored_path"] for row in index["artifacts"]}
    actual = set()
    for path in archive.rglob("*"):
        require(not path.is_symlink(), "Archive contains symlink")
        if path.is_file():
            actual.add(path.relative_to(archive).as_posix())
    require(actual == expected, "Archive has missing or unindexed files")
    require(
        file_sha(checked_file(archive, GUIDE))
        == (index["guide_sha256"], index["guide_size_bytes"]),
        "Archive guide hash mismatch",
    )
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    for row in index["artifacts"]:
        stored = checked_file(archive, row["stored_path"])
        require(
            file_sha(stored) == (row["stored_sha256"], row["stored_size_bytes"]),
            f"Stored hash/size mismatch: {row['stored_path']}",
        )
        digest, total, findings, trailing = hashlib.sha256(), 0, set(), b""
        opener = gzip.open if row["compression"] != "none" else open
        target = None
        if output is not None:
            target_path = output / safe_relative(row["source_path"])
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target = target_path.open("xb")
        try:
            with opener(stored, "rb") as reader:
                while data := reader.read(CHUNK):
                    total += len(data)
                    require(
                        total <= row["uncompressed_size_bytes"],
                        f"Decompressed size exceeds index: {row['stored_path']}",
                    )
                    digest.update(data)
                    findings.update(scan_sensitive(trailing + data))
                    trailing = data[-512:]
                    if target:
                        target.write(data)
        finally:
            if target:
                target.close()
        require(
            not findings,
            f"Sensitive field/parameter presence in {row['source_path']}: "
            + ", ".join(sorted(findings)),
        )
        require(
            (digest.hexdigest(), total)
            == (row["uncompressed_sha256"], row["uncompressed_size_bytes"]),
            f"Decompressed hash mismatch: {row['stored_path']}",
        )
        if original_source is not None:
            original = input_file(original_source, row["source_path"])
            require(
                file_sha(original) == (row["source_sha256"], row["source_size_bytes"]),
                f"Original source hash mismatch: {row['source_path']}",
            )
    return {
        "passed": True,
        "artifacts": len(index["artifacts"]),
        "source_hashes_verified": original_source is not None,
        "profiles": index["profiles"],
    }


def publish(source, destination, threshold, require_final=True):
    import zlib

    source = source.resolve()
    destination = destination.resolve()
    require(
        not destination.exists(),
        "Archive destination already exists; refusing mutation",
    )
    require(
        not destination.is_relative_to(source),
        "Archive cannot be inside source directory",
    )
    inventory = select_files(source)
    if require_final:
        final = json.loads(
            checked_file(source, "contention-audit.json").read_bytes()
        ).get("final", {})
        require(
            final.get("status") in {"passed", "complete"},
            "Final comparison audit is pending or failed",
        )
    before = {
        row["path"]: signature(input_file(source, row["path"]))
        for row in inventory["files"]
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".contention-archive-", dir=destination.parent))
    try:
        rows = []
        for row in inventory["files"]:
            name = row["path"]
            compress = (
                Path(name).suffix in {".json", ".jsonl", ".csv"}
                and row["size_bytes"] >= threshold
            )
            rows.append(write_artifact(source, name, temp, compress))
        guide = guide_text().encode()
        (temp / GUIDE).write_bytes(guide)
        classifier = source.parents[1] / "scripts/analyze_npu_trace.py"
        dependencies = []
        if classifier.is_file():
            dependencies.append(
                {
                    "repo_path": "scripts/analyze_npu_trace.py",
                    "sha256": file_sha(classifier)[0],
                }
            )
        index = {
            "schema": "tp8-contention-archive-v1",
            "artifacts": rows,
            "profiles": inventory["profiles"],
            "excluded": inventory["excluded"],
            "compression_threshold_bytes": threshold,
            "compression_level": 6,
            "python_version": sys.version.split()[0],
            "zlib_version": zlib.ZLIB_VERSION,
            "guide_sha256": sha(guide),
            "guide_size_bytes": len(guide),
            "external_replay_dependencies": dependencies,
            "policy": (
                "preserve all selected measured stages; "
                "environment.processes only is omitted; no other redaction"
            ),
        }
        (temp / INDEX).write_bytes(json_bytes(index))
        result = verify(temp, original_source=source)
        require(
            select_files(source) == inventory, "Source inventory changed while packing"
        )
        require(
            all(
                signature(input_file(source, name)) == old
                for name, old in before.items()
            ),
            "Source changed while packing; stop producers first",
        )
        temp.rename(destination)
        return result
    except BaseException:
        shutil.rmtree(temp)
        raise


def unpack(archive, destination):
    destination = destination.resolve()
    require(not destination.exists(), "Unpack destination already exists")
    require(
        not destination.is_relative_to(archive.resolve()),
        "Unpack cannot mutate the archive",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    container = Path(
        tempfile.mkdtemp(prefix=".contention-unpack-", dir=destination.parent)
    )
    restored = container / "restored"
    try:
        result = verify(archive, output=restored)
        shutil.copyfile(archive / INDEX, restored / INDEX)
        shutil.copyfile(archive / GUIDE, restored / GUIDE)
        restored.rename(destination)
        return result
    finally:
        shutil.rmtree(container)


def self_test():
    """Tiny temporary fixtures only; no real evidence is read or changed."""
    import unittest

    class ArchiveTests(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory()
            self.addCleanup(self.temp.cleanup)
            self.base = Path(self.temp.name)
            self.source = self.base / "inputs"
            self.source.mkdir()
            for stage in PROFILE_STAGES:
                for rank in range(8):
                    directory = (
                        self.source
                        / stage
                        / f"dp0_rank{rank}_fixture"
                        / "ASCEND_PROFILER_OUTPUT"
                    )
                    directory.mkdir(parents=True)
                    (directory / "trace_view.json").write_bytes(b'{"traceEvents":[]}\n')
                    (directory / "kernel_details.csv").write_bytes(
                        b"Name,Duration\nfixture,1\n"
                    )
            for name in VALIDATION_LOGS:
                (self.base / name).write_text("fixture validation receipt\n")
            (self.source / "contention-audit.json").write_bytes(
                json_bytes({"final": {"status": "passed"}})
            )
            self.raw = json_bytes(
                {"token_ids": [1, 2] * 40, "max_tokens": 80, "complete": True}
            )
            (self.source / "measured.json").write_bytes(self.raw)
            self.original_environment = json_bytes(
                {
                    "cpu_max": "32",
                    "processes": "--connection" + "-token=fixture-never-publish",
                }
            )
            (self.source / "environment-initial.json").write_bytes(
                self.original_environment
            )
            (self.source / "service.pid").write_text("42")
            (self.source / "pr-draft.md").write_text("administrative fixture")
            (self.source / "figures").mkdir()
            (self.source / "figures" / "timeline.svg").write_text("<svg/>\n")

        def pack(self, name="archive"):
            destination = self.base / name
            publish(self.source, destination, 1)
            return destination

        def test_round_trip_all_ranks_transform_external_logs_and_exclusions(self):
            archive = self.pack()
            result = verify(archive, original_source=self.source)
            self.assertTrue(result["passed"])
            self.assertEqual(result["profiles"][PROFILE_STAGES[0]]["files"], 16)
            index = json.loads((archive / INDEX).read_bytes())
            environment = next(
                row
                for row in index["artifacts"]
                if row["source_path"] == "environment-initial.json"
            )
            self.assertEqual(
                environment["source_sha256"], sha(self.original_environment)
            )
            self.assertNotEqual(
                environment["source_sha256"], environment["uncompressed_sha256"]
            )
            restored = self.base / "restored"
            unpack(archive, restored)
            self.assertEqual((restored / "measured.json").read_bytes(), self.raw)
            self.assertNotIn(
                "processes",
                json.loads((restored / "environment-initial.json").read_bytes()),
            )
            self.assertFalse((restored / "service.pid").exists())
            self.assertFalse((restored / "pr-draft.md").exists())
            self.assertEqual(
                (restored / "figures" / "timeline.svg").read_text(), "<svg/>\n"
            )
            self.assertEqual(
                len(
                    list(
                        restored.glob(
                            "*profile*/*rank*/ASCEND_PROFILER_OUTPUT/trace_view.json"
                        )
                    )
                ),
                16,
            )
            self.assertEqual(
                (restored / "validation" / VALIDATION_LOGS[0]).read_bytes(),
                (self.base / VALIDATION_LOGS[0]).read_bytes(),
            )

        def test_gzip_bytes_are_deterministic(self):
            first, second = self.pack("first"), self.pack("second")
            self.assertEqual(
                (first / "measured.json.gz").read_bytes(),
                (second / "measured.json.gz").read_bytes(),
            )
            self.assertEqual(
                (first / INDEX).read_bytes(), (second / INDEX).read_bytes()
            )

        def test_sensitive_names_fail_without_echoing_values(self):
            key = "pass" + "word"
            secret = "fixture-do-not-echo"
            (self.source / "sensitive.json").write_bytes(json_bytes({key: secret}))
            with self.assertRaises(ValueError) as caught:
                self.pack()
            self.assertIn("sensitive.json", str(caught.exception))
            self.assertIn(key, str(caught.exception))
            self.assertNotIn(secret, str(caught.exception))
            self.assertFalse((self.base / "archive").exists())
            self.assertFalse(list(self.base.glob(".contention-archive-*")))

        def test_normal_inference_token_fields_are_preserved(self):
            self.assertFalse(scan_sensitive(self.raw))
            self.assertEqual(
                scan_sensitive(b'{"api_' + b'key":"fixture"}'), {"field:api_key"}
            )

        def test_missing_rank_and_symlink_are_rejected(self):
            rank_dir = self.source / PROFILE_STAGES[0] / "dp0_rank7_fixture"
            shutil.rmtree(rank_dir)
            with self.assertRaises(ValueError):
                select_files(self.source)
            (self.source / "unexpected.json").symlink_to(self.source / "measured.json")
            with self.assertRaises(ValueError):
                select_files(self.source)

        def test_corruption_extra_file_and_path_escape_are_rejected(self):
            archive = self.pack()
            stored = archive / "measured.json.gz"
            original = stored.read_bytes()
            stored.write_bytes(original + b"corrupt")
            with self.assertRaises(ValueError):
                verify(archive)
            stored.write_bytes(original)
            (archive / "unexpected.txt").write_text("unexpected")
            with self.assertRaises(ValueError):
                verify(archive)
            (archive / "unexpected.txt").unlink()
            index = json.loads((archive / INDEX).read_bytes())
            index["artifacts"][0]["source_path"] = "../escape"
            (archive / INDEX).write_bytes(json_bytes(index))
            with self.assertRaises(ValueError):
                unpack(archive, self.base / "output")
            self.assertFalse((self.base / "output").exists())

        def test_pending_audit_existing_dest_and_changed_source_fail(self):
            archive = self.pack()
            with self.assertRaises(ValueError):
                publish(self.source, archive, 1)
            (self.source / "measured.json").write_bytes(self.raw + b" ")
            with self.assertRaises(ValueError):
                verify(archive, original_source=self.source)
            (self.source / "contention-audit.json").write_bytes(
                json_bytes({"final": {"status": "pending"}})
            )
            with self.assertRaises(ValueError):
                self.pack("pending")

        def test_unpack_parent_symlink_and_dotdot_cannot_mutate_archive(self):
            archive = self.pack()
            before = sorted(str(p.relative_to(archive)) for p in archive.rglob("*"))
            alias = self.base / "archive-alias"
            alias.symlink_to(archive, target_is_directory=True)
            for destination in (
                alias / "escape",
                self.base / "unused" / ".." / "archive" / "escape",
            ):
                with self.assertRaises(ValueError):
                    unpack(archive, destination)
            self.assertEqual(
                sorted(str(p.relative_to(archive)) for p in archive.rglob("*")),
                before,
            )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ArchiveTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), "Archive self-tests failed")
    return {"passed": True, "tests": result.testsRun, "fixtures_only": True}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "self-test",
        help="small CPU-only temporary fixtures; wait until inference timing stops",
    )
    for name in ("plan", "pack"):
        command = sub.add_parser(name)
        command.add_argument("--source", type=Path, default=SOURCE)
        if name == "pack":
            command.add_argument("--destination", type=Path, default=ARCHIVE)
            command.add_argument("--gzip-threshold", type=int, default=1024 * 1024)
            command.add_argument(
                "--quiescent",
                action="store_true",
                help="assert all producers, hardware timing and CPU tests have stopped",
            )
    for name in ("verify", "unpack"):
        command = sub.add_parser(name)
        command.add_argument("--archive", type=Path, default=ARCHIVE)
        if name == "verify":
            command.add_argument(
                "--original-source",
                type=Path,
                help="also verify private original source hashes locally",
            )
        else:
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        result = self_test()
    elif args.command == "plan":
        result = select_files(args.source)
        result["note"] = (
            "Metadata-only plan; contents not hashed/scanned. Pack validates "
            "credentials, source stability and final audit."
        )
    elif args.command == "pack":
        require(
            args.quiescent,
            "Pass --quiescent only after timing, producers and CPU tests have stopped",
        )
        require(args.gzip_threshold >= 0, "Compression threshold must be nonnegative")
        result = publish(args.source, args.destination, args.gzip_threshold)
    elif args.command == "verify":
        result = verify(args.archive, original_source=args.original_source)
    else:
        result = unpack(args.archive, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(f"Archive operation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
