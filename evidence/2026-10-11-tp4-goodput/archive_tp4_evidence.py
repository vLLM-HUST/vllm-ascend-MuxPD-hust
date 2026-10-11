"""Pack immutable TP4 evidence and verify every decompressed member byte.

Default: create a NEW local-results/tp4-evidence-release directory, one tar.gz,
and a JSON index. --verify-only RELEASE_DIR only rereads that archive/index.
No files are extracted onto the source tree and no raw auditor is rerun.
Failed/partial output is retained; use a different output directory to retry.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import stat
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SOURCE_DIRECTORY = "local-results/tp4-goodput"
EXTRA_FILES = (
    "local-results/tp8-goodput/audit_goodput.py",
    "local-results/tp8-goodput/capture_cpu_environment.py",
    "local-results/tp8-goodput/plan-rate050.json",
    "scripts/benchmark_goodput.py",
    "scripts/benchmark_contention.py",
    "scripts/compare_outputs.py",
)
BLOCK = 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha_file(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        while data := stream.read(BLOCK):
            size += len(data)
            digest.update(data)
    return size, digest.hexdigest()


def stamp(path):
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode), "Expected a regular file: " + str(path))
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
    )


def safe_member(name):
    member = PurePosixPath(name)
    require(
        isinstance(name, str)
        and name
        and not member.is_absolute()
        and member.as_posix() == name
        and ".." not in member.parts
        and "." not in member.parts
        and "\\" not in name,
        "Unsafe/noncanonical archive member: " + str(name),
    )
    return member


def excluded_directory(name):
    return name.endswith("-cache") or name in ("__pycache__", ".ownership")


def enumerate_sources(repo, output):
    """Never follow directory symlinks; include only ordinary regular files."""
    root = repo / SOURCE_DIRECTORY
    require(
        root.is_dir() and not root.is_symlink(), "Evidence source directory missing"
    )
    paths, excluded = [], []
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        parent = Path(directory)
        kept = []
        for name in sorted(directories):
            candidate = parent / name
            if (
                excluded_directory(name)
                or candidate.is_symlink()
                or candidate.resolve() == output
                or candidate.resolve().is_relative_to(output)
            ):
                excluded.append(
                    {
                        "path": candidate.relative_to(repo).as_posix(),
                        "reason": "excluded directory/cache/ownership/symlink/output",
                    }
                )
            else:
                kept.append(name)
        directories[:] = kept
        for name in sorted(files):
            path = parent / name
            if name == ".ownership" or not stat.S_ISREG(path.lstat().st_mode):
                excluded.append(
                    {
                        "path": path.relative_to(repo).as_posix(),
                        "reason": "ownership or non-regular file",
                    }
                )
                continue
            if path.resolve().is_relative_to(output):
                continue
            require(path.resolve().is_relative_to(repo), "Source escapes repository")
            paths.append(path)
    for relative in EXTRA_FILES:
        path = repo / relative
        stamp(path)
        require(path.resolve().is_relative_to(repo), "Extra source escapes repository")
        paths.append(path)
    paths = sorted(set(paths), key=lambda p: p.relative_to(repo).as_posix())
    require(paths, "No evidence files")
    return paths, excluded


def inventory(repo, paths):
    entries, stamps = [], {}
    for path in paths:
        name = path.relative_to(repo).as_posix()
        safe_member(name)
        before = stamp(path)
        size, sha = sha_file(path)
        require(
            stamp(path) == before and size == before[2],
            "Source changed while hashing: " + name,
        )
        entries.append({"path": name, "bytes": size, "sha256": sha})
        stamps[name] = before
    return entries, stamps


class DigestReader:
    """Measure exactly the bytes consumed by tar.addfile, without a second copy."""

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, amount=-1):
        data = self.stream.read(amount)
        self.size += len(data)
        self.digest.update(data)
        return data


def pack(repo, target, entries, stamps):
    with target.open("xb") as raw:
        # filename="" and mtime=0 remove output-path/time variability from gzip.
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
            ) as archive:
                for entry in entries:
                    path = repo / entry["path"]
                    require(
                        stamp(path) == stamps[entry["path"]],
                        "Source changed before packing: " + entry["path"],
                    )
                    info = tarfile.TarInfo(entry["path"])
                    info.size = entry["bytes"]
                    info.mode = 0o644
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as source:
                        reader = DigestReader(source)
                        archive.addfile(info, reader)
                        require(
                            reader.size == entry["bytes"]
                            and reader.digest.hexdigest() == entry["sha256"],
                            "Source bytes changed while packing: " + entry["path"],
                        )
                    require(
                        stamp(path) == stamps[entry["path"]],
                        "Source changed during packing: " + entry["path"],
                    )


def verify_asset(release, index):
    require(index["schema"] == "tp4-evidence-release-index-v1", "Unknown index schema")
    assets = index["assets"]
    require(len(assets) == 1, "This version expects one complete tar.gz asset")
    asset = assets[0]
    name = safe_member(asset["file"])
    require(len(name.parts) == 1, "Asset must be beside the index")
    path = release / name.as_posix()
    require(not path.is_symlink(), "Archive asset cannot be a symlink")
    size, sha = sha_file(path)
    require(
        size == asset["bytes"] and sha == asset["sha256"],
        "Compressed asset SHA/size mismatch",
    )
    expected = {}
    for entry in index["files"]:
        safe_member(entry["path"])
        require(entry["path"] not in expected, "Duplicate indexed member")
        require(
            type(entry["bytes"]) is int and entry["bytes"] >= 0,
            "Invalid indexed byte count",
        )
        expected[entry["path"]] = entry
    seen, total_bytes = set(), 0
    # Streaming decompression checks contents without materializing files into
    # either the repository or a second tree. Links/devices/directories are refused.
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            safe_member(member.name)
            require(
                member.isfile() and member.name in expected and member.name not in seen,
                "Unexpected/duplicate/non-regular tar member: " + member.name,
            )
            entry = expected[member.name]
            require(
                member.size == entry["bytes"],
                "Tar member size mismatch: " + member.name,
            )
            source = archive.extractfile(member)
            require(source is not None, "Missing regular member stream")
            with source:
                digest, count = hashlib.sha256(), 0
                while data := source.read(BLOCK):
                    count += len(data)
                    digest.update(data)
            require(
                count == entry["bytes"] and digest.hexdigest() == entry["sha256"],
                "Decompressed member SHA mismatch: " + member.name,
            )
            seen.add(member.name)
            total_bytes += count
    require(seen == set(expected), "Archive is missing indexed source members")
    require(
        total_bytes == index["source_bytes"]
        and len(seen) == index["source_file_count"],
        "Index totals differ from verified members",
    )
    return {
        "passed": True,
        "verified_utc": utc_now(),
        "asset_sha256": sha,
        "verified_files": len(seen),
        "verified_uncompressed_bytes": total_bytes,
        "method": (
            "Stream-decompress every tar member; exact path/size/SHA match; "
            "no filesystem extraction"
        ),
        "raw_auditor_rerun": False,
    }


def create_release(repo, output):
    require(not output.exists(), "Output directory already exists; choose a new one")
    require(
        output != repo and not repo.is_relative_to(output),
        "Output cannot contain the repository",
    )
    # An output nested under the source tree is allowed but pruned explicitly.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    phase = "inventory"
    try:
        paths, excluded = enumerate_sources(repo, output)
        entries, stamps = inventory(repo, paths)
        inventory_path = output / "source-inventory.json"
        write_new(
            inventory_path,
            {
                "schema": "tp4-evidence-source-inventory-v1",
                "created_utc": utc_now(),
                "source_root": SOURCE_DIRECTORY,
                "extra_files": list(EXTRA_FILES),
                "files": entries,
                "excluded": excluded,
            },
        )
        asset_path = output / "tp4-goodput-evidence.tar.gz"
        phase = "compress"
        pack(repo, asset_path, entries, stamps)
        phase = "source-stability"
        after_paths, _ = enumerate_sources(repo, output)
        require(after_paths == paths, "Source file set changed during archiving")
        require(
            all(stamp(repo / name) == previous for name, previous in stamps.items()),
            "Source metadata changed after a member was packed",
        )
        size, sha = sha_file(asset_path)
        index = {
            "schema": "tp4-evidence-release-index-v1",
            "created_utc": utc_now(),
            "archive_tool": {
                "path": Path(__file__).resolve().relative_to(repo).as_posix(),
                "sha256": sha_file(__file__)[1],
            },
            "source_root": SOURCE_DIRECTORY,
            "extra_files": list(EXTRA_FILES),
            "member_path_base": "repository root",
            "source_file_count": len(entries),
            "source_bytes": sum(e["bytes"] for e in entries),
            "files": entries,
            "excluded": excluded,
            "assets": [{"file": asset_path.name, "bytes": size, "sha256": sha}],
            "normalization": {
                "gzip_mtime": 0,
                "gzip_filename": "",
                "tar_mtime": 0,
                "tar_uid": 0,
                "tar_gid": 0,
                "tar_mode": "0644",
                "member_order": "relative path lexical order",
                "contents": "exact source bytes, no redaction/transformation",
            },
        }
        phase = "verify-all-members"
        verification = verify_asset(output, index)
        index["verification"] = verification
        write_new(output / "archive-index.json", index)
        write_new(output / "verification.json", verification)
        return index
    except BaseException as error:
        write_new(
            output / "archive-failure.json",
            {
                "passed": False,
                "utc": utc_now(),
                "phase": phase,
                "error": f"{type(error).__name__}: {error}",
                "partial_outputs_retained": True,
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", type=Path, metavar="RELEASE_DIR")
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve(strict=True)
    if args.verify_only:
        require(args.output_dir is None, "--verify-only and --output-dir are exclusive")
        release = args.verify_only.resolve(strict=True)
        with (release / "archive-index.json").open() as stream:
            index = json.load(stream)
        result = verify_asset(release, index)
        print(json.dumps(result))
        return 0
    output = (args.output_dir or repo / "local-results/tp4-evidence-release").resolve()
    result = create_release(repo, output)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "source_files": result["source_file_count"],
                "source_bytes": result["source_bytes"],
                "assets": result["assets"],
                "verified": result["verification"]["passed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
