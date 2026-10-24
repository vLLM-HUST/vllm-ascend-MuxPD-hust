import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_mod_consumer_evidence.py"
SPEC = importlib.util.spec_from_file_location("capsule", SCRIPT)
capsule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capsule)


def bundle(
    name="snapshot/data.json", content=b"failure retained", link=False, duplicate=False
):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            member.size = 0
        archive.addfile(member, io.BytesIO(content))
        if duplicate:
            archive.addfile(member, io.BytesIO(content))
    raw = stream.getvalue()
    manifest = {
        "archive": {
            "bytes": len(raw),
            "sha256": capsule.digest(raw),
            "member_prefix": "snapshot/",
        },
        "files": [
            {
                "path": "data.json",
                "bytes": len(content),
                "sha256": capsule.digest(content),
            }
        ],
    }
    return manifest, raw


def test_preserves_original_bytes():
    manifest, raw = bundle()
    assert capsule.verify(manifest, raw) == {"data.json": b"failure retained"}


@pytest.mark.parametrize(
    "kwargs", [{"name": "snapshot/../escape"}, {"link": True}, {"duplicate": True}]
)
def test_unsafe_members_rejected(kwargs):
    manifest, raw = bundle(**kwargs)
    with pytest.raises(ValueError):
        capsule.verify(manifest, raw)


def test_hash_missing_and_unsafe_manifest_rejected():
    manifest, raw = bundle()
    with pytest.raises(ValueError):
        capsule.verify(manifest, raw + b"changed")
    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        capsule.verify(manifest, raw)
    manifest["files"][0]["path"] = "../escape"
    with pytest.raises(ValueError):
        capsule.verify(manifest, raw)


SNAPSHOT_URL = (
    "https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/"
    + "a" * 40
    + "/archive/test/manifest.json"
)


def snapshot():
    return {
        "schema": "mod-git-custody/v1",
        "source_visibility": "public",
        "files": [
            {"path": "docs/original.md", "bytes": 3, "sha256": capsule.digest(b"raw")}
        ],
    }


def test_snapshot_preserves_bytes(monkeypatch):
    calls = []

    def fetch(url, limit):
        calls.append((url, limit))
        return b"raw"

    monkeypatch.setattr(capsule, "fetch", fetch)
    assert capsule.fetch_snapshot(snapshot(), SNAPSHOT_URL) == {
        "docs/original.md": b"raw"
    }
    assert calls == [(SNAPSHOT_URL.rsplit("/", 1)[0] + "/snapshot/docs/original.md", 3)]


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "a/../b", "a//b", ".", "a\\b"]
)
def test_snapshot_rejects_paths(monkeypatch, path):
    manifest = snapshot()
    manifest["files"][0]["path"] = path
    monkeypatch.setattr(capsule, "fetch", lambda *_: pytest.fail("unsafe fetch"))
    with pytest.raises(ValueError):
        capsule.fetch_snapshot(manifest, SNAPSHOT_URL)


@pytest.mark.parametrize(
    "kind",
    ["hash", "size", "duplicate", "oversize", "empty", "private", "schema", "unpinned"],
)
def test_snapshot_rejects_invalid_identity(monkeypatch, kind):
    manifest = snapshot()
    url = SNAPSHOT_URL
    if kind == "hash":
        manifest["files"][0]["sha256"] = "0" * 64
    elif kind == "size":
        manifest["files"][0]["bytes"] = 4
    elif kind == "duplicate":
        manifest["files"] *= 2
    elif kind == "oversize":
        manifest["files"][0]["bytes"] = 5_000_001
    elif kind == "empty":
        manifest["files"] = []
    elif kind == "private":
        manifest["source_visibility"] = "private"
    elif kind == "schema":
        manifest["schema"] = "unknown"
    else:
        url = url.replace("a" * 40, "main")
    monkeypatch.setattr(capsule, "fetch", lambda *_: b"raw")
    with pytest.raises(ValueError):
        capsule.fetch_snapshot(manifest, url)


def test_document_projection_links():
    root = SCRIPT.parents[1]
    config = json.loads((root / "BENCHMARK_DOCUMENTS.json").read_text())
    for row in config["document_projections"]:
        assert row["original_url"] in (root / row["path"]).read_text()


@pytest.mark.parametrize("valid", [True, False])
def test_snapshot_cli_checks_binding_before_writing(tmp_path, monkeypatch, valid):
    raw = json.dumps(snapshot()).encode()
    config = {
        "manifest_url": SNAPSHOT_URL,
        "manifest_sha256": capsule.digest(raw),
        "document_projections": [
            {
                "path": "docs/original.md",
                "original_url": SNAPSHOT_URL.rsplit("/", 1)[0]
                .replace(
                    "https://raw.githubusercontent.com/",
                    "https://github.com/",
                )
                .replace("vllm-hust-benchmark/", "vllm-hust-benchmark/blob/")
                + "/snapshot/docs/"
                + ("original.md" if valid else "wrong.md"),
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    destination = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--config", str(config_path), "--destination", str(destination)],
    )
    monkeypatch.setattr(
        capsule, "fetch", lambda url, _: raw if url == SNAPSHOT_URL else b"raw"
    )
    if valid:
        capsule.main()
        assert (destination / "docs/original.md").read_bytes() == b"raw"
    else:
        with pytest.raises(ValueError, match="projection"):
            capsule.main()
        assert not destination.exists()
