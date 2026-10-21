import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "groupfx", ROOT / "scripts/verify_groupfx_custody.py"
)
groupfx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(groupfx)


def test_adaptation_preserves_all_original_identities():
    original = json.loads(
        (ROOT / "evidence/2026-10-12-tp4-group-fx/archive-index.json").read_text()
    )
    adapted = groupfx.adapt_index(original)
    assert adapted["files"] == original["files"]
    assert adapted["source_file_count"] == len(adapted["files"]) == 348
    assert adapted["assets"][0]["sha256"] == original["archive_sha256"]
    assert adapted["source_bytes"] == sum(row["bytes"] for row in adapted["files"])


def test_unknown_schema_rejected():
    with pytest.raises(ValueError):
        groupfx.adapt_index({"schema": "unknown"})
