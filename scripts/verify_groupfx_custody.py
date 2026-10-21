"""Verify canonical grouped-prefill bytes with the existing streaming verifier."""

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archive_custody",
    ROOT / "evidence/2026-10-11-tp4-goodput/archive_tp4_evidence.py",
)
custody = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(custody)


def adapt_index(index):
    if index["schema"] != "muxpd-tp4-groupfx-evidence-archive-v1":
        raise ValueError("Unknown grouped-prefill index schema")
    return {
        "schema": "tp4-evidence-release-index-v1",
        "assets": [
            {
                "file": index["asset"],
                "bytes": index["archive_bytes"],
                "sha256": index["archive_sha256"],
            }
        ],
        "files": index["files"],
        "source_bytes": index["uncompressed_bytes"],
        "source_file_count": index["members"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    index = json.loads(
        (ROOT / "evidence/2026-10-12-tp4-group-fx/archive-index.json").read_text()
    )
    print(
        json.dumps(custody.verify_asset(args.directory, adapt_index(index)), indent=2)
    )
