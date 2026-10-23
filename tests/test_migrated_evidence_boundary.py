"""Migrated originals cannot silently return to this source checkout."""

import subprocess
from pathlib import Path


def test_migrated_directory_has_no_tracked_copies():
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--",
            "evidence/2026-10-10/",
        ]
    )
    assert not tracked, "Real data belongs in the pinned benchmark capsule"
