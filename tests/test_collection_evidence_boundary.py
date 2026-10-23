"""Only reviewed code/configuration and forwarding docs remain."""

import subprocess
from pathlib import Path

PREFIXES = ["evidence/"]
RETAINED = [
]


def test_migrated_collection_boundary():
    root = Path(__file__).resolve().parents[1]
    actual = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--", *PREFIXES], text=True
    ).splitlines()
    assert set(actual) == set(RETAINED), (
        "Review new code or migrate original data to benchmark"
    )
