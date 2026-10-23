"""Only reviewed code/configuration and forwarding docs remain."""

import subprocess
from pathlib import Path

PREFIXES = ["evidence/"]
RETAINED = [
    "evidence/2026-10-10-tp4/README.md",
    "evidence/2026-10-10-tp4/overlap-profile-summary.md",
    "evidence/2026-10-10-tp4/tp4-source-audit.md",
    "evidence/2026-10-10-tp8/README.md",
    "evidence/2026-10-10-tp8/correctness-summary.md",
    "evidence/2026-10-11-matched-numeric/review.md",
    "evidence/2026-10-11-tp4-goodput/README.md",
    "evidence/2026-10-11-tp4-goodput/summary.md",
    "evidence/2026-10-11-tp8-contention/ARCHIVE.md",
    "evidence/2026-10-11-tp8-contention/contention-audit.md",
    "evidence/2026-10-11-tp8-contention/profile-hotspots.md",
    "evidence/2026-10-12-tp4-group-fx/README.md",
    "evidence/2026-10-12-tp4-group-fx/candidate-startup-artifact-audit.md",
    "evidence/2026-10-12-tp4-group-fx/numeric-early-divergence-location.md",
    "evidence/2026-10-12-tp4-group-fx/numeric19-independent-review.md",
]


def test_migrated_collection_boundary():
    root = Path(__file__).resolve().parents[1]
    actual = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--", *PREFIXES], text=True
    ).splitlines()
    assert set(actual) == set(RETAINED), (
        "Review new code or migrate original data to benchmark"
    )
