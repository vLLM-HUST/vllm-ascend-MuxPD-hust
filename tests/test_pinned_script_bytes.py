"""Experiment controllers whose exact bytes are recorded by archived evidence.

The external benchmark collection (see EVIDENCE.md) cites each of these files
by SHA-256. Editing one, even for formatting, breaks that custody link.
Version a changed controller under a new name instead.
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# script -> (sha256, one archived collection member that records it)
PINNED = {
    "analyze_npu_trace.py": (
        "f9cb2fcef2af0e425b77d5de803524ee0eeff46ffba36aa43ce003cd2c68e6d2",
        "evidence/2026-10-11-tp8-contention-replay-validation.json",
    ),
    "benchmark_contention.py": (
        "571eb1ce9b8a2218bb00f4c64d2b82d56c889d8922f4cfb5ab21f47fa6d49a9f",
        "evidence/2026-10-11-matched-numeric/audit.py",
    ),
    "benchmark_goodput.py": (
        "d80706a15e7b3f6cbd11a9fce054f07f819154576fff922c2c02462ed6a6cb3f",
        "evidence/2026-10-11-tp4-goodput/archive-index.json",
    ),
    "benchmark_serving.py": (
        "32f5d86f52cc78fe5dd0a61599c2523c03914376c700155102b16517482bd4aa",
        "evidence/2026-10-10-tp4/benchmark-summary.json",
    ),
    "check_decode_graph_serving.py": (
        "1e6cb3ef7e96fd8d2521e85f2d7f90833a556b7565fb1edc6ae79b06a0ad5a77",
        "evidence/2026-10-10-tp8/correctness-summary.md",
    ),
    "check_decoupled_trace.py": (
        "c7fec7d4dcaee2375fd2fdce7f231e23b46cf038f074d9afedd70dc745a8adba",
        "evidence/2026-10-12-tp4-group-fx/archive-index.json",
    ),
    "check_trace.py": (
        "5fe305fb5f3b521299234b4efd24361b21c99b863b6bcfe4562bd678bdd01331",
        "evidence/2026-10-10-tp8/environment-initial.json",
    ),
    "compare_outputs.py": (
        "1ef81477269707616c5d0a14becce80df4acf3a671f17484e5716d7fe25fa551",
        "evidence/2026-10-10-tp8/correctness-summary.md",
    ),
    "exercise_decoupled.py": (
        "4e096a4bb4bed5df266ed7a945132da4b7ddabe2ad6e2aa211df07f690071fa7",
        "evidence/2026-10-10-tp8/environment-initial.json",
    ),
    "probe_graph_state_reuse.py": (
        "07fc9dedcd3c56db77e5aca3e28eb4f9c944fcb762902bf81fda239b6b3d8da4",
        "evidence/2026-10-10-tp8/correctness-summary.md",
    ),
    "probe_prefill_group_outputs.py": (
        "94c62c43c30efb6dbdd0a437ea8aa70b4ce111fc623350f7af0f4f345fe17883",
        "evidence/2026-10-12-tp4-group-fx/numeric19-independent-audit.json",
    ),
    "serve-qwen35.sh": (
        "fb82c792849c5fc800366434ab9562073b06a57453479bab7958a563bbee5e30",
        "evidence/2026-10-10-tp4/environment.json",
    ),
    "serve-qwen35-decode-graph.sh": (
        "dfef83405a05a46234df824ef8ec77868e77f4b62d6d51d2409e66980180e5d0",
        "evidence/2026-10-10-tp8/environment-initial.json",
    ),
    "serve-qwen35-streaming.sh": (
        "36464a77684891bbf16d96d984702198542064116348e02b17c4991f7f345706",
        "evidence/2026-10-12-tp4-group-fx/archive-index.json",
    ),
    "smoke_serving.py": (
        "b77a461a2750e9166bc8bf15c50f54c5b52fab1ff7a7fd1808251cb78b66d704",
        "evidence/2026-10-10-tp8/environment-initial.json",
    ),
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_pinned_script_bytes_are_unchanged(name):
    digest = hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
    expected, member = PINNED[name]
    assert digest == expected, f"scripts/{name} is pinned by {member}"
