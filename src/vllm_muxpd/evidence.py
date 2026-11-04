"""Process-local ECPA receipts; ordinary direct launches remain unchanged.

Call only after native output sampling or validated scheduler update commits.
The Manager proves current process ownership, not model correctness or all-rank
accelerator completion. Importing this module never produces evidence.
"""

from __future__ import annotations

import os
import threading

_BINDING_ENV = (
    "VLLM_ECPA_EVIDENCE_SINK",
    "VLLM_ECPA_EVIDENCE_STRICT",
    "VLLM_ECPA_PLAN_ID",
    "VLLM_ECPA_LAUNCH_ID",
    "VLLM_HUST_EXT_EVIDENCE_PATH",
    "VLLM_HUST_EXT_MODULE_BINDING",
)
_seen: set[tuple[int, tuple[str | None, ...], type, str]] = set()
_lock = threading.RLock()
_owner_pid = os.getpid()


def emit_runtime_effective(
    owner: object,
    *,
    role: str,
    request_count: int,
    output_token_count: int,
) -> bool:
    """Publish the first successful step receipt for this live carrier process.

    Counts describe the step's returned outputs, not completed HTTP requests.
    Empty/cancelled steps must not call this helper. Successful delivery alone
    enables deduplication; errors remain visible and may be retried by the owner.
    """
    global _lock, _owner_pid
    values = {name: os.environ.get(name) for name in _BINDING_ENV}
    if all(value is None for value in values.values()):
        return False
    if not all(values.values()):
        raise ValueError("incomplete Manager module evidence environment")
    if role not in {"scheduler", "worker", "admission"}:
        raise ValueError("unknown MuxPD evidence role")
    if any(
        type(value) is not int or value <= 0
        for value in (request_count, output_token_count)
    ):
        raise ValueError("MuxPD evidence requires positive successful output counts")
    pid = os.getpid()
    if _owner_pid != pid:
        _owner_pid = pid
        _lock = threading.RLock()
        _seen.clear()
    implementation = type(owner)
    key = (
        pid,
        tuple(values.values()),
        implementation,
        role,
    )
    with _lock:
        if key in _seen:
            return False
        from vllm_hust_ext.runtime_evidence import emit_module_effective

        accepted = emit_module_effective(
            implementation,
            role=role,
            request_count=request_count,
            output_token_count=output_token_count,
        )
        if accepted is not True:
            raise RuntimeError("Manager did not accept the module runtime receipt")
        _seen.add(key)
    return True
