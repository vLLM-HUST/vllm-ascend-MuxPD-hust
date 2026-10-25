"""Opt-in HCCL requirement and startup metadata."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Mapping

POLICY_KEY = "require_hccl_deterministic"
PROFILE = "experimental-hccl-true-v1"


def profile_required(policy: Mapping[str, Any]) -> bool:
    required = policy.get(POLICY_KEY, False)
    if type(required) is not bool:
        raise ValueError(f"MuxPD {POLICY_KEY} must be a boolean")
    return required


def requested_environment(enabled: bool) -> dict[str, str]:
    if type(enabled) is not bool:
        raise ValueError("deterministic_collectives must be a boolean")
    return {"HCCL_DETERMINISTIC": "true"} if enabled else {}


def validate_profile_environment(policy: Mapping[str, Any]) -> bool:
    if not profile_required(policy):
        return False
    if os.environ.get("HCCL_DETERMINISTIC") != "true":
        raise ValueError(
            "MuxPD require_hccl_deterministic needs HCCL_DETERMINISTIC=true "
            "before worker distributed initialization"
        )
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise ValueError(
            "MuxPD deterministic-collectives profile conflicts with "
            "VLLM_BATCH_INVARIANT; batch invariance is a separate profile"
        )
    return True


@contextmanager
def launch_environment(enabled: bool):
    """Set explicit CLI intent before Manager spawn; restore the caller on exit.

    Manager's current adapter only permits static manifest environment keys.
    Adding a default HCCL key there would alter ordinary inherited launches.
    The policy requirement remains in its existing projected JSON configuration.
    """
    override = requested_environment(enabled)
    if not override:
        yield
        return
    original = os.environ.get("HCCL_DETERMINISTIC")
    if original not in (None, "true"):
        raise ValueError(
            "deterministic-collectives conflicts with existing "
            f"HCCL_DETERMINISTIC={original!r}; expected 'true' or unset"
        )
    os.environ["HCCL_DETERMINISTIC"] = "true"
    try:
        validate_profile_environment({POLICY_KEY: True})
        yield
    finally:
        if original is None:
            os.environ.pop("HCCL_DETERMINISTIC", None)
        else:
            os.environ["HCCL_DETERMINISTIC"] = original


def startup_readback(
    policy: Mapping[str, Any],
    *,
    rank: int,
    worker: str,
    torch_deterministic: bool,
    torch_warn_only: bool,
    prefill_groups: list[dict],
) -> dict:
    if not validate_profile_environment(policy):
        raise ValueError("Readback is only emitted for an explicitly requested profile")
    return {
        "schema": "muxpd-collective-profile-startup-v1",
        "profile": PROFILE,
        "experimental": True,
        "rank": rank,
        "pid": os.getpid(),
        "worker": worker,
        "requested_environment": requested_environment(True),
        "observed_environment": {
            name: os.environ.get(name)
            for name in ("HCCL_DETERMINISTIC", "VLLM_BATCH_INVARIANT")
        },
        "torch_deterministic_algorithms": torch_deterministic,
        "torch_deterministic_warn_only": torch_warn_only,
        "manager_plan_id": os.environ.get("VLLM_ECPA_PLAN_ID"),
        "manager_launch_id": os.environ.get("VLLM_ECPA_LAUNCH_ID"),
        "overlap_communication": policy.get("overlap_communication", "shared"),
        "compile_prefill_groups": policy.get("compile_prefill_groups", False),
        "isolated_prefill_groups": prefill_groups,
        "boundary": "after native and isolated communication group initialization",
        "scope": (
            "Environment/host metadata only; not proof of effective collective "
            "algorithm, model quality, global determinism or performance."
        ),
    }
