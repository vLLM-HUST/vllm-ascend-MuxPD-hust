"""Narrow, fail-closed bridges for two inspected HUST native contracts.

Matching these interfaces is a CPU contract check, not device qualification.
The source pairs inspected are d0f22d2/03766ac (October 10) and
62c941c/35c3e67 (October 25). Binary/runtime qualification is separate.
This module deliberately imports neither torch nor native vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from inspect import Parameter, Signature, signature
from typing import Any

_SCHEDULER_OLD = frozenset(
    """
scheduled_new_reqs scheduled_cached_reqs num_scheduled_tokens
total_num_scheduled_tokens scheduled_spec_decode_tokens scheduled_encoder_inputs
num_common_prefix_blocks finished_req_ids free_encoder_mm_hashes
preempted_req_ids has_structured_output_requests pending_structured_output_tokens
num_invalid_spec_tokens kv_connector_metadata ec_connector_metadata
new_block_ids_to_zero num_spec_tokens_to_schedule
""".split()
)
_SCHEDULER_NEW = _SCHEDULER_OLD | frozenset(
    """
scheduled_encoder_input_stats has_sync_kv_loads aux_output_connector_metadata
ec_manager_metadata kv_cache_block_copies kv_connector_block_state
""".split()
)
_RUNNER_OLD = frozenset(
    """
req_ids req_id_to_index sampled_token_ids logprobs prompt_logprobs_dict
pooler_output kv_connector_output ec_connector_output num_nans_in_logits
cudagraph_stats routed_experts
""".split()
)
_RUNNER_NEW = (_RUNNER_OLD - {"routed_experts"}) | {
    "prompt_token_id_logprobs_dict",
    "aux_output_connector_output",
    "sampling_masks",
}
_CACHED = frozenset(
    """
req_ids resumed_req_ids new_token_ids all_token_ids new_block_ids
num_computed_tokens num_output_tokens
""".split()
)
_NEW_REQUEST_OLD = frozenset(
    """
req_id prompt_token_ids mm_features sampling_params pooling_params block_ids
num_computed_tokens lora_request prompt_embeds prompt_is_token_ids prefill_token_ids
""".split()
)


def _parameters(function: Any, names: tuple[str, ...], optional: int = 0) -> None:
    parameters = tuple(signature(function).parameters.values())
    if tuple(p.name for p in parameters) != names or any(
        p.kind is not Parameter.POSITIONAL_OR_KEYWORD for p in parameters
    ):
        raise RuntimeError(f"Unsupported native signature: {function.__qualname__}")
    required = len(names) - optional
    if any(p.default is not Parameter.empty for p in parameters[:required]):
        raise RuntimeError("Unsupported native required-argument defaults")
    if any(p.default is Parameter.empty for p in parameters[required:]):
        raise RuntimeError("Unsupported native optional-argument defaults")


def _field_names(cls: type) -> frozenset[str]:
    return frozenset(f.name for f in fields(cls))


def validate_output_types(scheduler_output: type, runner_output: type) -> str:
    pair = (_field_names(scheduler_output), _field_names(runner_output))
    if pair == (_SCHEDULER_OLD, _RUNNER_OLD):
        return "hust-oct10"
    if pair == (_SCHEDULER_NEW, _RUNNER_NEW):
        return "hust-oct25"
    raise RuntimeError("Unsupported native scheduler/model output fields")


@dataclass(frozen=True)
class SchedulerContract:
    name: str
    blocked_method: str
    event_bus: Any = None
    reclaimed_event: Any = None

    @property
    def reports_reclamation(self) -> bool:
        return self.name == "hust-oct25"

    def held_result(self, blocks: list) -> tuple[int, bool] | None:
        # Even an empty detachment waits for the worker ACK before emitting the
        # matching deferred event. No request remains native-owned meanwhile.
        return (len(blocks), False) if self.reports_reclamation else None

    def emit_reclaimed(self, request_id: str, session_id: str | None, count: int):
        if self.reports_reclamation and self.event_bus.enabled:
            self.event_bus.emit(
                self.reclaimed_event(
                    request_id=request_id,
                    session_id=session_id,
                    kv_blocks=count,
                    path="deferred",
                )
            )


def scheduler_contract(
    scheduler: type,
    scheduler_output: type,
    runner_output: type,
    cached_request: type,
    new_request: type,
) -> SchedulerContract:
    """Validate coupled signatures/schema before native allocator initialization."""
    name = validate_output_types(scheduler_output, runner_output)
    _parameters(scheduler._free_request_blocks, ("self", "request"))
    annotation = signature(scheduler._free_request_blocks).return_annotation
    preempt = ("self", "request", "timestamp")
    expected_new = _NEW_REQUEST_OLD
    bus = event = None
    if name == "hust-oct25":
        if str(annotation) != "tuple[int, bool]":
            raise RuntimeError("Unsupported native KV reclamation return contract")
        _parameters(scheduler._preempt_request, (*preempt, "drop_stale_output"), 1)
        if (
            signature(scheduler._preempt_request)
            .parameters["drop_stale_output"]
            .default
            is not False
        ):
            raise RuntimeError("Unsupported native preemption default")
        blocked = "_handle_blocked_waiting_request"
        expected_new |= {"replay_start"}
        # Use the very same event outlet as the native method, never a second bus.
        namespace = scheduler._free_request_blocks.__globals__
        bus, event = namespace.get("EventBus"), namespace.get("RequestKvReclaimed")
        if (
            bus is None
            or event is None
            or _field_names(event)
            != {"ts_monotonic_ns", "request_id", "session_id", "kv_blocks", "path"}
        ):
            raise RuntimeError("Unsupported native request lifecycle event contract")
    else:
        if annotation is not Signature.empty:
            raise RuntimeError("Unsupported legacy KV reclamation return contract")
        _parameters(scheduler._preempt_request, preempt)
        blocked = "_try_promote_blocked_waiting_request"
    _parameters(getattr(scheduler, blocked), ("self", "request"))
    if (
        _field_names(cached_request) != _CACHED
        or _field_names(new_request) != expected_new
    ):
        raise RuntimeError("Unsupported native request registration fields")
    return SchedulerContract(name, blocked, bus, event)


def runner_contract(runner: type) -> str:
    initialize = runner.initialize_kv_cache
    tensors = runner.initialize_kv_cache_tensors
    if len(signature(initialize).parameters) == 2:
        _parameters(initialize, ("self", "kv_cache_config"))
        _parameters(tensors, ("self", "kv_cache_config"))
        return "hust-oct10"
    _parameters(
        initialize, ("self", "kv_cache_config", "kv_cache_allocation_context"), 1
    )
    _parameters(
        tensors,
        (
            "self",
            "kv_cache_config",
            "kernel_block_sizes",
            "kv_cache_allocation_context",
        ),
        2,
    )
    for function in (initialize, tensors):
        for p in tuple(signature(function).parameters.values())[2:]:
            if p.default is not None:
                raise RuntimeError("Unsupported native KV initialization default")
    return "hust-oct25"


def _nonempty(value: Any) -> bool:
    # Do not ask an arbitrary tensor to implement truthiness.
    return (
        value is not None
        and not (type(value) in (dict, list, tuple, set, frozenset) and len(value) == 0)
        and value is not False
    )


def validate_lane_schedule(output: Any) -> None:
    """Reject work whose side effects cannot be split across independent lanes."""
    for name in (
        "scheduled_encoder_inputs",
        "scheduled_encoder_input_stats",
        "scheduled_spec_decode_tokens",
        "has_structured_output_requests",
        "pending_structured_output_tokens",
        "num_invalid_spec_tokens",
        "kv_connector_metadata",
        "has_sync_kv_loads",
        "aux_output_connector_metadata",
        "ec_connector_metadata",
        "ec_manager_metadata",
        "kv_cache_block_copies",
        "kv_connector_block_state",
    ):
        if _nonempty(getattr(output, name, None)):
            raise RuntimeError(f"Unsupported MuxPD lane scheduling field: {name}")
    if output.num_spec_tokens_to_schedule:
        raise RuntimeError("Unsupported MuxPD speculative scheduling")
    for request in output.scheduled_new_reqs:
        if getattr(request, "replay_start", 0) != 0:
            raise RuntimeError("MuxPD cannot replay an external partial KV prefix")


def validate_lane_result(output: Any) -> None:
    for name in (
        "logprobs",
        "pooler_output",
        "kv_connector_output",
        "ec_connector_output",
        "routed_experts",
        "aux_output_connector_output",
        "sampling_masks",
        "prompt_token_id_logprobs_dict",
        "cudagraph_stats",
    ):
        if _nonempty(getattr(output, name, None)):
            raise RuntimeError(f"Unsupported MuxPD lane output metadata: {name}")
    if (
        len(output.req_ids) != len(output.sampled_token_ids)
        or (output.req_id_to_index != {rid: i for i, rid in enumerate(output.req_ids)})
        or len(set(output.req_ids)) != len(output.req_ids)
    ):
        raise RuntimeError("MuxPD lane output has inconsistent request rows")
