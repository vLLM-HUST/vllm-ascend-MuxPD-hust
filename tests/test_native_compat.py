"""Compatibility boundaries without importing a platform or allocating tensors."""

from dataclasses import dataclass, field, make_dataclass
from types import SimpleNamespace as NS

import pytest

from vllm_muxpd.native_compat import (
    SchedulerContract,
    runner_contract,
    validate_lane_result,
    validate_lane_schedule,
    validate_output_types,
)


class OldRunner:
    def initialize_kv_cache(self, kv_cache_config):
        pass

    def initialize_kv_cache_tensors(self, kv_cache_config):
        pass


class NewRunner:
    def initialize_kv_cache(self, kv_cache_config, kv_cache_allocation_context=None):
        pass

    def initialize_kv_cache_tensors(
        self, kv_cache_config, kernel_block_sizes=None, kv_cache_allocation_context=None
    ):
        pass


@pytest.mark.parametrize(
    "runner,expected", [(OldRunner, "hust-oct10"), (NewRunner, "hust-oct25")]
)
def test_recognizes_only_complete_cache_signature_pairs(runner, expected):
    assert runner_contract(runner) == expected


@pytest.mark.parametrize(
    "method", ["initialize_kv_cache", "initialize_kv_cache_tensors"]
)
def test_unknown_cache_kwarg_is_rejected_before_call(method):
    class Changed(NewRunner):
        pass

    def unknown(self, kv_cache_config, **kwargs):
        pytest.fail("Signature validation must not execute native code")

    setattr(Changed, method, unknown)
    with pytest.raises(RuntimeError, match="Unsupported native signature"):
        runner_contract(Changed)


def test_unknown_output_field_rejected_even_when_empty():
    # An empty unknown field may acquire meaning later; it is not ignored.
    output = make_dataclass("UnknownOutput", [("future_field", object)])
    with pytest.raises(RuntimeError, match="output fields"):
        validate_output_types(output, output)


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("kv_cache_block_copies", [object()]),
        ("aux_output_connector_metadata", object()),
        ("ec_manager_metadata", object()),
        ("has_sync_kv_loads", True),
        ("kv_connector_block_state", object()),
        ("scheduled_encoder_input_stats", object()),
    ],
)
def test_new_side_effects_cannot_be_silently_duplicated_into_lanes(field_name, value):
    output = NS(scheduled_new_reqs=[], num_spec_tokens_to_schedule=0)
    setattr(output, field_name, value)
    with pytest.raises(RuntimeError, match=field_name):
        validate_lane_schedule(output)


@pytest.mark.parametrize(
    "field_name",
    [
        "sampling_masks",
        "aux_output_connector_output",
        "prompt_token_id_logprobs_dict",
        "routed_experts",
        "kv_connector_output",
        "ec_connector_output",
        "cudagraph_stats",
    ],
)
def test_optional_result_metadata_cannot_be_silently_dropped(field_name):
    result = NS(req_ids=["r"], req_id_to_index={"r": 0}, sampled_token_ids=[[1]])
    setattr(result, field_name, object())
    with pytest.raises(RuntimeError, match=field_name):
        validate_lane_result(result)


def test_empty_current_metadata_is_accepted_but_bad_rows_are_not():
    result = NS(
        req_ids=["r"],
        req_id_to_index={"r": 0},
        sampled_token_ids=[[1]],
        prompt_token_id_logprobs_dict={},
        sampling_masks=None,
        aux_output_connector_output=None,
        pooler_output=[],
    )
    validate_lane_result(result)
    result.req_id_to_index = {"r": 1}
    with pytest.raises(RuntimeError, match="inconsistent request rows"):
        validate_lane_result(result)


def test_external_prefix_replay_is_rejected():
    with pytest.raises(RuntimeError, match="partial KV prefix"):
        validate_lane_schedule(
            NS(scheduled_new_reqs=[NS(replay_start=1)], num_spec_tokens_to_schedule=0)
        )


def test_reclamation_bridge_only_emits_after_explicit_ack_call():
    @dataclass
    class Reclaimed:
        request_id: str
        session_id: str | None
        kv_blocks: int
        path: str
        ts_monotonic_ns: int = field(default=0, init=False)

    events = []
    bus = NS(enabled=True, emit=events.append)
    contract = SchedulerContract("hust-oct25", "blocked", bus, Reclaimed)
    assert contract.held_result([1, 2]) == (2, False)
    assert not events
    contract.emit_reclaimed("r", "session", 2)
    assert events == [Reclaimed("r", "session", 2, "deferred")]
    bus.enabled = False
    contract.emit_reclaimed("r", "session", 2)
    assert len(events) == 1
    old = SchedulerContract("hust-oct10", "blocked")
    assert old.held_result([1]) is None
    old.emit_reclaimed("r", None, 1)


def test_lane_validators_only_name_known_native_fields():
    from vllm_muxpd import native_compat as nc

    assert set(nc._UNSPLITTABLE_SCHEDULE_FIELDS) <= nc._SCHEDULER_NEW
    assert set(nc._UNSUPPORTED_RESULT_FIELDS) <= nc._RUNNER_OLD | nc._RUNNER_NEW


_SCHEDULE_FIELDS = (
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
)
_RESULT_FIELDS = (
    "logprobs",
    "pooler_output",
    "kv_connector_output",
    "ec_connector_output",
    "routed_experts",
    "aux_output_connector_output",
    "sampling_masks",
    "prompt_token_id_logprobs_dict",
    "cudagraph_stats",
)


def _schedule(**fields):
    from types import SimpleNamespace

    return SimpleNamespace(
        num_spec_tokens_to_schedule=0, scheduled_new_reqs=[], **fields
    )


def _result(**fields):
    from types import SimpleNamespace

    return SimpleNamespace(
        req_ids=["a"], req_id_to_index={"a": 0}, sampled_token_ids=[[1]], **fields
    )


@pytest.mark.parametrize("name", _SCHEDULE_FIELDS)
def test_lane_schedule_rejects_each_unsplittable_field(name):
    from vllm_muxpd.native_compat import validate_lane_schedule

    validate_lane_schedule(_schedule(**{name: None}))
    with pytest.raises(RuntimeError, match=f"scheduling field: {name}$"):
        validate_lane_schedule(_schedule(**{name: [1]}))


@pytest.mark.parametrize("name", _RESULT_FIELDS)
def test_lane_result_rejects_each_unsupported_field(name):
    from vllm_muxpd.native_compat import validate_lane_result

    validate_lane_result(_result(**{name: {}}))
    with pytest.raises(RuntimeError, match=f"output metadata: {name}$"):
        validate_lane_result(_result(**{name: [1]}))


def test_lane_validators_report_the_first_field_in_declared_order():
    from vllm_muxpd.native_compat import validate_lane_result, validate_lane_schedule

    both = dict.fromkeys((_SCHEDULE_FIELDS[0], _SCHEDULE_FIELDS[-1]), [1])
    with pytest.raises(RuntimeError, match=_SCHEDULE_FIELDS[0]):
        validate_lane_schedule(_schedule(**both))
    both = dict.fromkeys((_RESULT_FIELDS[0], _RESULT_FIELDS[-1]), [1])
    with pytest.raises(RuntimeError, match=_RESULT_FIELDS[0]):
        validate_lane_result(_result(**both))
