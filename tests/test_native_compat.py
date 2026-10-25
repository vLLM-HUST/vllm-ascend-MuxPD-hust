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
