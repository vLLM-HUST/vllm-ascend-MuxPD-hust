from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm_muxpd.config import MuxPDConfig, validate_engine_config
from vllm_muxpd.policy import MuxPDPolicy, remaining_tokens, request_phase


@dataclass(frozen=True)
class RequestView:
    request_id: str
    state: str = "waiting"
    num_prompt_tokens: int = 16
    num_output_tokens: int = 0
    num_computed_tokens: int = 0
    num_in_flight_tokens: int = 0
    priority: int = 0
    arrival_time: float = 1_700_000_000.0
    max_tokens: int = 128

    @property
    def context_length(self):
        return self.num_prompt_tokens + self.num_output_tokens


def decode(rid="d"):
    return RequestView(rid, "running", 16, 2, 17)


def test_recomputed_output_history_is_prefill():
    request = RequestView("resumed", "waiting", 16, 9, 0)
    assert request_phase(request) == "prefill"
    assert remaining_tokens(request) == 25
    assert request_phase(RequestView("one-token", num_prompt_tokens=1)) == "prefill"
    assert request_phase(RequestView("partial", "running", 16, 9, 8)) == "prefill"
    assert request_phase(decode()) == "decode"


def test_initial_decode_priority_and_bounded_prefill_wait():
    policy = MuxPDPolicy(MuxPDConfig(max_decode_steps=2, tpot_slo_ms=1), 256, 8)
    requests = [decode(), RequestView("p")]
    for now in (1.0, 1.001):
        decision = policy.choose(requests, now)
        assert decision.phase == "decode"
        policy.commit(decision.phase, decision.request_ids, now)
    decision = policy.choose(requests, 1.002)
    assert decision.phase == "prefill"
    assert decision.reason == "decode_step_bound"


def test_time_bound_uses_monotonic_observation_not_epoch_arrival():
    policy = MuxPDPolicy(MuxPDConfig(tpot_slo_ms=1, max_prefill_wait_ms=10), 256, 8)
    requests = [decode(), RequestView("p")]
    assert policy.choose(requests, 50.0).phase == "decode"
    decision = policy.choose(requests, 50.011)
    assert decision.phase == "prefill"
    assert decision.reason == "prefill_wait_bound"


def test_prefill_only_packs_whole_requests_and_soft_cap_allows_one_long_prompt():
    policy = MuxPDPolicy(MuxPDConfig(max_prefill_batch_tokens=32), 128, 8)
    requests = [
        RequestView("a", num_prompt_tokens=64),
        RequestView("b", num_prompt_tokens=8),
    ]
    assert policy.choose(requests, 0).request_ids == ("a",)
    policy = MuxPDPolicy(token_budget=128, max_num_seqs=8)
    requests = [
        RequestView("a", num_prompt_tokens=80),
        RequestView("b", num_prompt_tokens=80),
    ]
    assert policy.choose(requests, 0).request_ids == ("a",)


def test_full_running_capacity_continues_decode():
    policy = MuxPDPolicy(MuxPDConfig(max_decode_steps=1), 128, 1)
    policy.decode_steps = 5
    decision = policy.choose([decode(), RequestView("p")], 0, available_prefill_slots=0)
    assert decision.phase == "decode"


def test_prefill_fits_decode_slack_after_decode_completion():
    policy = MuxPDPolicy(token_budget=128, max_num_seqs=8)
    policy.commit("decode", ["d"], 10.0)
    decision = policy.choose([decode(), RequestView("p")], 10.001)
    assert decision.phase == "prefill"
    assert decision.reason == "fits_decode_slack"
    assert decision.estimated_prefill_ms <= decision.decode_slack_ms


def test_observed_latency_updates_bounded_ewma():
    policy = MuxPDPolicy()
    policy.observe_batch("prefill", 10000, 16, 16)
    assert 1 < policy.prefill_scale <= 1.6
    assert policy.decode_scale == 1
    policy.observe_batch("decode", 10000, 1, 16)
    assert 1 < policy.decode_scale <= 1.6


def test_oversized_prefill_fails_instead_of_chunking():
    policy = MuxPDPolicy(token_budget=8)
    with pytest.raises(ValueError, match="full prefill"):
        policy.choose([RequestView("p")], 0)


def test_policy_cannot_silently_run_without_paired_scheduler():
    config = engine_config()
    config.scheduler_config.batch_admission_policy = "vllm_muxpd.policy.MuxPDPolicy"
    with pytest.raises(ValueError, match="paired MuxPD scheduler_cls"):
        MuxPDPolicy.from_config({}, config)


@pytest.mark.parametrize(
    "mode,scheduler",
    [
        ("serial", "vllm_muxpd.scheduler.MuxPDScheduler"),
        ("layered", "vllm_muxpd.multiplex_scheduler.LayeredMuxPDScheduler"),
        ("decoupled", "vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler"),
    ],
)
def test_supported_execution_modes_and_scheduler_pairing(mode, scheduler):
    config = engine_config()
    config.scheduler_config.scheduler_cls = scheduler
    config.scheduler_config.max_num_seqs = 8
    policy = MuxPDPolicy.from_config({"execution": mode}, config)
    assert policy.config.execution == mode


@pytest.mark.parametrize(
    "raw",
    [
        {"max_decode_steps": 0},
        {"max_prefill_requests": True},
        {"ewma_alpha": 1.1},
        {"prefill_token_ms": float("nan")},
        {"estimate_safety_factor": 0.5},
        {"unknown_knob": 1},
        {"trace_path": "{unknown}"},
        {"execution": "other"},
        {"layers_per_slice": 0},
        [],
        False,
        "",
    ],
)
def test_invalid_config_fails_closed(raw):
    with pytest.raises(ValueError):
        MuxPDConfig.from_mapping(raw)


def engine_config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            enable_chunked_prefill=False,
            long_prefill_token_threshold=0,
            max_num_batched_tokens=256,
            max_num_scheduled_tokens=None,
            batch_admission_policy=None,
        ),
        model_config=SimpleNamespace(max_model_len=256, runner_type="generate"),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False, mamba_cache_mode="none"
        ),
        parallel_config=SimpleNamespace(),
        max_concurrent_batches=1,
    )


@pytest.mark.parametrize(
    "group,name,value",
    [
        ("scheduler_config", "async_scheduling", True),
        ("scheduler_config", "enable_chunked_prefill", True),
        ("scheduler_config", "long_prefill_token_threshold", 16),
        ("scheduler_config", "max_num_batched_tokens", 128),
        ("scheduler_config", "max_num_scheduled_tokens", 128),
        ("cache_config", "mamba_cache_mode", "align"),
        ("cache_config", "enable_prefix_caching", True),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
    ],
)
def test_incompatible_engine_modes_rejected(group, name, value):
    config = engine_config()
    validate_engine_config(config)
    setattr(getattr(config, group), name, value)
    with pytest.raises(ValueError, match=name):
        validate_engine_config(config)
