"""Async admission must be an explicit, matching scheduler/worker opt-in."""

import pytest
from test_policy import engine_config

from vllm_muxpd.config import MuxPDConfig, validate_engine_config


def config():
    result = engine_config()
    result.scheduler_config.batch_admission_policy_config = {
        "execution": "decoupled",
        "async_decode_pipeline": True,
    }
    result.scheduler_config.async_scheduling = True
    result.scheduler_config.scheduler_cls = (
        "vllm_muxpd.async_scheduler.AsyncDecoupledMuxPDScheduler"
    )
    result.parallel_config.worker_cls = (
        "vllm_muxpd.async_worker.AsyncDecoupledMuxPDNPUWorker"
    )
    result.max_concurrent_batches = 2
    return result


def test_async_is_disabled_by_default_and_explicit_pair_is_accepted():
    assert MuxPDConfig().async_decode_pipeline is False
    validate_engine_config(engine_config())
    validate_engine_config(config())


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_async_flag_requires_a_literal_boolean(value):
    with pytest.raises(ValueError, match="async_decode_pipeline"):
        MuxPDConfig(execution="decoupled", async_decode_pipeline=value)


@pytest.mark.parametrize(
    "group, name, value",
    [
        ("scheduler_config", "async_scheduling", False),
        ("scheduler_config", "scheduler_cls", "vllm_muxpd.scheduler.MuxPDScheduler"),
        (
            "parallel_config",
            "worker_cls",
            "vllm_muxpd.atomic_worker.AtomicDecoupledMuxPDNPUWorker",
        ),
    ],
)
def test_async_rejects_incompatible_pair_or_engine_mode(group, name, value):
    result = config()
    setattr(getattr(result, group), name, value)
    with pytest.raises(ValueError, match="async_decode_pipeline"):
        validate_engine_config(result)


def test_async_requires_exactly_two_native_concurrent_batches():
    result = config()
    result.max_concurrent_batches = 1
    with pytest.raises(ValueError, match="max_concurrent_batches"):
        validate_engine_config(result)


@pytest.mark.parametrize(
    "other",
    [
        {"adaptive_layers": True},
        {"prefill_first": True},
        {"decode_budget_guard": True},
        {"compile_full_prefill": True},
        {"prefill_layer_token_budget": 8192},
    ],
)
def test_async_excludes_other_adaptive_policy_experiments(other):
    with pytest.raises(ValueError, match="fixed layer quotas"):
        MuxPDConfig(execution="decoupled", async_decode_pipeline=True, **other)
