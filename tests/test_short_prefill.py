"""Short-P preparation contracts; these fakes are not NPU correctness evidence."""

from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
from test_decode_graph import Runner, Runtime, Stream, config

from vllm_muxpd.config import MuxPDConfig
from vllm_muxpd.decode_graph import DecodeGraphController
from vllm_muxpd.short_prefill import PreparedShortPrefill, warmup_sizes


def compilation(ranges=((1, 16), (17, 16384)), exact=(1, 2, 4, 8, 16)):
    return NS(
        get_compile_ranges=lambda: [NS(start=a, end=b) for a, b in ranges],
        compile_sizes=list(exact),
    )


def test_warmup_covers_exact_and_interval_dispatch_without_enumerating_tokens():
    assert warmup_sizes(compilation(), 512) == (1, 2, 3, 4, 8, 16, 17)
    assert warmup_sizes(compilation(), 2) == (1, 2)


@pytest.mark.parametrize(
    "cfg,limit",
    [
        (compilation(((2, 512),), ()), 512),
        (compilation(((1, 32),), ()), 512),
        (compilation(exact=tuple(range(1, 66))), 512),
        (compilation(exact=("cudagraph_capture_sizes",)), 512),
    ],
)
def test_warmup_rejects_gaps_unknown_specialization_and_unbounded_entries(cfg, limit):
    with pytest.raises(ValueError):
        warmup_sizes(cfg, limit)


def test_prepared_call_counts_only_successful_active_aot_returns():
    active = False
    fail = False

    def aot(*args):
        if fail:
            raise ValueError("actual callable failed")
        return args

    model = NS(
        _dynamic_arg_dims={"input_ids": 0, "positions": -1},
        aot_compiled_fn=aot,
    )
    prepared = PreparedShortPrefill(model, lambda: active)
    model.aot_compiled_fn = prepared.observed
    assert model.aot_compiled_fn(3) == (3,)
    assert prepared.successful_calls == 0  # D startup invokes the same model.
    active = True
    assert model.aot_compiled_fn(4) == (4,)
    prepared.require_return(0)
    fail = True
    with pytest.raises(ValueError):
        model.aot_compiled_fn(5)
    assert prepared.successful_calls == 1
    with pytest.raises(RuntimeError, match="did not return"):
        prepared.require_return(1)
    model.aot_compiled_fn = aot
    with pytest.raises(RuntimeError, match="changed"):
        prepared.check_installed()


@pytest.mark.parametrize(
    "field,value",
    [
        ("aot_compiled_fn", None),
        ("do_not_compile", True),
        ("_dynamic_arg_dims", {"input_ids": 0}),
    ],
)
def test_preparation_never_creates_missing_aot_or_assumes_static_inputs(field, value):
    model = NS(
        _dynamic_arg_dims={"input_ids": 0, "positions": -1},
        aot_compiled_fn=lambda: None,
    )
    setattr(model, field, value)
    with pytest.raises(RuntimeError):
        PreparedShortPrefill(model, lambda: True)


@pytest.mark.parametrize("limit", [True, -1, 0.5, "512"])
def test_short_config_rejects_non_integer_threshold(limit):
    with pytest.raises(ValueError, match="compile_short_prefill_tokens"):
        MuxPDConfig(compile_short_prefill_tokens=limit)


def test_short_config_opt_in_preserves_default_and_requires_isolated_graph():
    assert MuxPDConfig().compile_short_prefill_tokens == 0
    with pytest.raises(ValueError, match="compile_short_prefill_tokens"):
        MuxPDConfig(compile_short_prefill_tokens=512)
    result = MuxPDConfig(
        execution="decoupled",
        decode_graph=True,
        stream_mode="overlap",
        overlap_communication="isolated",
        compile_short_prefill_tokens=512,
    )
    assert result.compile_short_prefill_tokens == 512


def short_controller():
    runtime = Runtime()
    cfg = config()
    cfg.scheduler_config.max_num_batched_tokens = 16384
    cfg.scheduler_config.batch_admission_policy_config.update(
        stream_mode="overlap",
        overlap_communication="isolated",
        compile_short_prefill_tokens=512,
    )
    cfg.compilation_config.get_compile_ranges = compilation().get_compile_ranges
    cfg.compilation_config.compile_sizes = compilation().compile_sizes
    owner, decode = Runner(cfg, runtime), Runner(cfg, runtime)
    model = NS(
        _dynamic_arg_dims={"input_ids": 0, "positions": -1},
        aot_compiled_fn=lambda: "actual AOT result",
    )
    scope = []

    @contextmanager
    def prefill_scope():
        scope.append("enter P lane/PG/workspace")
        try:
            yield
        finally:
            scope.append("restore caller")

    runtime.validate_short_prefill_backend = lambda runner: None
    runtime.prefill_compile_model = lambda runner: model
    runtime.warmup_short_prefill = lambda runner, size: runner._dummy_run(size)
    owner.forward_effect = lambda: model.aot_compiled_fn()
    ctrl = DecodeGraphController(
        owner,
        decode,
        Stream("decode"),
        object(),
        runtime=runtime,
        prefill_startup_scope=prefill_scope,
    )
    ctrl.install_prefill_guards()
    ctrl.attach_shared_model()
    runtime.params = object()
    return ctrl, owner, decode, model, scope


def test_startup_prepares_every_dispatch_then_serving_requires_actual_aot_return():
    ctrl, owner, _, model, scope = short_controller()
    original = model.aot_compiled_fn
    owner.profile_run()
    assert scope == ["enter P lane/PG/workspace", "restore caller"]
    warmed = ctrl.snapshot()["short_prefill_warmup_sizes"]
    assert ctrl.snapshot()["short_prefill_aot_successful_calls"] == len(warmed)
    ctrl.capture_decode()
    with ctrl.full_prefill_compile_context():
        assert owner._dummy_run(512)[2:] == ("NONE", False, False)
    assert ctrl.snapshot()["short_prefill_aot_successful_calls"] == len(warmed) + 1
    owner.forward_effect = None
    with pytest.raises(RuntimeError, match="did not return"):
        with ctrl.full_prefill_compile_context():
            owner._dummy_run(512)
    ctrl.close()
    assert model.aot_compiled_fn is original
    assert ctrl._short_prefill is None and ctrl._prefill_startup_scope is None


def test_failed_startup_restores_scope_and_blocks_capture():
    ctrl, owner, _, model, scope = short_controller()
    model.aot_compiled_fn = None
    owner.forward_effect = None
    with pytest.raises(RuntimeError, match="already prepared"):
        owner.profile_run()
    assert scope[-1] == "restore caller"
    assert ctrl.snapshot()["failed"]
    with pytest.raises(RuntimeError, match="restart"):
        ctrl.capture_decode()
