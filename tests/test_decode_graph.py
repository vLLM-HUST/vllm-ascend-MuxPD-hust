"""Decode graph startup/lifecycle checks using only CPU fakes."""

from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

from vllm_muxpd.decode_graph import (
    DecodeGraphController,
    GraphFencedExecutionBackend,
    GraphLaneState,
    fixed_decode_capture_context,
    validate_decode_graph_config,
)


class Stream:
    device = "fake-npu"

    def __init__(self, name):
        self.name = name
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other)


class NPU:
    def __init__(self):
        self.current = Stream("caller")
        self.records = []
        self.waits = []

        class Graph:
            default_capture_stream = Stream("native-default-capture")

            def __init__(self):
                self.capture_stream = self.default_capture_stream

        self.graph = Graph

    def current_stream(self):
        return self.current

    def set_stream(self, stream):
        self.current = stream

    @contextmanager
    def stream(self, stream):
        old = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = old

    def record_event(self, stream):
        event = (stream, len(self.records))
        self.records.append(event)
        return event

    def wait_event(self, stream, event):
        self.waits.append((stream, event))


def config():
    return NS(
        model_config=NS(enforce_eager=False),
        compilation_config=NS(
            mode="VLLM_COMPILE",
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[1, 2, 4],
            max_cudagraph_capture_size=4,
        ),
        scheduler_config=NS(
            max_num_seqs=4,
            async_scheduling=False,
            batch_admission_policy_config={
                "decode_graph": True,
                "execution": "decoupled",
                "stream_mode": "fenced",
            },
        ),
    )


class Runtime:
    none_mode = "NONE"
    full_mode = "FULL"

    def __init__(self):
        self.npu = NPU()
        self.forward = NS(
            _forward_context=NS(
                cudagraph_runtime_mode="FULL", skip_compiled=False, capturing=True
            )
        )
        self.workspace = NS(_manager=object())
        self.ascend = NS(_CURRENT_STREAM=self.npu.current)
        self.npu_runner = NS(graph_capture=object())
        self.gpu_runner = NS(graph_capture=object())
        self.params = None

    def graph_params(self):
        return self.params

    def unwrap(self, model):
        return getattr(model, "raw", model)

    def wrap(self, raw, decode):
        return NS(raw=raw, concrete_aclgraph_entries={})

    def new_stream(self, device):
        assert device == "fake-npu"
        return Stream("update")

    def capture_context(self, decode, stream, workspace):
        return fixed_decode_capture_context(
            npu=self.npu,
            npu_runner_module=self.npu_runner,
            gpu_runner_module=self.gpu_runner,
            workspace_module=self.workspace,
            ascend_utils_module=self.ascend,
            stream=stream,
            workspace=workspace,
        )


class Runner:
    def __init__(self, cfg, runtime):
        self.vllm_config = cfg
        self.runtime = runtime
        self.use_aclgraph = True
        self.model = object()
        self.model_memory_usage = 1000
        self._has_sinks = False
        self.is_multimodal_pruning_enabled = False
        self.input_batch = object()
        self.input_ids = NS(gpu=object())
        self.positions = object()
        self.seq_lens = object()
        self.query_start_loc = NS(gpu=object())
        self.cudagraph_dispatcher = NS(
            get_capture_descs=lambda: [("FULL", [NS(uniform=True)])]
        )
        self.calls = []
        self.fail_capture = False
        self.fail_forward = False
        self.forward_effect = None

    def _determine_batch_execution_and_padding(self, num_tokens, force_eager=False):
        self.calls.append(("dispatch", num_tokens, force_eager))
        return "NONE" if force_eager else "FULL"

    def _model_forward(self, num_tokens):
        ctx = self.runtime.forward._forward_context
        result = (
            "forward",
            num_tokens,
            ctx.cudagraph_runtime_mode,
            ctx.skip_compiled,
            ctx.capturing,
        )
        self.calls.append(result)
        if self.fail_forward:
            raise ValueError("forward failed")
        if self.forward_effect is not None:
            self.forward_effect()
        return result

    def _dummy_run(self, num_tokens):
        self._determine_batch_execution_and_padding(num_tokens, False)
        return self._model_forward(num_tokens)

    def profile_run(self):
        return self._dummy_run(16384)

    def capture_model(self):
        r = self.runtime
        with r.npu_runner.graph_capture("fake-npu") as ctx:
            self.calls.append(
                (
                    "capture",
                    ctx.stream,
                    r.npu.graph().capture_stream,
                    r.npu.current_stream(),
                    r.workspace._manager,
                    r.ascend._CURRENT_STREAM,
                )
            )
            # Model native NPU wrapper's assignment to the GPU module on exit.
            r.gpu_runner.graph_capture = r.npu_runner.graph_capture
            if self.fail_capture:
                raise RuntimeError("capture failed after allocating graph resources")
            self.model.concrete_aclgraph_entries["batch1"] = NS(aclgraph=object())
        return 32


def controller(*, attached=True, compile_full_prefill=False):
    r = Runtime()
    cfg = config()
    cfg.scheduler_config.batch_admission_policy_config["compile_full_prefill"] = (
        compile_full_prefill
    )
    p, d = Runner(cfg, r), Runner(cfg, r)
    stream, workspace = Stream("decode"), object()
    ctrl = DecodeGraphController(p, d, stream, workspace, runtime=r)
    ctrl.install_prefill_guards()
    if attached:
        ctrl.attach_shared_model()
        r.params = object()  # Native D KV initialization registers GraphParams.
    return ctrl, p, d, r


def test_prefill_guards_cover_one_token_and_native_dummy_warmup():
    ctrl, p, d, r = controller()
    old = vars(r.forward._forward_context).copy()
    for count in (1, 16384):
        assert p._dummy_run(count) == ("forward", count, "NONE", True, False)
        assert vars(r.forward._forward_context) == old
    assert all(call[2] for call in p.calls if call[0] == "dispatch")
    assert p.use_aclgraph is False and d.use_aclgraph is True
    assert p.vllm_config.compilation_config.cudagraph_mode == "FULL_DECODE_ONLY"
    assert not p.vllm_config.model_config.enforce_eager
    assert ctrl.snapshot()["prefill_forced_eager_calls"] == 2
    assert ctrl.snapshot()["decode_replay_calls"] == 0
    with pytest.raises(RuntimeError, match="already installed"):
        ctrl.install_prefill_guards()


def test_attached_model_only_shares_raw_weights_not_wrapper_or_inputs():
    ctrl, p, d, r = controller(attached=False)
    raw = object()
    old_wrapper = NS(raw=raw)
    p.model = old_wrapper
    ctrl.attach_shared_model()
    assert p.model is raw and d.model.raw is raw
    assert d.model is not old_wrapper
    assert d.model_memory_usage == 0 and p.model_memory_usage == 1000
    assert d.update_stream is not ctrl.stream
    with pytest.raises(RuntimeError, match="once"):
        ctrl.attach_shared_model()


def test_prefill_context_fields_are_restored_even_if_forward_fails():
    _, p, _, r = controller()
    del r.forward._forward_context.capturing
    old = vars(r.forward._forward_context).copy()
    p.fail_forward = True
    with pytest.raises(ValueError, match="forward failed"):
        p._model_forward(1)
    assert vars(r.forward._forward_context) == old


def test_capture_uses_decode_stream_in_both_native_layers_and_restores():
    ctrl, p, d, r = controller()
    original = (
        r.npu_runner.graph_capture,
        r.gpu_runner.graph_capture,
        r.npu.graph.default_capture_stream,
        r.npu.current,
        r.workspace._manager,
        r.ascend._CURRENT_STREAM,
    )
    assert ctrl.capture_decode() == 32
    assert d.calls[-1] == (
        "capture", ctrl.stream, ctrl.stream, ctrl.stream, ctrl.workspace, ctrl.stream
    )
    assert original == (
        r.npu_runner.graph_capture,
        r.gpu_runner.graph_capture,
        r.npu.graph.default_capture_stream,
        r.npu.current,
        r.workspace._manager,
        r.ascend._CURRENT_STREAM,
    )
    stats = ctrl.snapshot()
    assert stats["capture_attempts"] == stats["capture_successes"] == 1
    assert stats["capture_failures"] == 0
    assert stats["captured"] and stats["captured_graph_entries"] == 1
    with pytest.raises(RuntimeError, match="already been attempted"):
        ctrl.capture_decode()


def test_owner_redirect_preserves_original_bound_method_and_warmup():
    ctrl, p, d, _ = controller()
    original_bound = p.capture_model
    with ctrl.redirect_owner_capture():
        assert p._dummy_run(1)[2:] == ("NONE", True, False)
        with pytest.raises(RuntimeError, match="redirection"):
            with ctrl.redirect_owner_capture():
                pass
        assert p.capture_model() == 32
    assert p.capture_model == original_bound
    assert "capture_model" not in vars(p)
    assert not any(call[0] == "capture" for call in p.calls)
    assert d.calls[-1][0] == "capture"


def test_redirect_restores_instance_override_on_warmup_exception():
    ctrl, p, _, _ = controller()
    original = lambda: 999  # noqa: E731
    p.capture_model = original
    with pytest.raises(ValueError, match="warmup"):
        with ctrl.redirect_owner_capture():
            raise ValueError("warmup failed before capture")
    assert p.capture_model is original
    assert not ctrl._capture_started


def test_partial_capture_failure_restores_all_context_and_poisons_controller():
    ctrl, p, d, r = controller()
    original_method = p.capture_model
    original_stream = r.npu.current_stream()
    original_manager = r.workspace._manager
    original_default = r.npu.graph.default_capture_stream
    original_npu_factory = r.npu_runner.graph_capture
    original_gpu_factory = r.gpu_runner.graph_capture
    d.fail_capture = True
    with pytest.raises(RuntimeError, match="allocating"):
        with ctrl.redirect_owner_capture():
            p.capture_model()
    assert p.capture_model == original_method
    assert r.npu.current_stream() is original_stream
    assert r.workspace._manager is original_manager
    assert r.npu.graph.default_capture_stream is original_default
    assert r.npu_runner.graph_capture is original_npu_factory
    assert r.gpu_runner.graph_capture is original_gpu_factory
    with pytest.raises(RuntimeError, match="restart"):
        ctrl.capture_decode()
    with pytest.raises(RuntimeError, match="restart"):
        p._dummy_run(1)
    stats = ctrl.snapshot()  # Even a poisoned controller remains auditable.
    assert stats["capture_attempts"] == stats["capture_failures"] == 1
    assert stats["capture_successes"] == 0
    assert stats["failed"] and not stats["captured"]


@pytest.mark.parametrize(
    "mode,capturing,entry_descriptor,complete,expected,uncaptured",
    [
        ("FULL", False, "batch1", True, "decode_replay_calls", 0),
        ("FULL", False, "batch1", False, "decode_eager_fallback_calls", 1),
        ("FULL", False, "batch2", True, "decode_eager_fallback_calls", 1),
        ("FULL", False, None, False, "decode_eager_fallback_calls", 1),
        ("NONE", False, "batch1", True, "decode_eager_fallback_calls", 0),
        ("PIECEWISE", False, "batch1", True, "decode_eager_fallback_calls", 0),
        ("FULL", True, "batch1", True, "decode_capture_forward_calls", 0),
        ("FULL", True, None, False, "decode_capture_forward_calls", 0),
    ],
)
def test_decode_forward_statistics_require_matching_captured_entry(
    mode, capturing, entry_descriptor, complete, expected, uncaptured
):
    ctrl, _, d, r = controller()
    ctx = r.forward._forward_context
    ctx.cudagraph_runtime_mode = mode
    ctx.capturing = capturing
    ctx.batch_descriptor = "batch1"
    if entry_descriptor is not None:
        d.model.concrete_aclgraph_entries[entry_descriptor] = NS(
            aclgraph=object() if complete else None
        )
    before = vars(ctx).copy()
    assert d._model_forward(1) == ("forward", 1, mode, False, capturing)
    assert vars(ctx) == before
    stats = ctrl.snapshot()
    assert stats[expected] == 1
    assert sum(stats[key] for key in (
        "decode_replay_calls", "decode_eager_fallback_calls",
        "decode_capture_forward_calls",
    )) == 1
    assert stats["decode_uncaptured_full_calls"] == uncaptured


@pytest.mark.parametrize("preserve_capture_flag", [False, True])
def test_first_full_capture_is_not_misclassified_as_replay(preserve_capture_flag):
    ctrl, _, d, r = controller()
    ctx = r.forward._forward_context
    ctx.capturing = False
    ctx.batch_descriptor = "batch1"

    def capture():
        ctx.capturing = preserve_capture_flag
        d.model.concrete_aclgraph_entries["batch1"] = NS(aclgraph=object())

    d.forward_effect = capture
    d._model_forward(1)
    first = ctrl.snapshot()
    assert first["decode_capture_forward_calls"] == 1
    assert first["decode_replay_calls"] == 0
    ctx.capturing = False
    d.forward_effect = None
    d._model_forward(1)
    assert ctrl.snapshot()["decode_replay_calls"] == 1


def test_snapshot_is_independent_json_data_and_failures_are_separate():
    import json

    ctrl, p, d, r = controller()
    ctx = r.forward._forward_context
    ctx.capturing = False
    ctx.batch_descriptor = "batch1"
    d.model.concrete_aclgraph_entries["batch1"] = NS(aclgraph=object())
    d.fail_forward = True
    with pytest.raises(ValueError, match="forward failed"):
        d._model_forward(1)
    stats = ctrl.snapshot()
    assert stats["decode_forward_failures"] == 1
    assert stats["decode_replay_calls"] == 0
    stats["decode_forward_failures"] = -1
    assert ctrl.snapshot()["decode_forward_failures"] == 1
    assert json.loads(json.dumps(ctrl.snapshot())) == ctrl.snapshot()
    p.fail_forward = True
    with pytest.raises(ValueError, match="forward failed"):
        p._model_forward(1)
    assert ctrl.snapshot()["prefill_forced_eager_calls"] == 1


def test_decode_forward_observer_rejects_an_existing_owner():
    ctrl, _, d, _ = controller(attached=False)
    d._muxpd_decode_graph_controller = object()
    with pytest.raises(RuntimeError, match="already installed"):
        ctrl.attach_shared_model()


def test_close_releases_graphs_and_model_references_before_native_owner_cleanup():
    import weakref

    class Resource:
        pass

    ctrl, p, d, _ = controller(attached=False)
    raw = Resource()
    reference = weakref.ref(raw)
    p.model = raw
    ctrl.attach_shared_model()
    graph = Resource()
    graph_reference = weakref.ref(graph)
    wrapper = d.model
    wrapper.concrete_aclgraph_entries["batch1"] = NS(aclgraph=graph)
    del graph
    d.shutdown = lambda: pytest.fail("D must not run native shared-cache cleanup")
    ctrl.close()
    assert graph_reference() is None
    assert not wrapper.concrete_aclgraph_entries
    assert p.model is raw  # The native owner still controls shared weight lifetime.
    assert d.model is None
    assert p.use_aclgraph is True
    for runner in (p, d):
        assert "_model_forward" not in vars(runner)
        assert "_muxpd_decode_graph_controller" not in vars(runner)
    assert "_determine_batch_execution_and_padding" not in vars(p)
    assert "update_stream" not in vars(d)
    assert ctrl.owner is ctrl.decode is ctrl.workspace is ctrl.runtime is None
    # Model an owner's native shutdown. A live controller must not retain weights.
    p.model = None
    del raw, wrapper
    assert reference() is None
    assert ctrl.snapshot()["closed"]
    assert ctrl.snapshot()["captured_graph_entries"] == 0
    ctrl.close()
    with pytest.raises(RuntimeError, match="closed"):
        ctrl.capture_decode()


@pytest.mark.parametrize("fail_capture", [False, True])
def test_close_restores_guards_after_failed_initialization_or_capture(fail_capture):
    ctrl, p, d, r = controller(attached=fail_capture)
    if fail_capture:
        d.fail_capture = True
        with pytest.raises(RuntimeError, match="capture failed"):
            ctrl.capture_decode()
    before = ctrl.snapshot()
    original_mode = r.forward._forward_context.cudagraph_runtime_mode
    ctrl.close()
    ctrl.close()
    assert p._dummy_run(1)[2] == original_mode
    assert ctrl.snapshot()["capture_failures"] == before["capture_failures"]
    assert ctrl.snapshot()["closed"]
    assert not ctrl.snapshot()["attached"]
    assert "_muxpd_decode_graph_controller" not in vars(p)
    assert "_muxpd_decode_graph_controller" not in vars(d)


def test_close_preserves_original_instance_methods_and_later_external_overrides():
    r = Runtime()
    cfg = config()
    p, d = Runner(cfg, r), Runner(cfg, r)
    original_p = lambda count: count + 1  # noqa: E731
    original_d = lambda count: count + 2  # noqa: E731
    p._model_forward = original_p
    d._model_forward = original_d
    ctrl = DecodeGraphController(p, d, Stream("decode"), object(), runtime=r)
    ctrl.install_prefill_guards()
    ctrl.attach_shared_model()
    original_dispatch = p._determine_batch_execution_and_padding
    external_dispatch = lambda: "external"  # noqa: E731
    p._determine_batch_execution_and_padding = external_dispatch
    ctrl.close()
    assert p._model_forward is original_p
    assert d._model_forward is original_d
    assert p._determine_batch_execution_and_padding is external_dispatch
    with pytest.raises(RuntimeError, match="closed"):
        original_dispatch(1)


def test_close_rejects_active_capture_redirection_without_corruption():
    ctrl, p, _, _ = controller()
    original = p.capture_model
    with ctrl.redirect_owner_capture():
        with pytest.raises(RuntimeError, match="capture redirection"):
            ctrl.close()
        assert not ctrl.snapshot()["closed"]
    assert p.capture_model == original
    ctrl.close()


@pytest.mark.parametrize("field", ["input_batch", "input_ids", "positions"])
def test_aliases_fail_before_capture_starts(field):
    ctrl, p, d, _ = controller()
    setattr(d, field, getattr(p, field))
    with pytest.raises(RuntimeError, match="aliases"):
        ctrl.capture_decode()
    assert not ctrl._capture_started


def test_nonuniform_capture_and_missing_graph_params_are_rejected():
    ctrl, _, d, r = controller()
    r.params = None
    with pytest.raises(RuntimeError, match="initialize GraphParams"):
        ctrl.capture_decode()
    r.params = object()
    d.cudagraph_dispatcher.get_capture_descs = lambda: [("FULL", [NS(uniform=False)])]
    with pytest.raises(RuntimeError, match="uniform"):
        ctrl.capture_decode()


def test_existing_graph_owner_is_rejected_before_install():
    r = Runtime()
    r.params = object()
    cfg = config()
    p, d = Runner(cfg, r), Runner(cfg, r)
    ctrl = DecodeGraphController(p, d, Stream("decode"), object(), runtime=r)
    with pytest.raises(RuntimeError, match="already has an owner"):
        ctrl.install_prefill_guards()
    assert p.use_aclgraph


@pytest.mark.parametrize(
    "target,name,value,match",
    [
        ("policy", "decode_graph", False, "explicitly true"),
        ("policy", "execution", "layered", "decoupled"),
        ("policy", "stream_mode", "overlap", "overlap"),
        ("model", "enforce_eager", True, "enforce_eager"),
        ("compilation", "cudagraph_mode", "FULL", "FULL_DECODE_ONLY"),
        ("compilation", "mode", "NONE", "VLLM_COMPILE"),
        ("compilation", "cudagraph_capture_sizes", [1, 8], "capture sizes"),
        ("compilation", "cudagraph_capture_sizes", [1, 1], "capture sizes"),
        ("compilation", "max_cudagraph_capture_size", 8, "largest"),
    ],
)
def test_invalid_graph_configuration(target, name, value, match):
    cfg = config()
    policy = cfg.scheduler_config.batch_admission_policy_config
    obj = {
        "policy": policy,
        "model": cfg.model_config,
        "compilation": cfg.compilation_config,
    }[target]
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)
    with pytest.raises(ValueError, match=match):
        validate_decode_graph_config(cfg, policy)


def test_fenced_backend_retains_order_and_each_lanes_workspace():
    r = Runtime()
    backend = GraphFencedExecutionBackend(
        runtime=r.npu,
        forward_context_module=r.forward,
        workspace_module=r.workspace,
        ascend_utils_module=r.ascend,
    )
    caller = backend.capture()
    d_stream, d_manager = Stream("decode"), object()
    d = GraphLaneState("decode-context", d_stream, d_manager)
    backend.restore(d)
    assert backend.capture() == d
    backend.fence()
    backend.restore(caller)
    assert backend.capture() == caller
    assert r.npu.waits == [(caller.stream, r.npu.records[-1])]
    assert r.ascend._CURRENT_STREAM is caller.stream


def test_fenced_backend_same_stream_fifo_keeps_latest_cross_stream_tail():
    r = Runtime()
    backend = GraphFencedExecutionBackend(
        runtime=r.npu,
        forward_context_module=r.forward,
        workspace_module=r.workspace,
        ascend_utils_module=r.ascend,
    )
    caller = backend.capture()

    class StreamAlias(Stream):
        def __eq__(self, other):
            return isinstance(other, StreamAlias) and self.name == other.name

    d = GraphLaneState("D", StreamAlias("decode"), object())
    p = GraphLaneState("P", StreamAlias("prefill"), object())
    backend.restore(d)
    backend.fence()
    backend.restore(d)
    assert not r.npu.waits
    backend.fence()
    latest_d = r.npu.records[-1]
    assert len(r.npu.records) == 2
    backend.restore(p)
    assert r.npu.waits == [(p.stream, latest_d)]
    backend.fence()
    # Two wrappers for the same native stream still have FIFO ordering, but
    # context/workspace restoration remains necessary and must not be skipped.
    p_alias = GraphLaneState("P-resumed", StreamAlias("prefill"), p.workspace_manager)
    backend.restore(p_alias)
    assert backend.capture() == p_alias
    assert len(r.npu.waits) == 1
    backend.fence()
    latest_p = r.npu.records[-1]
    backend.restore(d)
    backend.restore(caller)
    assert len(r.npu.records) == 4
    assert r.npu.waits == [
        (p.stream, latest_d),
        (d.stream, latest_p),
        (caller.stream, latest_p),
    ]
    assert backend.capture() == caller


def test_capture_context_rejects_nesting_without_corrupting_outer_state():
    r = Runtime()
    stream, manager = Stream("decode"), object()
    with r.capture_context(None, stream, manager):
        with pytest.raises(RuntimeError, match="nest"):
            with r.capture_context(None, stream, manager):
                pass
        assert r.npu.current_stream() is stream
        assert r.workspace._manager is manager


def test_no_hook_decode_task_can_complete_without_fake_layer_progress():
    from vllm_muxpd.lanes import LayerInterleaver

    r = Runtime()
    backend = GraphFencedExecutionBackend(
        runtime=r.npu,
        forward_context_module=r.forward,
        workspace_module=r.workspace,
        ascend_utils_module=r.ascend,
    )
    task = LayerInterleaver(backend)
    task.add("decode", lambda: None, backend.capture())
    task.step("decode", 1)
    assert task.tasks["decode"].done
    assert task.tasks["decode"].completed_layers == 0


def test_full_prefill_compile_profiles_before_capture_and_keeps_p_graph_disabled():
    ctrl, p, d, r = controller(compile_full_prefill=True)
    old = vars(r.forward._forward_context).copy()
    with pytest.raises(RuntimeError, match="startup profile"):
        with ctrl.full_prefill_compile_context():
            pass
    with pytest.raises(RuntimeError, match="startup profiling"):
        ctrl.capture_decode()
    assert not ctrl.snapshot()["capture_started"]
    assert p.profile_run() == ("forward", 16384, "NONE", False, False)
    assert ctrl.snapshot()["prefill_compile_profiled"]
    assert vars(r.forward._forward_context) == old
    assert p._dummy_run(1) == ("forward", 1, "NONE", True, False)
    ctrl.capture_decode()
    with ctrl.full_prefill_compile_context():
        assert p._model_forward(9) == ("forward", 9, "NONE", False, False)
    assert vars(r.forward._forward_context) == old
    assert p._model_forward(9) == ("forward", 9, "NONE", True, False)
    snap = ctrl.snapshot()
    assert snap["prefill_compiled_calls"] == 2
    assert snap["prefill_forced_eager_calls"] == 4
    assert snap["prefill_compile_profile_runs"] == 1
    assert d.model.raw is p.model
    ctrl.close()
    assert "profile_run" not in vars(p)


def test_full_prefill_compile_scope_is_exclusive_and_recovers_after_forward_error():
    ctrl, p, _, _ = controller(compile_full_prefill=True)
    p.profile_run()
    ctrl.capture_decode()
    with ctrl.full_prefill_compile_context():
        with pytest.raises(RuntimeError, match="nested"):
            with ctrl.full_prefill_compile_context():
                pass
        with pytest.raises(RuntimeError, match="Cannot close"):
            ctrl.close()
        assert not ctrl.snapshot()["closed"]
    p.fail_forward = True
    with pytest.raises(ValueError, match="forward failed"):
        with ctrl.full_prefill_compile_context():
            p._model_forward(2)
    p.fail_forward = False
    assert p._model_forward(2) == ("forward", 2, "NONE", True, False)


@pytest.mark.parametrize("failure", ["raises", "no_forward"])
def test_prefill_compile_startup_failure_poisoned_and_restorable(failure):
    r, cfg = Runtime(), config()
    cfg.scheduler_config.batch_admission_policy_config["compile_full_prefill"] = True
    p, d = Runner(cfg, r), Runner(cfg, r)
    if failure == "no_forward":
        p.profile_run = lambda: None
    else:
        p.fail_forward = True
    original = p.profile_run
    ctrl = DecodeGraphController(p, d, Stream("decode"), object(), runtime=r)
    ctrl.install_prefill_guards()
    ctrl.attach_shared_model()
    with pytest.raises((RuntimeError, ValueError)):
        p.profile_run()
    assert ctrl.snapshot()["failed"]
    assert ctrl.snapshot()["prefill_compile_profile_failures"] == 1
    assert not ctrl.snapshot()["prefill_compile_profiled"]
    with pytest.raises(RuntimeError, match="restart"):
        ctrl.capture_decode()
    ctrl.close()
    assert p.profile_run == original


def test_full_prefill_compile_disabled_and_post_capture_profile_rejected():
    ctrl, p, _, _ = controller()
    assert "profile_run" not in vars(p)
    with pytest.raises(RuntimeError, match="not enabled"):
        with ctrl.full_prefill_compile_context():
            pass
    ctrl, p, _, _ = controller(compile_full_prefill=True)
    p.profile_run()
    ctrl.capture_decode()
    before = len(p.calls)
    with pytest.raises(RuntimeError, match="precede capture"):
        p.profile_run()
    assert len(p.calls) == before


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_graph_validator_requires_strict_compile_full_prefill_boolean(value):
    cfg = config()
    policy = cfg.scheduler_config.batch_admission_policy_config
    policy["compile_full_prefill"] = value
    with pytest.raises(ValueError, match="compile_full_prefill"):
        validate_decode_graph_config(cfg, policy)
