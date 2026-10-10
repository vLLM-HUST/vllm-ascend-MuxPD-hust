"""Exercise the worker's graph integration with real controller CPU fakes."""

import importlib
import os
from types import SimpleNamespace as NS

import pytest
from test_decode_graph import Stream, controller

from vllm_muxpd.decode_graph import GraphFencedExecutionBackend, GraphLaneState


@pytest.fixture
def worker_factory():
    if os.environ.get("VLLM_VERSION") != "0.25.1":
        pytest.skip("source runtime/env.sh for pinned HUST worker contracts")
    module = importlib.import_module("vllm_muxpd.worker")

    def create(*, graph=True, attached=True):
        ctrl, owner, decode, runtime = controller(attached=attached)
        worker = module.MuxPDNPUWorker.__new__(module.MuxPDNPUWorker)
        worker.model_runner = owner
        worker.decode_runner = decode
        worker._decode_graph_controller = ctrl if graph else None
        worker._decode_graph_enabled = graph
        worker.rank = 0
        worker._interleaver = None
        worker._prefill_task = None
        worker._awaiting_sample = False
        worker._pending_lanes = []
        worker._hook_handles = []
        worker._layers_per_slice = 8
        worker._stream_mode = "fenced"
        worker._overlap_communication = "shared"
        worker._collective_router = None
        worker._collective_gates = None
        worker._prefill_groups = []
        worker.muxpd_stats = {"slices": 0}
        worker._streams = {"prefill": Stream("prefill"), "decode": ctrl.stream}
        worker._execution_backend = GraphFencedExecutionBackend(
            runtime=runtime.npu,
            forward_context_module=runtime.forward,
            workspace_module=runtime.workspace,
            ascend_utils_module=runtime.ascend,
        )
        worker._lane_states = {
            "decode": GraphLaneState(None, ctrl.stream, ctrl.workspace)
        }
        return NS(
            module=module,
            worker=worker,
            controller=ctrl,
            owner=owner,
            decode=decode,
            runtime=runtime,
        )

    return create


@pytest.mark.integration
def test_native_compile_keeps_memory_owner_and_native_tail_return(
    worker_factory, monkeypatch
):
    state = worker_factory()
    events = []
    native_return = NS(compilation_time=1.25)
    original_capture = state.owner.capture_model

    def native_compile(worker):
        assert worker.model_runner is state.owner
        events.append("native_start")
        assert worker.model_runner._dummy_run(1)[2:] == ("NONE", True, False)
        assert worker.model_runner.capture_model() == 32
        # Memory accounting, CPU binding and seed-reset code after capture must
        # still see the original owner and execute before the return is passed on.
        assert worker.model_runner is state.owner
        assert worker.model_runner.model_memory_usage == 1000
        events.append("native_tail")
        return native_return

    monkeypatch.setattr(
        state.module.NPUWorker, "compile_or_warm_up_model", native_compile
    )
    assert state.worker.compile_or_warm_up_model() is native_return
    assert events == ["native_start", "native_tail"]
    assert state.worker.model_runner is state.owner
    assert state.owner.capture_model == original_capture
    assert "capture_model" not in vars(state.owner)
    assert not any(call[0] == "capture" for call in state.owner.calls)
    assert state.decode.calls[-1][0] == "capture"
    assert state.controller._captured
    # Do not mask a worker/controller API mismatch with a fake snapshot method.
    assert isinstance(state.controller.snapshot(), dict)


@pytest.mark.integration
@pytest.mark.parametrize("fail_after_capture", [False, True])
def test_native_compile_failure_restores_capture_and_keeps_owner(
    worker_factory, monkeypatch, fail_after_capture
):
    state = worker_factory()
    original_capture = state.owner.capture_model

    def native_compile(worker):
        if fail_after_capture:
            assert worker.model_runner.capture_model() == 32
        raise ValueError("native warmup tail failed")

    monkeypatch.setattr(
        state.module.NPUWorker, "compile_or_warm_up_model", native_compile
    )
    with pytest.raises(ValueError, match="native warmup tail"):
        state.worker.compile_or_warm_up_model()
    assert state.worker.model_runner is state.owner
    assert state.owner.capture_model == original_capture
    assert "capture_model" not in vars(state.owner)
    assert state.controller._captured is fail_after_capture


@pytest.mark.integration
@pytest.mark.parametrize(
    "attribute,value",
    [
        ("_interleaver", object()),
        ("_pending_lanes", ["decode"]),
        ("_prefill_task", object()),
        ("_awaiting_sample", True),
    ],
)
def test_capture_rejects_live_or_pending_work_before_native_call(
    worker_factory, monkeypatch, attribute, value
):
    state = worker_factory()
    calls = []
    monkeypatch.setattr(
        state.module.NPUWorker,
        "compile_or_warm_up_model",
        lambda worker: calls.append(worker.model_runner),
    )
    setattr(state.worker, attribute, value)
    with pytest.raises(RuntimeError, match="startup"):
        state.worker.compile_or_warm_up_model()
    assert not calls
    assert not state.controller._capture_started


@pytest.mark.integration
def test_graphs_disabled_preserves_direct_native_compile_path(
    worker_factory, monkeypatch
):
    state = worker_factory(graph=False)
    native_return = object()
    calls = []

    def native_compile(worker):
        calls.append(worker.model_runner)
        return native_return

    monkeypatch.setattr(
        state.module.NPUWorker, "compile_or_warm_up_model", native_compile
    )
    assert state.worker.compile_or_warm_up_model() is native_return
    assert calls == [state.owner]
    assert not state.controller._capture_started


@pytest.mark.integration
@pytest.mark.parametrize("graph", [False, True])
def test_loads_weights_once_and_attaches_correct_decode_model(
    worker_factory, monkeypatch, graph
):
    state = worker_factory(graph=graph, attached=False)
    events = []
    hooks = []
    layers = [
        NS(register_forward_hook=lambda hook: hooks.append(hook) or object())
        for _ in range(3)
    ]
    raw_model = NS(model=NS(layers=layers))

    def native_load(worker):
        assert worker.model_runner is state.owner
        events.append("load_owner")
        state.owner.model = NS(raw=raw_model) if graph else raw_model

    def regular_attach():
        events.append("regular_attach")
        state.decode.model = state.owner.model

    state.owner.get_model = lambda: state.owner.model
    state.decode.attach_model = regular_attach
    monkeypatch.setattr(state.module.NPUWorker, "load_model", native_load)
    state.worker.load_model()
    assert events == (["load_owner"] if graph else ["load_owner", "regular_attach"])
    assert state.worker.model_runner is state.owner
    assert state.owner.model is raw_model
    assert state.controller._attached is graph
    if graph:
        assert state.decode.model.raw is raw_model
        assert state.decode.model is not raw_model
        assert state.decode.model_memory_usage == 0
    else:
        assert state.decode.model is raw_model
    assert state.worker._decoder_layer_count == 3
    assert len(state.worker._hook_handles) == len(hooks) == 3
    # The actual installed hook stays inert during native startup warmup.
    hooks[0](layers[0], (), None)
    assert state.worker.muxpd_stats["slices"] == 0


@pytest.mark.integration
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("decode_fails", [False, True])
def test_kv_owner_initializes_before_decode_for_same_config(
    worker_factory, monkeypatch, graph, decode_fails
):
    state = worker_factory(graph=graph)
    cfg = object()
    events = []
    backend = state.worker._execution_backend
    caller = backend.capture()

    def native_initialize(worker, kv_config):
        assert worker.model_runner is state.owner
        assert kv_config is cfg
        assert backend.capture() == caller
        events.append("prefill")
        state.owner.kv_caches = [object()]

    def decode_initialize(kv_config):
        assert kv_config is cfg
        assert state.owner.kv_caches
        assert backend.capture() == (
            state.worker._lane_states["decode"] if graph else caller
        )
        events.append("decode")
        if decode_fails:
            raise RuntimeError("decode builder failed")

    monkeypatch.setattr(
        state.module.NPUWorker, "initialize_from_config", native_initialize
    )
    state.decode.initialize_kv_cache = decode_initialize
    if decode_fails:
        with pytest.raises(RuntimeError, match="decode builder failed"):
            state.worker.initialize_from_config(cfg)
    else:
        state.worker.initialize_from_config(cfg)
    assert events == ["prefill", "decode"]
    assert backend.capture() == caller
    assert state.runtime.ascend._CURRENT_STREAM is caller.stream
    if graph:
        assert len(state.runtime.npu.records) == 2
        assert [stream for stream, _ in state.runtime.npu.records] == [
            caller.stream,
            state.worker._streams["decode"],
        ]
        assert state.runtime.npu.waits == [
            (state.worker._streams["decode"], state.runtime.npu.records[0]),
            (caller.stream, state.runtime.npu.records[1]),
        ]
    else:
        assert not state.runtime.npu.records
        assert not state.runtime.npu.waits


@pytest.mark.integration
@pytest.mark.parametrize("graph", [False, True])
def test_shutdown_joins_graph_streams_before_native_cleanup(
    worker_factory, monkeypatch, graph
):
    state = worker_factory(graph=graph)
    events = []
    for name, stream in state.worker._streams.items():
        stream.synchronize = lambda name=name: events.append(("sync", name))
    state.decode.update_stream.synchronize = lambda: events.append(("sync", "update"))
    state.worker._hook_handles = [NS(remove=lambda: events.append(("remove_hook",)))]
    state.worker._interleaver = NS(close=lambda: events.append(("close_task",)))
    state.decode.shutdown = lambda: pytest.fail("D native shutdown is forbidden")

    def native_shutdown(worker):
        assert worker.model_runner is state.owner
        assert worker.decode_runner is state.decode
        if graph:
            assert state.controller.snapshot()["closed"]
            assert state.decode.model is None
            assert state.controller.owner is None
        events.append(("native_shutdown",))

    monkeypatch.setattr(state.module.NPUWorker, "shutdown", native_shutdown)
    state.worker.shutdown()
    state.worker.shutdown()  # Repeated shutdown must not free shared state twice.
    assert events == [
        ("close_task",),
        ("remove_hook",),
        *(
            [("sync", "prefill"), ("sync", "decode"), ("sync", "update")]
            if graph
            else []
        ),
        ("native_shutdown",),
    ]
    assert state.worker.decode_runner is None
    assert state.worker._hook_handles == []
    assert state.worker._decode_graph_controller is None
    assert not state.worker._lane_states and not state.worker._streams


@pytest.mark.integration
def test_shutdown_before_model_attach_does_not_require_update_stream(
    worker_factory, monkeypatch
):
    state = worker_factory(attached=False)
    events = []
    assert not hasattr(state.decode, "update_stream")
    for name, stream in state.worker._streams.items():
        stream.synchronize = lambda name=name: events.append(("sync", name))
    monkeypatch.setattr(
        state.module.NPUWorker,
        "shutdown",
        lambda worker: events.append(("native_shutdown",)),
    )
    state.worker.shutdown()
    state.worker.shutdown()
    assert events == [
        ("sync", "prefill"),
        ("sync", "decode"),
        ("native_shutdown",),
    ]
    assert state.worker.decode_runner is None
    assert state.controller.snapshot()["closed"]
    assert state.controller.owner is state.controller.decode is None


@pytest.mark.integration
def test_shutdown_releases_graph_references_even_if_native_cleanup_fails(
    worker_factory, monkeypatch
):
    state = worker_factory()
    for stream in state.worker._streams.values():
        stream.synchronize = lambda: None
    state.decode.update_stream.synchronize = lambda: None
    state.decode.shutdown = lambda: pytest.fail("D native shutdown is forbidden")
    native_calls = []

    def native_shutdown(worker):
        native_calls.append(worker)
        assert state.decode.model is None
        raise ValueError("native cleanup failed")

    monkeypatch.setattr(state.module.NPUWorker, "shutdown", native_shutdown)
    with pytest.raises(ValueError, match="native cleanup"):
        state.worker.shutdown()
    state.worker.shutdown()
    assert len(native_calls) == 1
    assert state.worker.decode_runner is None
    assert state.worker._decode_graph_controller is None
    assert state.controller.snapshot()["closed"]
    assert not state.worker._lane_states and not state.worker._streams


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["graph_join", "communication_release"])
def test_shutdown_preserves_resources_and_can_retry_failed_safety_joins(
    worker_factory, monkeypatch, failure
):
    state = worker_factory(graph=failure == "graph_join")
    original_streams = state.worker._streams.copy()
    native_calls = []
    for stream in state.worker._streams.values():
        stream.synchronize = lambda: None
    state.decode.update_stream.synchronize = lambda: None

    def fail():
        raise RuntimeError("device join failed")

    if failure == "graph_join":
        state.worker._streams["decode"].synchronize = fail
    else:
        state.worker._release_overlap_communication = fail
    monkeypatch.setattr(
        state.module.NPUWorker, "shutdown", lambda worker: native_calls.append(worker)
    )
    with pytest.raises(RuntimeError, match="device join failed"):
        state.worker.shutdown()
    assert not native_calls
    assert state.worker.decode_runner is state.decode
    assert state.worker._streams == original_streams
    assert not state.controller.snapshot()["closed"]
    assert not state.worker._muxpd_shutdown_in_progress
    if failure == "graph_join":
        state.worker._streams["decode"].synchronize = lambda: None
    else:
        state.worker._release_overlap_communication = lambda: None
    state.worker.shutdown()
    state.worker.shutdown()
    assert native_calls == [state.worker]
    assert state.worker.decode_runner is None
    assert state.worker._muxpd_shutdown_complete
