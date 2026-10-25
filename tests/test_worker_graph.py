"""Exercise the worker's graph integration with real controller CPU fakes."""

import importlib
import os
from types import SimpleNamespace as NS

import pytest
from test_decode_graph import Stream, controller

from vllm_muxpd.decode_graph import GraphFencedExecutionBackend, GraphLaneState
from vllm_muxpd.overlap import NPUOverlapBackend


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
@pytest.mark.parametrize("current", [False, True])
def test_shared_cache_retains_native_init_arguments_without_allocating(
    worker_factory, monkeypatch, current
):
    state = worker_factory()
    module = state.module
    native = module.NPUModelRunner
    calls = []
    config = NS(kv_cache_groups=[NS(layer_names=["layer"])])
    owner = NS(
        kv_cache_config=config,
        kernel_block_sizes=[16],
        kv_caches=[object()],
        shared_kv_cache_layers={},
        runner_only_attn_layers=set(),
    )
    runner = module._SharedCacheRunner.__new__(module._SharedCacheRunner)
    runner._owner = owner
    runner.compilation_config = NS(
        static_forward_context={"layer": NS(kv_cache=owner.kv_caches[0])}
    )
    context = object()

    if current:

        def initialize(self, kv_cache_config, kv_cache_allocation_context=None):
            self.kv_cache_config = kv_cache_config
            calls.append(kv_cache_allocation_context)
            self.initialize_kv_cache_tensors(
                kv_cache_config,
                kernel_block_sizes=[16],
                kv_cache_allocation_context=kv_cache_allocation_context,
            )

        def tensors(
            self,
            kv_cache_config,
            kernel_block_sizes=None,
            kv_cache_allocation_context=None,
        ):
            pytest.fail("D must not allocate or bind shared backing tensors")
    else:

        def initialize(self, kv_cache_config):
            self.kv_cache_config = kv_cache_config
            calls.append(None)
            self.initialize_kv_cache_tensors(kv_cache_config)

        def tensors(self, kv_cache_config):
            pytest.fail("D must not allocate or bind shared backing tensors")

    monkeypatch.setattr(native, "initialize_kv_cache", initialize)
    monkeypatch.setattr(native, "initialize_kv_cache_tensors", tensors)
    runner.initialize_kv_cache(config, context if current else None)
    assert calls == [context if current else None]
    assert runner.kv_caches is owner.kv_caches
    if current:
        runner.kv_caches = None
        with pytest.raises(RuntimeError, match="block layouts"):
            runner.initialize_kv_cache_tensors(config, kernel_block_sizes=[32])
        assert runner.kv_caches is None


@pytest.fixture
def overlap_graph_factory(worker_factory, monkeypatch):
    def create(*, communication_fails=False):
        state = worker_factory(graph=False, attached=False)
        state.controller.close()
        worker, runtime = state.worker, state.runtime
        workspace = importlib.import_module("vllm.v1.worker.workspace")
        profiled = object()
        monkeypatch.setattr(workspace, "_manager", profiled)
        monkeypatch.setattr(
            workspace,
            "WorkspaceManager",
            lambda device, num_ubatches: NS(device=device, num_ubatches=num_ubatches),
        )
        runtime.workspace = workspace
        worker.device = "fake-npu"
        worker._decode_graph_enabled = True
        worker._stream_mode = "overlap"
        worker._overlap_communication = "isolated"
        policy = state.owner.vllm_config.scheduler_config.batch_admission_policy_config
        policy.update(stream_mode="overlap", overlap_communication="isolated")
        original_controller = state.module.DecodeGraphController
        monkeypatch.setattr(
            state.module,
            "DecodeGraphController",
            lambda *args: original_controller(*args, runtime=runtime),
        )
        monkeypatch.setattr(
            state.module,
            "NPUOverlapBackend",
            lambda: NPUOverlapBackend(
                runtime=runtime.npu,
                forward_context_module=runtime.forward,
                workspace_module=workspace,
                ascend_utils_module=runtime.ascend,
            ),
        )
        configured = []

        def communication(backend):
            configured.append(backend)
            assert worker._decode_graph_controller is not None
            if communication_fails:
                raise RuntimeError("communication setup failed")

        worker._configure_overlap_communication = communication
        state.profiled_workspace = profiled
        state.configured_communication = configured
        return state

    return create


@pytest.mark.integration
def test_overlap_graph_composes_controller_with_profiled_owner_workspace(
    overlap_graph_factory,
):
    state = overlap_graph_factory()
    worker = state.worker
    worker._configure_execution_backend()
    backend, ctrl = worker._execution_backend, worker._decode_graph_controller
    assert isinstance(backend, NPUOverlapBackend)
    assert state.configured_communication == [backend]
    assert worker._lane_states["prefill"].workspace_manager is state.profiled_workspace
    assert worker._lane_states["decode"].workspace_manager is ctrl.workspace
    assert ctrl.workspace is not state.profiled_workspace
    assert ctrl.stream is worker._streams["decode"]
    assert state.owner._dummy_run(4)[2:] == ("NONE", True, False)
    assert not ctrl.compile_full_prefill


@pytest.mark.integration
def test_overlap_communication_failure_restores_graph_guards(overlap_graph_factory):
    state = overlap_graph_factory(communication_fails=True)
    with pytest.raises(RuntimeError, match="communication setup failed"):
        state.worker._configure_execution_backend()
    assert state.worker._decode_graph_controller is None
    assert "_muxpd_decode_graph_controller" not in vars(state.owner)
    assert state.owner.use_aclgraph


@pytest.mark.integration
@pytest.mark.parametrize("decode_fails", [False, True])
def test_overlap_graph_kv_initialization_joins_startup_and_restores_caller(
    overlap_graph_factory,
    monkeypatch,
    decode_fails,
):
    state = overlap_graph_factory()
    worker = state.worker
    worker._configure_execution_backend()
    backend, runtime = worker._execution_backend, state.runtime
    caller = backend.capture()
    events = []

    def owner_init(worker, cfg):
        assert backend.capture() == caller
        events.append("owner")

    def decode_init(cfg):
        assert backend.capture() == worker._lane_states["decode"]
        events.append("decode")
        if decode_fails:
            raise RuntimeError("decode metadata failed")

    monkeypatch.setattr(state.module.NPUWorker, "initialize_from_config", owner_init)
    state.decode.initialize_kv_cache = decode_init
    if decode_fails:
        with pytest.raises(RuntimeError, match="decode metadata failed"):
            worker.initialize_from_config(object())
    else:
        worker.initialize_from_config(object())
    assert events == ["owner", "decode"]
    assert backend.capture() == caller
    assert runtime.npu.waits == [
        (worker._streams["decode"], runtime.npu.records[0]),
        (caller.stream, runtime.npu.records[-1]),
    ]


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
def test_shutdown_releases_graphs_but_retries_failed_native_cleanup(
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
        if len(native_calls) == 1:
            raise ValueError("native cleanup failed")

    monkeypatch.setattr(state.module.NPUWorker, "shutdown", native_shutdown)
    with pytest.raises(ValueError, match="native cleanup"):
        state.worker.shutdown()
    assert state.worker.decode_runner is state.decode
    assert state.worker._streams and not getattr(
        state.worker, "_muxpd_shutdown_complete", False
    )
    state.worker.shutdown()
    state.worker.shutdown()
    assert len(native_calls) == 2
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
