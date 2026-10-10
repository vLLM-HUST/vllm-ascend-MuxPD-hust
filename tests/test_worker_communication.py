"""Isolated coordinator ownership without creating any device resources."""

import importlib
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


@pytest.fixture
def setup_worker(monkeypatch):
    if os.environ.get("VLLM_VERSION") != "0.25.1":
        pytest.skip("source runtime/env.sh for pinned HUST worker contracts")
    module = importlib.import_module("vllm_muxpd.worker")
    worker = module.MuxPDNPUWorker.__new__(module.MuxPDNPUWorker)
    worker.rank = 0
    worker.device = "fake-npu"
    worker.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True)
    )
    worker._overlap_communication = "isolated"
    worker._collective_gates = None
    worker._collective_router = None
    worker._prefill_groups = []
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def synchronize(self):
            events.append(("sync", self.name))

    worker._streams = {lane: Stream(lane) for lane in ("prefill", "decode")}

    class Communicator:
        def __init__(self, name, device_group):
            self.name = name
            self.device_group = device_group

        def all_reduce(self, value):
            events.append(("all_reduce", self.name, value))
            return value

        def all_gather(self, value, dim=-1):
            events.append(("all_gather", self.name, value, dim))
            return value

        def reduce_scatter(self, value, dim=-1):
            events.append(("reduce_scatter", self.name, value, dim))
            return value

    class Group:
        def __init__(self, name, device_group=None):
            self.unique_name = name
            self.ranks = [0, 1, 2, 3]
            self.group_ranks = [[0, 1, 2, 3]]
            self.world_size = 4
            self.local_rank = 0
            self.backend = "hccl"
            self.device_group = device_group or object()
            self.device_communicator = Communicator(name, self.device_group)

        def destroy(self):
            events.append(("destroy", self.unique_name))

    native = [Group("tp:0"), Group("ep:0")]
    monkeypatch.setattr(module, "get_tp_group", lambda: native[0])
    monkeypatch.setattr(module, "get_ep_group", lambda: native[1])
    monkeypatch.setattr(module.torch.distributed, "get_backend", lambda group: "hccl")
    creations = []

    def create(**kwargs):
        creations.append(kwargs)
        events.append(("create", kwargs["group_name"]))
        return Group(kwargs["group_name"])

    monkeypatch.setattr(module, "init_model_parallel_group", create)

    @contextmanager
    def stream_context(stream):
        events.append(("stream_enter", stream.name))
        yield
        events.append(("stream_exit", stream.name))

    monkeypatch.setattr(module.torch.npu, "stream", stream_context)
    monkeypatch.setattr(module.torch, "ones", lambda *args, **kwargs: "probe")
    backend = SimpleNamespace(active_lane=None, runtime=object())
    return SimpleNamespace(
        module=module,
        worker=worker,
        events=events,
        native=native,
        creations=creations,
        backend=backend,
        Group=Group,
        Communicator=Communicator,
    )


@pytest.mark.integration
def test_clones_native_membership_warms_groups_and_routes_only_prefill(setup_worker):
    state = setup_worker
    worker = state.worker
    worker._configure_overlap_communication(state.backend)
    assert state.creations == [
        {
            "group_ranks": [[0, 1, 2, 3]],
            "local_rank": 0,
            "backend": "hccl",
            "use_message_queue_broadcaster": False,
            "group_name": f"{name}:muxpd_prefill",
        }
        for name in ("tp", "ep")
    ]
    assert state.events[:7] == [
        ("create", "tp:muxpd_prefill"),
        ("create", "ep:muxpd_prefill"),
        ("stream_enter", "prefill"),
        ("all_reduce", "tp:muxpd_prefill", "probe"),
        ("all_reduce", "ep:muxpd_prefill", "probe"),
        ("sync", "prefill"),
        ("stream_exit", "prefill"),
    ]
    assert state.module.get_tp_group() is state.native[0]
    assert state.module.get_ep_group() is state.native[1]
    for lane in (None, "decode", "prefill"):
        state.backend.active_lane = lane
        for group in state.native:
            group.device_communicator.all_reduce("payload")
            assert state.events[-1][1] == (
                group.unique_name.split(":")[0] + ":muxpd_prefill"
                if lane == "prefill"
                else group.unique_name
            )
    assert worker._collective_gates is None
    state.backend.active_lane = None
    worker._release_overlap_communication()
    assert state.events[-4:] == [
        ("sync", "prefill"),
        ("sync", "decode"),
        ("destroy", "ep:muxpd_prefill"),
        ("destroy", "tp:muxpd_prefill"),
    ]
    assert worker._prefill_groups == []
    assert worker._collective_router is None
    assert all("all_reduce" not in vars(g.device_communicator) for g in state.native)


@pytest.mark.integration
def test_native_device_group_aliases_share_one_prefill_coordinator(setup_worker):
    state = setup_worker
    state.native[1] = state.Group("ep:0", state.native[0].device_group)
    state.worker._configure_overlap_communication(state.backend)
    assert len(state.creations) == 1
    state.backend.active_lane = "prefill"
    for group in state.native:
        group.device_communicator.all_gather("payload")
        assert state.events[-1][1] == "tp:muxpd_prefill"
    state.worker._release_overlap_communication()


@pytest.mark.integration
def test_single_rank_groups_need_no_router_or_clones(setup_worker):
    state = setup_worker
    for group in state.native:
        group.world_size = 1
    state.worker._configure_overlap_communication(state.backend)
    state.worker._release_overlap_communication()
    assert state.events == []
    assert state.worker._collective_router is None


@pytest.mark.integration
def test_shared_mode_installs_gates_without_siblings(setup_worker, monkeypatch):
    state = setup_worker
    worker = state.worker
    worker._overlap_communication = "shared"
    installed = []

    class Gates:
        def __init__(self, runtime):
            assert runtime is state.backend.runtime

        def install(self, communicators):
            installed.extend(communicators)
            return self

        def restore(self):
            state.events.append(("restore",))

    monkeypatch.setattr(state.module, "CollectiveGates", Gates)
    worker._configure_overlap_communication(state.backend)
    assert installed == [group.device_communicator for group in state.native]
    assert worker._collective_router is None
    assert not state.creations
    worker._release_overlap_communication()
    assert state.events == [("sync", "prefill"), ("sync", "decode"), ("restore",)]


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["creation", "warmup", "installation"])
def test_initialization_failure_cleans_only_owned_groups(
    setup_worker, monkeypatch, failure
):
    state = setup_worker
    create = state.module.init_model_parallel_group

    def create_or_fail(**kwargs):
        if failure == "creation" and kwargs["group_name"].startswith("ep:"):
            raise RuntimeError("creation failed")
        return create(**kwargs)

    monkeypatch.setattr(state.module, "init_model_parallel_group", create_or_fail)
    if failure == "warmup":
        monkeypatch.setattr(
            state.worker,
            "_warmup_prefill_groups",
            lambda: (_ for _ in ()).throw(RuntimeError("warmup failed")),
        )
    elif failure == "installation":
        # Router validation fails after three TP wrappers were installed. Its
        # rollback and worker cleanup must restore TP before owning groups die.
        state.native[1].device_communicator.reduce_scatter = None
    with pytest.raises((RuntimeError, ValueError)):
        state.worker._configure_overlap_communication(state.backend)
    destroyed = [event[1] for event in state.events if event[0] == "destroy"]
    assert destroyed == (
        ["tp:muxpd_prefill"]
        if failure == "creation"
        else ["ep:muxpd_prefill", "tp:muxpd_prefill"]
    )
    assert state.worker._prefill_groups == []
    assert state.worker._collective_router is None
    assert "all_reduce" not in vars(state.native[0].device_communicator)
    first_destroy = next(
        i for i, event in enumerate(state.events) if event[0] == "destroy"
    )
    assert ("sync", "decode") in state.events[:first_destroy]


@pytest.mark.integration
def test_cleanup_wait_failure_retains_ownership_for_safe_retry(
    setup_worker, monkeypatch
):
    state = setup_worker
    state.worker._configure_overlap_communication(state.backend)
    stream = state.worker._streams["prefill"]
    synchronize = stream.synchronize
    monkeypatch.setattr(
        stream, "synchronize", lambda: (_ for _ in ()).throw(RuntimeError("pending"))
    )
    with pytest.raises(RuntimeError, match="pending"):
        state.worker._release_overlap_communication()
    assert len(state.worker._prefill_groups) == 2
    assert state.worker._collective_router is not None
    assert not any(event[0] == "destroy" for event in state.events)
    monkeypatch.setattr(stream, "synchronize", synchronize)
    state.worker._release_overlap_communication()
    assert not state.worker._prefill_groups


@pytest.mark.integration
def test_cleanup_continues_after_destroy_error_and_retries_owned_group(
    setup_worker, monkeypatch
):
    state = setup_worker
    state.worker._configure_overlap_communication(state.backend)
    failing = state.worker._prefill_groups[1]
    destroy = failing.destroy
    monkeypatch.setattr(
        failing, "destroy", lambda: (_ for _ in ()).throw(RuntimeError("destroy"))
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        state.worker._release_overlap_communication()
    assert state.worker._prefill_groups == [failing]
    assert state.worker._collective_router is None
    assert ("destroy", "tp:muxpd_prefill") in state.events
    monkeypatch.setattr(failing, "destroy", destroy)
    state.worker._release_overlap_communication()
    assert not state.worker._prefill_groups


@pytest.mark.integration
def test_reused_native_process_group_is_rejected_before_warmup(
    setup_worker, monkeypatch
):
    state = setup_worker
    sibling = state.Group("tp:muxpd_prefill", state.native[0].device_group)
    monkeypatch.setattr(
        state.module, "init_model_parallel_group", lambda **kwargs: sibling
    )
    with pytest.raises(RuntimeError, match="aliases a native group"):
        state.worker._configure_overlap_communication(state.backend)
    assert ("destroy", "tp:muxpd_prefill") in state.events
    assert not any(event[0] == "all_reduce" for event in state.events)
    assert state.worker._prefill_groups == []


@pytest.mark.integration
def test_actual_ascend_coordinator_contract_and_registry_keep_prefill_independent(
    setup_worker, monkeypatch
):
    """Run the real patched constructor/options/registry with fake PG handles.

    Mock only distributed process creation, NPU objects and the communicator;
    retaining the actual constructor catches backend/layout and registry drift.
    """
    state = setup_worker
    parallel = importlib.import_module("vllm.distributed.parallel_state")
    patched = importlib.import_module("vllm_ascend.patch.worker.patch_distributed")
    registry_module = importlib.import_module(
        "vllm_ascend.patch.worker._hccl_pg_registry"
    )
    registry = registry_module.HcclPgRegistry()
    monkeypatch.setattr(patched, "_HCCL_PG_REGISTRY", registry)
    config_module = importlib.import_module("vllm.config")
    state.worker.vllm_config.parallel_config.data_parallel_size = 1
    monkeypatch.setattr(
        config_module, "get_current_vllm_config", lambda: state.worker.vllm_config
    )
    destroyed = []
    created = []

    def new_group(ranks, backend, pg_options=None):
        group = SimpleNamespace(ranks=list(ranks), backend=backend, options=pg_options)
        created.append(group)
        return group

    def get_backend(group):
        assert any(group is existing for existing in created)
        return group.backend

    monkeypatch.setattr(state.module.torch.distributed, "new_group", new_group)
    monkeypatch.setattr(state.module.torch.distributed, "get_backend", get_backend)
    monkeypatch.setattr(state.module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        state.module.torch.distributed, "destroy_process_group", destroyed.append
    )
    monkeypatch.setattr(state.module.torch.npu, "current_device", lambda: 0)
    # Preserve the real options helper, including its group_name in hccl_config,
    # and replace just the C++ host options constructor with a plain object.
    monkeypatch.setattr(
        state.module.ascend_utils,
        "torch_npu",
        SimpleNamespace(
            _C=SimpleNamespace(
                _distributed_c10d=SimpleNamespace(
                    ProcessGroupHCCL=SimpleNamespace(Options=SimpleNamespace)
                )
            )
        ),
    )

    def communicator(**kwargs):
        comm = state.Communicator(kwargs["unique_name"], kwargs["device_group"])
        comm.destroy = lambda: None
        return comm

    monkeypatch.setattr(patched, "NPUCommunicator", communicator)
    monkeypatch.setattr(
        state.module, "init_model_parallel_group", parallel.init_model_parallel_group
    )
    assert parallel.GroupCoordinator is patched.GroupCoordinatorPatch
    state.native[:] = [
        parallel.init_model_parallel_group(
            [[0, 1, 2, 3]], local_rank=0, backend="hccl", group_name=name
        )
        for name in ("tp", "ep")
    ]
    assert all(
        isinstance(group, patched.GroupCoordinatorPatch) for group in state.native
    )
    assert all(
        not hasattr(group, "torch_distributed_backend") for group in state.native
    )
    native_handles = [group.device_group for group in state.native]
    state.worker._configure_overlap_communication(state.backend)
    siblings = state.worker._prefill_groups.copy()
    assert [group.group_name for group in siblings] == [
        "tp:muxpd_prefill",
        "ep:muxpd_prefill",
    ]
    for native, sibling in zip(state.native, siblings):
        assert isinstance(sibling, patched.GroupCoordinatorPatch)
        assert sibling.group_ranks == native.group_ranks
        assert sibling.local_rank == native.local_rank
        assert sibling.backend == native.backend == "hccl"
        assert not any(sibling.device_group is handle for handle in native_handles)
        assert (
            sibling.device_group.options.hccl_config["group_name"] == sibling.group_name
        )
    state.backend.active_lane = "prefill"
    state.native[0].device_communicator.all_reduce("payload")
    assert state.events[-1][1] == siblings[0].unique_name
    assert len(registry._entries) == 4
    state.worker._release_overlap_communication()
    assert len(registry._entries) == 2
    assert not any(
        handle is destroyed_handle
        for handle in native_handles
        for destroyed_handle in destroyed
    )
    for native in state.native:
        native.destroy()
    assert not registry._entries
