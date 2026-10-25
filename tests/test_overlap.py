"""Dependency-graph tests; no NPU initialization or device allocation."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vllm_muxpd.lanes import LayerInterleaver
from vllm_muxpd.overlap import CollectiveGates, NPUOverlapBackend


class Stream:
    device = "fake-npu"

    def __init__(self, name):
        self.name = name


class Runtime:
    def __init__(self):
        self.stream = Stream("caller")
        self.records = []
        self.waits = []

    def current_stream(self):
        return self.stream

    def set_stream(self, stream):
        self.stream = stream

    def record_event(self, stream):
        event = (stream.name, len(self.records))
        self.records.append(event)
        return event

    def wait_event(self, stream, event):
        self.waits.append((stream.name, event))


class Communicator:
    def __init__(self, group):
        self.device_group = group

    def all_reduce(self, value):
        return ("reduce", value)

    def all_gather(self, value, dim=-1):
        return ("gather", value, dim)

    def reduce_scatter(self, value, dim=-1):
        return ("scatter", value, dim)


def test_gate_restore_keeps_failed_entry_and_tail_until_retry():
    runtime = Runtime()
    comm = Communicator(object())
    gate = CollectiveGates(runtime).install([comm])
    comm.all_reduce(1)
    wrapper = comm.all_gather
    comm.all_gather = lambda value: value
    with pytest.raises(RuntimeError, match="changed externally"):
        gate.restore()
    assert len(gate._installed) == 1 and gate._tails
    comm.all_gather = wrapper
    gate.restore()
    assert not gate._installed and not gate._tails


def backend():
    runtime = Runtime()
    original_manager = object()

    class Manager:
        def __init__(self, device, num_ubatches):
            self.device, self.num_ubatches = device, num_ubatches
            self.buffer = object()

    workspace = SimpleNamespace(_manager=original_manager, WorkspaceManager=Manager)
    context = SimpleNamespace(_forward_context="caller-context")
    ascend = SimpleNamespace(_CURRENT_STREAM=runtime.stream)
    instance = NPUOverlapBackend(
        runtime=runtime,
        workspace_module=workspace,
        forward_context_module=context,
        ascend_utils_module=ascend,
    )
    return instance, runtime, workspace, context, ascend


def test_collectives_share_gate_by_group_identity_and_restore_descriptors():
    runtime = Runtime()
    group = object()
    tp, ep, other = Communicator(group), Communicator(group), Communicator(object())
    gates = CollectiveGates(runtime).install([tp, ep, other, tp])
    assert tp.all_reduce(1) == ("reduce", 1)
    first = runtime.records[-1]
    runtime.stream = Stream("prefill")
    assert ep.all_gather(2, dim=0) == ("gather", 2, 0)
    assert runtime.waits == [("prefill", first)]
    assert other.reduce_scatter(3) == ("scatter", 3, -1)
    assert len(runtime.waits) == 1
    gates.restore()
    assert not any(name in vars(tp) for name in CollectiveGates.METHODS)
    tp.all_reduce(4)
    assert gates.calls == 3


def test_graph_overlap_retains_profiled_manager_and_rejects_lane_alias():
    instance, runtime, workspace, _, _ = backend()
    profiled = workspace._manager
    prefill = instance.lane_state(
        "prefill", Stream("prefill"), workspace_manager=profiled
    )
    with pytest.raises(ValueError, match="independent workspace"):
        instance.lane_state("decode", Stream("decode"), workspace_manager=profiled)
    decode = instance.lane_state("decode", Stream("decode"))
    caller = instance.capture()
    instance.begin_batch(None)
    instance.restore(prefill)
    assert workspace._manager is profiled
    instance.fence()
    prefill_tail = runtime.records[-1]
    instance.restore(decode)
    assert workspace._manager is decode.workspace_manager
    assert decode.workspace_manager is not profiled
    # Graph D retains its own workspace without waiting for P's kernel tail.
    assert all(event != prefill_tail for _, event in runtime.waits)
    instance.restore(caller)


def test_collective_failure_poisoning_and_install_rollback():
    runtime = Runtime()
    comm = Communicator(object())
    original = comm.all_reduce

    def broken(value):
        raise RuntimeError("HCCL failed")

    comm.all_reduce = broken
    gates = CollectiveGates(runtime).install([comm])
    with pytest.raises(RuntimeError, match="HCCL failed"):
        comm.all_reduce(1)
    with pytest.raises(RuntimeError, match="poisoned"):
        comm.all_gather(1)
    gates.restore()
    assert comm.all_reduce is broken
    comm.all_reduce = original
    with pytest.raises(ValueError, match="device_group"):
        CollectiveGates(runtime).install([comm, SimpleNamespace()])
    assert comm.all_reduce is original
    assert "all_gather" not in vars(comm)


def test_same_stream_fifo_skips_wait_but_switch_uses_latest_collective():
    class EqualStream(Stream):
        def __eq__(self, other):
            return isinstance(other, EqualStream) and self.name == other.name

    runtime = Runtime()
    group = object()
    tp, ep = Communicator(group), Communicator(group)
    gates = CollectiveGates(runtime).install([tp, ep])
    runtime.stream = EqualStream("decode")
    tp.all_reduce(1)
    first = runtime.records[-1]
    # current_stream() may return another Python wrapper for the same stream.
    runtime.stream = EqualStream("decode")
    ep.all_gather(2)
    second = runtime.records[-1]
    assert second != first
    assert runtime.waits == []
    assert len(runtime.records) == 2

    runtime.stream = EqualStream("prefill")
    tp.reduce_scatter(3)
    assert runtime.waits == [("prefill", second)]
    ep.all_reduce(4)
    prefill_tail = runtime.records[-1]
    assert len(runtime.records) == 4
    assert runtime.waits == [("prefill", second)]

    runtime.stream = EqualStream("decode")
    tp.all_gather(5)
    assert runtime.waits == [("prefill", second), ("decode", prefill_tail)]
    assert len(runtime.records) == gates.calls == 5
    gates.restore()


def test_nested_gate_install_and_cross_thread_calls_rejected():
    runtime = Runtime()
    comm = Communicator(object())
    gates = CollectiveGates(runtime).install([comm])
    with pytest.raises(RuntimeError, match="already installed"):
        CollectiveGates(runtime).install([comm])
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(comm.all_reduce, 1)
        with pytest.raises(RuntimeError, match="one host thread"):
            future.result()
    gates.restore()


def test_streams_share_initial_preparation_barrier_then_run_independently():
    instance, runtime, workspace, context, ascend = backend()
    assert instance.active_lane is None
    caller = instance.capture()
    decode = instance.lane_state("decode", Stream("decode"))
    prefill = instance.lane_state("prefill", Stream("prefill"))
    instance.begin_batch("decode")
    initial = runtime.records[-1]
    with pytest.raises(RuntimeError, match="preparation lane"):
        instance.restore(prefill)
    instance.restore(decode)
    assert instance.active_lane == "decode"
    assert runtime.waits == [("decode", initial)]
    context._forward_context = "decode-attention"
    instance.fence()  # Includes first runner's global new-block zeroing.
    prep_done = runtime.records[-1]
    saved_decode = instance.capture()
    instance.restore(caller)
    assert instance.active_lane is None
    assert workspace._manager is caller.workspace_manager
    instance.restore(prefill)
    assert runtime.waits[-1] == ("prefill", prep_done)
    assert workspace._manager is prefill.workspace_manager
    assert workspace._manager is not decode.workspace_manager
    context._forward_context = "prefill-attention"
    instance.fence()
    saved_prefill = instance.capture()
    instance.restore(caller)
    wait_count = len(runtime.waits)
    instance.restore(saved_decode)
    assert context._forward_context == "decode-attention"
    assert ascend._CURRENT_STREAM is decode.stream
    instance.fence()
    instance.restore(caller)
    instance.restore(saved_prefill)
    assert context._forward_context == "prefill-attention"
    instance.fence()
    instance.restore(caller)
    assert len(runtime.waits) == wait_count  # No per-layer cross-stream waits.
    instance.join(["decode"])
    assert runtime.waits[-1] == ("caller", instance._tails["decode"])


def test_overlap_backend_runs_real_greenlet_stacks_and_restores_workspace():
    instance, runtime, workspace, context, _ = backend()
    original = workspace._manager
    interleaver = LayerInterleaver(instance)
    states = {name: instance.lane_state(name, Stream(name)) for name in ("d", "p")}
    visits = []

    def forward(lane):
        context._forward_context = lane
        for layer in range(3):
            assert context._forward_context == lane
            assert workspace._manager is states[lane].workspace_manager
            visits.append((lane, layer))
            interleaver.checkpoint()
        return lane

    for lane in states:
        interleaver.add(lane, lambda lane=lane: forward(lane), states[lane])
    instance.begin_batch("d")
    assert interleaver.run(("d", "p")) == {"d": "d", "p": "p"}
    assert visits == [(name, layer) for layer in range(3) for name in ("d", "p")]
    assert workspace._manager is original
    assert len(runtime.waits) == 2
    instance.join()
    assert len(runtime.waits) == 4


def test_new_transaction_reestablishes_shared_zeroing_barrier():
    instance, runtime, _, _, _ = backend()
    caller = instance.capture()
    decode = instance.lane_state("decode", Stream("decode"))
    prefill = instance.lane_state("prefill", Stream("prefill"))
    for _ in range(2):
        instance.begin_batch("decode")
        instance.restore(decode)
        instance.fence()
        first_yield = runtime.records[-1]
        instance.restore(caller)
        instance.restore(prefill)
        assert runtime.waits[-1] == ("prefill", first_yield)
        instance.fence()
        instance.restore(caller)
    assert len(runtime.waits) == 4


def test_caller_preparation_allows_independent_first_launches():
    instance, runtime, _, _, _ = backend()
    caller = instance.capture()
    decode = instance.lane_state("decode", Stream("decode"))
    prefill = instance.lane_state("prefill", Stream("prefill"))
    # Global zeroing was enqueued on caller before begin_batch; neither lane
    # requires the other's first checkpoint to see that initialization.
    instance.begin_batch(None)
    preparation = runtime.records[-1]
    instance.restore(prefill)
    instance.restore(caller)
    instance.restore(decode)
    instance.restore(caller)
    assert runtime.waits == [("prefill", preparation), ("decode", preparation)]


def test_caller_preparation_restores_persistent_prefill_without_decode_tail_wait():
    instance, runtime, _, context, _ = backend()
    caller = instance.capture()
    decode = instance.lane_state("decode", Stream("decode"))
    prefill = instance.lane_state("prefill", Stream("prefill"))
    instance.begin_batch(None)
    instance.restore(prefill)
    context._forward_context = "persistent-prefill"
    instance.fence()
    saved_prefill = instance.capture()
    instance.restore(caller)
    instance.restore(decode)
    instance.fence()
    instance.restore(caller)
    instance.join(["decode"])
    previous_decode_tail = instance._tails["decode"]
    wait_count = len(runtime.waits)
    # The next caller event follows new zeroing and D retirement, but P's own
    # tail was never joined to caller. Stream FIFO retains P's earlier work.
    instance.begin_batch(None)
    preparation = runtime.records[-1]
    instance.restore(decode)
    instance.fence()
    instance.restore(caller)
    instance.restore(saved_prefill)
    assert context._forward_context == "persistent-prefill"
    assert runtime.waits[wait_count:] == [
        ("decode", preparation),
        ("prefill", preparation),
    ]
    assert instance._tails["decode"] != previous_decode_tail
    instance.restore(caller)


def test_restore_preserves_external_method_replacement():
    runtime = Runtime()
    comm = Communicator(object())
    gates = CollectiveGates(runtime).install([comm])

    def replacement(value):
        return value

    comm.all_reduce = replacement
    with pytest.raises(RuntimeError, match="changed externally"):
        gates.restore()
    assert comm.all_reduce is replacement
    assert "all_gather" not in vars(comm)
