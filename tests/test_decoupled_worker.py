"""Persistent forward and cancellation protocol without allocating model weights."""

import importlib
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


@pytest.fixture(params=["fenced", "graph", "overlap", "isolated", "graph_isolated"])
def simulation(request):
    if os.environ.get("VLLM_VERSION") != "0.25.1":
        pytest.skip("source runtime/env.sh for pinned HUST worker contracts")
    module = importlib.import_module("vllm_muxpd.decoupled_worker")
    from vllm.v1.outputs import ModelRunnerOutput

    from vllm_muxpd.overlap import NPUOverlapBackend
    from vllm_muxpd.worker import _NPUState

    worker = module.DecoupledMuxPDNPUWorker.__new__(module.DecoupledMuxPDNPUWorker)
    worker._prefill_task = None
    worker._prefill_cohort_id = None
    worker._prefill_order = []
    worker._prefill_cancelled = set()
    worker._awaiting_sample = False
    worker._decode_only_fast_path = False
    worker._decode_graph_enabled = request.param in ("graph", "graph_isolated")
    worker._iteration_cohort_id = None
    worker._iteration_prefill_layers_completed = 0
    worker._iteration_prefill_layers_total = 0
    worker._cancelled_ack = set()
    worker._interleaver = None
    worker._pending_lanes = []
    worker._pending_order = []
    worker._layers_per_slice = 1
    worker._decoder_layer_count = 3
    worker._stream_mode = (
        "fenced" if request.param in ("fenced", "graph") else "overlap"
    )
    worker._overlap_communication = (
        "isolated" if request.param in ("isolated", "graph_isolated") else "shared"
    )
    worker._collective_gates = None
    worker._collective_router = None
    worker._prefill_groups = []
    worker._hook_handles = []
    worker.muxpd_stats = {"prefill_batches": 0, "decode_batches": 0, "slices": 0}
    events = []

    class Stream:
        device = "fake-npu"

        def __init__(self, name):
            self.name = name

        def synchronize(self):
            events.append(("sync",))

    worker._streams = {phase: Stream(phase) for phase in ("prefill", "decode")}
    worker.profiler = SimpleNamespace(step=lambda: events.append(("profiler",)))

    class Backend:
        state = _NPUState(None, Stream("caller"))

        def capture(self):
            return self.state

        def restore(self, state):
            self.state = state

        def fence(self):
            events.append(("fence", self.state.stream))

    class Runtime:
        stream = Stream("caller")

        def current_stream(self):
            return self.stream

        def set_stream(self, stream):
            self.stream = stream

        def record_event(self, stream):
            event = (stream.name, len(events))
            events.append(("event", event))
            return event

        def wait_event(self, stream, event):
            events.append(("wait", stream.name, event))

    if request.param == "graph":
        from vllm_muxpd.decode_graph import GraphFencedExecutionBackend, GraphLaneState

        runtime = Runtime()
        workspace = SimpleNamespace(_manager=object())
        worker._execution_backend = GraphFencedExecutionBackend(
            runtime=runtime,
            forward_context_module=SimpleNamespace(_forward_context=None),
            workspace_module=workspace,
            ascend_utils_module=SimpleNamespace(_CURRENT_STREAM=runtime.stream),
        )
        worker._lane_states = {
            phase: GraphLaneState(None, stream, object())
            for phase, stream in worker._streams.items()
        }
    elif worker._stream_mode == "overlap":
        runtime = Runtime()
        backend = NPUOverlapBackend(
            runtime=runtime,
            forward_context_module=SimpleNamespace(_forward_context=None),
            ascend_utils_module=SimpleNamespace(_CURRENT_STREAM=runtime.stream),
            workspace_module=SimpleNamespace(
                _manager=None,
                WorkspaceManager=lambda device, num_ubatches: object(),
            ),
        )
        worker._execution_backend = backend
        worker._lane_states = {
            phase: backend.lane_state(phase, stream)
            for phase, stream in worker._streams.items()
        }
    else:
        worker._execution_backend = Backend()
        worker._lane_states = {
            phase: _NPUState(None, stream) for phase, stream in worker._streams.items()
        }
    registry = {}

    class Runner:
        def __init__(self, phase, layers):
            self.phase = phase
            self.layers = layers
            self.requests = registry
            self.num_prompt_logprobs = {}
            self.mamba_state_idx = {}
            self._hybrid_accepted_by_req = {}
            self.input_batch = SimpleNamespace(remove_request=lambda req_id: None)
            self.execute_model_state = None
            self.sample_count = 0
            self.closed = 0

        def execute_model(self, output):
            assert self.execute_model_state is None
            if worker._stream_mode == "overlap":
                assert output.new_block_ids_to_zero is None
            ids = tuple(output.num_scheduled_tokens)
            try:
                for layer in range(self.layers):
                    events.append((self.phase, ids, layer))
                    if worker._interleaver is not None and not (
                        self.phase == "decode" and request.param == "graph_isolated"
                    ):
                        # A captured decode graph returns as one forward and
                        # does not execute Python decoder-layer hooks on replay.
                        worker._interleaver.checkpoint(final=layer == self.layers - 1)
                self.execute_model_state = output
            finally:
                self.closed += 1
            return None

        def sample_tokens(self, grammar):
            assert grammar is None
            assert self.execute_model_state is not None
            self.sample_count += 1
            ids = list(self.execute_model_state.num_scheduled_tokens)
            self.execute_model_state = None
            return ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={req_id: i for i, req_id in enumerate(ids)},
                sampled_token_ids=[[7] for _ in ids],
                pooler_output=[],
            )

        def _zero_block_ids(self, block_ids):
            events.append(("zero", tuple(block_ids)))

    worker.model_runner = Runner("prefill", 3)
    worker.decode_runner = Runner("decode", 1)
    return worker, events


def schedule(*, new_p=(), pending=(), decode=(), cancel=(), finish=False, pause=False):
    from vllm.v1.core.sched.output import SchedulerOutput

    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {
        **{req_id: 1 for req_id in decode},
        **{req_id: 32 for req_id in new_p},
    }
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    output.muxpd_execution = "decoupled"
    output.muxpd_prefill_request_ids = frozenset(new_p)
    output.muxpd_pending_prefill_ids = frozenset(pending)
    output.muxpd_cancel_prefill_request_ids = frozenset(cancel)
    output.muxpd_prefill_cohort_id = 1 if pending else None
    output.muxpd_layers_per_slice = 1
    output.muxpd_finish_prefill = finish
    output.muxpd_pause_prefill = pause
    return output


@pytest.mark.integration
def test_decode_only_reuses_native_rows_without_greenlet_and_preserves_lifecycle(
    simulation, monkeypatch
):
    from vllm.v1.core.sched.output import CachedRequestData

    worker, events = simulation
    module = importlib.import_module("vllm_muxpd.decoupled_worker")

    def forbidden(*args, **kwargs):
        pytest.fail("D-only execution must not create a task or slice native rows")

    monkeypatch.setattr(module, "LayerInterleaver", forbidden)
    monkeypatch.setattr(module, "_select_requests", forbidden)
    removed = {"prefill": [], "decode": []}
    for runner in (worker.model_runner, worker.decode_runner):
        runner.input_batch.remove_request = removed[runner.phase].append
        runner.requests["finished"] = object()
        runner._hybrid_accepted_by_req.update(finished=1, preempted=1, d=1)
    output = schedule(decode=("d",))
    output.finished_req_ids = {"finished"}
    output.preempted_req_ids = {"preempted"}
    output.new_block_ids_to_zero = [42]
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["d"], resumed_req_ids={"d"}, new_token_ids=[],
        all_token_ids={"d": [3, 4]}, new_block_ids=[([42],)],
        num_computed_tokens=[9], num_output_tokens=[2],
    )
    caller = worker._execution_backend.capture()
    worker.execute_model(output)
    native_output = worker.decode_runner.execute_model_state
    assert native_output.scheduled_cached_reqs is output.scheduled_cached_reqs
    assert native_output.num_scheduled_tokens is output.num_scheduled_tokens
    assert not native_output.finished_req_ids and not native_output.preempted_req_ids
    assert output.finished_req_ids == {"finished"}
    assert output.preempted_req_ids == {"preempted"}
    assert all(set(ids) == {"finished", "preempted", "d"} for ids in removed.values())
    assert "finished" not in worker.model_runner.requests
    assert not worker.decode_runner._hybrid_accepted_by_req
    assert worker._execution_backend.capture() == caller
    assert worker._interleaver is None and worker.model_runner.closed == 0
    native_samples = []
    sample = worker.decode_runner.sample_tokens

    def capture_sample(grammar):
        result = sample(grammar)
        native_samples.append(result)
        return result

    monkeypatch.setattr(worker.decode_runner, "sample_tokens", capture_sample)
    result = worker.sample_tokens(None)
    assert result is native_samples[0]
    assert result.req_ids == ["d"] and result.sampled_token_ids == [[7]]
    assert result.muxpd_prefill_layers_completed == 0
    assert result.muxpd_prefill_layers_total == 0
    assert result.muxpd_prefill_cohort_id is None
    assert not result.muxpd_completed_prefill_ids
    assert not result.muxpd_cancelled_prefill_ids
    assert worker._execution_backend.capture() == caller
    assert worker.decode_runner.sample_count == 1
    if worker._stream_mode == "overlap":
        assert events.count(("zero", (42,))) == 1
        assert events[-1][0] == "wait" and events[-1][-1][0] == "decode"


@pytest.mark.integration
@pytest.mark.parametrize("cancel", [False, True])
def test_decode_only_path_yields_to_persistent_prefill_then_returns(simulation, cancel):
    worker, _ = simulation
    worker.execute_model(schedule(decode=("d",)))
    assert worker._decode_only_fast_path
    worker.sample_tokens(None)
    worker.execute_model(schedule(new_p=("p",), pending=("p",), decode=("d",)))
    assert not worker._decode_only_fast_path
    first = worker.sample_tokens(None)
    assert first.req_ids == ["d"] and first.muxpd_prefill_layers_completed == 1
    task = worker._prefill_task
    worker.execute_model(
        schedule(pending=("p",), decode=("d",), cancel=("p",) if cancel else (),
                 finish=not cancel)
    )
    assert not worker._decode_only_fast_path
    result = worker.sample_tokens(None)
    assert result.muxpd_cancelled_prefill_ids == ({"p"} if cancel else set())
    assert result.muxpd_completed_prefill_ids == (set() if cancel else {"p"})
    assert task.tasks["prefill"].done and worker._prefill_task is None
    worker.execute_model(schedule(decode=("d",)))
    assert worker._decode_only_fast_path
    assert worker.sample_tokens(None).req_ids == ["d"]
    assert worker.decode_runner.sample_count == 4


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["execute", "sample"])
def test_decode_only_failures_restore_caller_state(simulation, monkeypatch, failure):
    worker, _ = simulation
    caller = worker._execution_backend.capture()

    def fail(*args):
        assert worker._interleaver is None
        assert worker._execution_backend.capture().stream is worker._streams["decode"]
        raise ValueError("native failure")

    if failure == "execute":
        monkeypatch.setattr(worker.decode_runner, "execute_model", fail)
        with pytest.raises(ValueError, match="native failure"):
            worker.execute_model(schedule(decode=("d",)))
        assert not worker._awaiting_sample and not worker._pending_lanes
    else:
        worker.execute_model(schedule(decode=("d",)))
        monkeypatch.setattr(worker.decode_runner, "sample_tokens", fail)
        with pytest.raises(ValueError, match="native failure"):
            worker.sample_tokens(None)
        assert worker._awaiting_sample and worker._pending_lanes == ["decode"]
        with pytest.raises(RuntimeError, match="sampler ACK"):
            worker.execute_model(schedule(decode=("d",)))
    assert worker._interleaver is None
    assert worker._execution_backend.capture() == caller


@pytest.mark.integration
def test_decode_only_native_compaction_order_falls_back_to_merge(
    simulation, monkeypatch
):
    from vllm.v1.outputs import ModelRunnerOutput

    worker, _ = simulation
    worker.execute_model(schedule(decode=("d1", "d2")))
    native = ModelRunnerOutput(
        req_ids=["d2", "d1"], req_id_to_index={"d2": 0, "d1": 1},
        sampled_token_ids=[[22], [11]],
    )
    monkeypatch.setattr(worker.decode_runner, "sample_tokens", lambda grammar: native)
    result = worker.sample_tokens(None)
    assert result is not native
    assert result.req_ids == ["d1", "d2"]
    assert result.req_id_to_index == {"d1": 0, "d2": 1}
    assert result.sampled_token_ids == [[11], [22]]


@pytest.mark.integration
def test_decode_samples_each_iteration_while_prefill_is_suspended(simulation):
    worker, events = simulation
    for index in range(3):
        worker.execute_model(
            schedule(new_p=("p",) if index == 0 else (), pending=("p",), decode=("d",))
        )
        result = worker.sample_tokens(None)
        assert result.req_ids == (["d"] if index < 2 else ["d", "p"])
        assert result.muxpd_completed_prefill_ids == (
            frozenset() if index < 2 else frozenset({"p"})
        )
        assert worker._execution_backend.capture().stream.name == "caller"
        assert result.muxpd_prefill_layers_completed == index + 1
        assert result.muxpd_prefill_layers_total == 3
    assert worker.model_runner.sample_count == 1
    assert worker.decode_runner.sample_count == 3
    assert events.count(("profiler",)) == 3
    assert worker._prefill_task is None
    expected = []
    for layer in range(3):
        pair = [("decode", 0), ("prefill", layer)]
        expected.extend(
            reversed(pair) if worker._overlap_communication == "isolated" else pair
        )
    assert [
        (event[0], event[-1]) for event in events if event[0] in ("decode", "prefill")
    ] == expected


@pytest.mark.integration
def test_no_decode_finishes_prefill_without_extra_engine_iterations(simulation):
    worker, _ = simulation
    worker.execute_model(schedule(new_p=("p",), pending=("p",), finish=True))
    output = worker.sample_tokens(None)
    assert output.req_ids == ["p"]
    assert output.muxpd_completed_prefill_ids == {"p"}
    assert output.muxpd_prefill_layers_completed == 3
    assert output.muxpd_prefill_layers_total == 3
    assert worker.model_runner.sample_count == 1


@pytest.mark.integration
def test_no_selected_decode_does_not_override_explicit_partial_prefill(simulation):
    worker, _ = simulation
    output = schedule(new_p=("p",), pending=("p",), finish=False)
    worker.execute_model(output)
    first = worker.sample_tokens(None)
    assert not first.req_ids
    assert not first.muxpd_completed_prefill_ids
    assert first.muxpd_prefill_layers_completed == 1
    assert worker.model_runner.sample_count == 0
    assert worker._prefill_task is not None
    worker.execute_model(schedule(pending=("p",), decode=("d",)))
    second = worker.sample_tokens(None)
    assert second.req_ids == ["d"]
    assert second.muxpd_prefill_layers_completed == 2
    worker.execute_model(schedule(pending=("p",), finish=True))
    assert worker.sample_tokens(None).muxpd_completed_prefill_ids == {"p"}


def enable_fake_slice_timing(worker, monkeypatch):
    module = importlib.import_module("vllm_muxpd.decoupled_worker")
    events = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            events.append(self)
            self.queries = 0

        def record(self, stream):
            assert stream is worker._streams["prefill"]
            assert worker._execution_backend.capture().stream is stream

        def query(self):
            # These fixtures use synchronous samplers and deliberately expose
            # no completion signal in an unfinished P-only iteration.
            assert worker.decode_runner.sample_count or worker.model_runner.sample_count
            self.queries += 1
            return True

        def elapsed_time(self, other):
            assert other.queries == 1
            return 5.0

        def synchronize(self):
            pytest.fail("slice timing must reuse the existing sampler barrier")

    monkeypatch.setattr(module.torch.npu, "Event", Event)
    worker._prefill_slice_timing_enabled = True
    return events


@pytest.mark.integration
@pytest.mark.parametrize("simulation", ["fenced", "graph"], indirect=True)
def test_slice_timing_only_reads_device_span_after_actual_sampler(
    simulation, monkeypatch
):
    worker, _ = simulation
    timers = enable_fake_slice_timing(worker, monkeypatch)
    worker.execute_model(schedule(new_p=("p",), pending=("p",)))
    first = worker.sample_tokens(None).muxpd_prefill_slice_timing
    assert first["first_slice"] and first["layer_delta"] == 1
    assert first["prefill_tokens"] == 32
    assert first["mode"] == "raw_layers"
    assert first["host_submit_ms"] >= 0
    assert first["device_elapsed_ms"] is None
    assert not first["device_complete"]
    assert sum(timer.queries for timer in timers) == 0

    worker.execute_model(schedule(pending=("p",), decode=("d",)))
    second = worker.sample_tokens(None).muxpd_prefill_slice_timing
    assert not second["first_slice"] and second["layer_delta"] == 1
    assert second["device_complete"] and second["device_elapsed_ms"] == 5.0
    assert second["prefill_tokens"] == 32
    assert sum(timer.queries for timer in timers) == 1

    worker.execute_model(schedule(pending=("p",), cancel=("p",), pause=True))
    aborted = worker.sample_tokens(None)
    assert aborted.muxpd_prefill_slice_timing is None
    assert aborted.muxpd_cancelled_prefill_ids == {"p"}
    assert len(timers) == 4


@pytest.mark.integration
def test_disabled_slice_timing_allocates_no_events(simulation, monkeypatch):
    worker, _ = simulation
    module = importlib.import_module("vllm_muxpd.decoupled_worker")

    def forbidden(*args, **kwargs):
        pytest.fail("disabled slice timing must not allocate timing events")

    monkeypatch.setattr(module.torch.npu, "Event", forbidden)
    worker.execute_model(schedule(new_p=("p",), pending=("p",), decode=("d",)))
    assert worker.sample_tokens(None).muxpd_prefill_slice_timing is None


@pytest.mark.integration
@pytest.mark.parametrize("simulation", ["fenced", "graph"], indirect=True)
def test_partial_cancel_keeps_forward_progress_without_training_timer(
    simulation, monkeypatch
):
    worker, _ = simulation
    timers = enable_fake_slice_timing(worker, monkeypatch)
    worker.execute_model(schedule(new_p=("p", "q"), pending=("p", "q")))
    worker.sample_tokens(None)
    assert len(timers) == 2
    worker.execute_model(
        schedule(pending=("p", "q"), decode=("d",), cancel=("p",))
    )
    partial = worker.sample_tokens(None)
    assert partial.muxpd_prefill_layers_completed == 2
    assert partial.muxpd_prefill_slice_timing is None
    assert not partial.muxpd_cancelled_prefill_ids
    worker.execute_model(schedule(pending=("p", "q"), finish=True))
    done = worker.sample_tokens(None)
    assert done.muxpd_completed_prefill_ids == {"q"}
    assert done.muxpd_cancelled_prefill_ids == {"p"}
    assert done.muxpd_prefill_slice_timing is None
    assert len(timers) == 2


@pytest.mark.integration
def test_cancel_entire_cohort_unwinds_and_synchronizes_before_ack(simulation):
    worker, events = simulation
    worker.execute_model(schedule(new_p=("p",), pending=("p",)))
    assert not worker.sample_tokens(None).req_ids
    worker.execute_model(schedule(pending=("p",), cancel=("p",), pause=True))
    assert ("sync",) in events
    output = worker.sample_tokens(None)
    assert output.req_ids == []
    assert output.muxpd_cancelled_prefill_ids == {"p"}
    assert output.muxpd_completed_prefill_ids == set()
    assert output.muxpd_prefill_layers_completed == 1
    assert output.muxpd_prefill_layers_total == 3
    assert worker.model_runner.closed == 1
    assert worker.model_runner.sample_count == 0
    assert worker._prefill_task is None


@pytest.mark.integration
def test_partial_cancel_waits_for_safe_completion_and_filters_sample(simulation):
    worker, _ = simulation
    worker.execute_model(schedule(new_p=("p1", "p2"), pending=("p1", "p2")))
    worker.sample_tokens(None)
    worker.execute_model(schedule(pending=("p1", "p2"), cancel=("p1",), pause=True))
    paused = worker.sample_tokens(None)
    assert not paused.muxpd_cancelled_prefill_ids
    assert paused.muxpd_prefill_layers_completed == 1
    assert paused.muxpd_prefill_layers_total == 3
    assert worker.model_runner.closed == 0
    worker.execute_model(schedule(pending=("p1", "p2"), finish=True))
    output = worker.sample_tokens(None)
    assert output.req_ids == ["p2"]
    assert output.muxpd_completed_prefill_ids == {"p2"}
    assert output.muxpd_cancelled_prefill_ids == {"p1"}
    assert output.muxpd_prefill_layers_completed == 3
    assert output.muxpd_prefill_layers_total == 3


@pytest.mark.integration
def test_invalid_pin_protocol_is_rejected_without_losing_prefill(simulation):
    worker, _ = simulation
    worker.execute_model(schedule(new_p=("p",), pending=("p",)))
    worker.sample_tokens(None)
    with pytest.raises(RuntimeError, match="pinned prefills"):
        worker.execute_model(schedule())
    with pytest.raises(RuntimeError, match="sampler ACK"):
        worker.execute_model(schedule(pending=("p",), pause=True))
        worker.execute_model(schedule(pending=("p",), pause=True))
    worker.sample_tokens(None)
    worker.execute_model(schedule(pending=("p",), cancel=("p",)))
    worker.sample_tokens(None)


@pytest.mark.integration
def test_overlap_zeroes_once_before_lanes_and_joins_only_sampled_decode(simulation):
    worker, events = simulation
    if worker._stream_mode != "overlap":
        pytest.skip("overlap-specific event dependencies")
    output = schedule(new_p=("p",), pending=("p",), decode=("d",))
    output.new_block_ids_to_zero = [11, 12]
    worker.execute_model(output)
    result = worker.sample_tokens(None)
    assert result.req_ids == ["d"]
    assert events.count(("zero", (11, 12))) == 1
    zero_index = events.index(("zero", (11, 12)))
    caller_event = events[zero_index + 1][1]
    assert caller_event[0] == "caller"
    assert ("wait", "decode", caller_event) in events
    assert ("wait", "prefill", caller_event) in events
    caller_waits = [event for event in events if event[:2] == ("wait", "caller")]
    assert len(caller_waits) == 1
    assert caller_waits[0][2][0] == "decode"
    assert not worker._prefill_task.tasks["prefill"].done
    assert output.new_block_ids_to_zero == [11, 12]
    worker.execute_model(schedule(pending=("p",), cancel=("p",)))
    worker.sample_tokens(None)


@pytest.mark.integration
def test_prefill_layer_quota_and_final_layer_retirement_with_complete_decode(
    simulation,
):
    worker, events = simulation
    worker.decode_runner.layers = 4
    worker.model_runner.layers = 5
    worker._decoder_layer_count = 5
    for iteration in range(3):
        start = len(events)
        output = schedule(
            new_p=("p",) if iteration == 0 else (),
            pending=("p",),
            decode=("d",),
        )
        output.muxpd_layers_per_slice = 2
        worker.execute_model(output)
        observed = [
            (event[0], event[-1])
            for event in events[start:]
            if event[0] in ("decode", "prefill")
        ]
        p_start = 2 * iteration
        p_layers = list(range(p_start, min(p_start + 2, 5)))
        assert [layer for lane, layer in observed if lane == "decode"] == list(range(4))
        assert [layer for lane, layer in observed if lane == "prefill"] == p_layers
        if worker._overlap_communication == "isolated":
            assert observed == [("prefill", i) for i in p_layers] + [
                ("decode", i) for i in range(4)
            ]
        elif worker._stream_mode == "overlap":
            prefix = [
                pair
                for i, p_layer in enumerate(p_layers)
                for pair in (("decode", i), ("prefill", p_layer))
            ]
            assert observed == prefix + [("decode", i) for i in range(len(p_layers), 4)]
        else:
            assert observed == [("decode", i) for i in range(4)] + [
                ("prefill", i) for i in p_layers
            ]
        result = worker.sample_tokens(None)
        assert result.req_ids == (["d"] if iteration < 2 else ["d", "p"])
        assert result.muxpd_completed_prefill_ids == (set() if iteration < 2 else {"p"})
        assert result.muxpd_prefill_layers_completed == min(p_start + 2, 5)
        assert result.muxpd_prefill_layers_total == 5
    assert worker.model_runner.sample_count == 1
    assert worker.decode_runner.sample_count == 3


@pytest.mark.integration
def test_short_decode_does_not_limit_prefill_slice_quota(simulation):
    worker, events = simulation
    worker.model_runner.layers = 7
    worker._decoder_layer_count = 7
    output = schedule(new_p=("p",), pending=("p",), decode=("d",))
    output.muxpd_layers_per_slice = 4
    worker.execute_model(output)
    result = worker.sample_tokens(None)
    assert result.req_ids == ["d"]
    assert worker._prefill_task.tasks["prefill"].completed_layers == 4
    assert [event[-1] for event in events if event[0] == "prefill"] == list(range(4))
    worker.execute_model(schedule(pending=("p",), finish=True))
    assert worker.sample_tokens(None).req_ids == ["p"]


@pytest.mark.integration
def test_progress_receipt_resets_without_cohort_and_changes_with_slice_budget(
    simulation,
):
    worker, _ = simulation
    worker.model_runner.layers = 8
    worker._decoder_layer_count = 8
    completed = 0
    for index, quota in enumerate((1, 4, 3)):
        output = schedule(
            new_p=("p",) if index == 0 else (),
            pending=("p",),
            decode=("d",),
        )
        output.muxpd_layers_per_slice = quota
        worker.execute_model(output)
        result = worker.sample_tokens(None)
        completed += quota
        assert type(result.muxpd_prefill_layers_completed) is int
        assert type(result.muxpd_prefill_layers_total) is int
        assert result.muxpd_prefill_layers_completed == completed
        assert result.muxpd_prefill_layers_total == 8
    assert result.muxpd_completed_prefill_ids == {"p"}
    worker.execute_model(schedule(decode=("d",)))
    result = worker.sample_tokens(None)
    assert result.muxpd_prefill_cohort_id is None
    assert result.muxpd_prefill_layers_completed == 0
    assert result.muxpd_prefill_layers_total == 0


@pytest.mark.integration
def test_unstarted_cohort_cancellation_retains_zero_progress_and_total(simulation):
    worker, _ = simulation
    worker.execute_model(
        schedule(new_p=("p",), pending=("p",), cancel=("p",), pause=True)
    )
    result = worker.sample_tokens(None)
    assert result.muxpd_cancelled_prefill_ids == {"p"}
    assert result.muxpd_prefill_layers_completed == 0
    assert result.muxpd_prefill_layers_total == 3
    worker.execute_model(schedule())
    result = worker.sample_tokens(None)
    assert result.muxpd_prefill_layers_completed == 0
    assert result.muxpd_prefill_layers_total == 0


@pytest.mark.integration
def test_shutdown_unwinds_prefill_before_releasing_communication(
    simulation, monkeypatch
):
    from vllm_muxpd.worker import NPUWorker

    worker, events = simulation
    worker.execute_model(schedule(new_p=("p",), pending=("p",), decode=("d",)))
    worker.sample_tokens(None)
    runner = worker.model_runner

    def restore():
        assert runner.closed == 1
        assert runner.sample_count == 0
        assert ("sync",) in events
        events.append(("restore",))

    worker._collective_router = SimpleNamespace(restore=restore)
    worker._prefill_groups = [
        SimpleNamespace(destroy=lambda: events.append(("destroy",)))
    ]
    monkeypatch.setattr(
        NPUWorker, "shutdown", lambda self: events.append(("native_shutdown",))
    )
    worker.shutdown()
    assert worker._prefill_task is None
    assert worker.decode_runner is None
    assert events.index(("restore",)) < events.index(("destroy",))
    assert events.index(("destroy",)) < events.index(("native_shutdown",))
    assert worker._prefill_groups == []
    assert worker._collective_router is None


@pytest.mark.integration
def test_shutdown_abort_join_failure_retains_task_and_resources_for_retry(
    simulation, monkeypatch
):
    from vllm_muxpd.worker import NPUWorker

    worker, events = simulation
    worker.execute_model(schedule(new_p=("p",), pending=("p",), decode=("d",)))
    worker.sample_tokens(None)
    task = worker._prefill_task
    streams = worker._streams.copy()
    decode = worker.decode_runner
    joins = []

    def prefill_join():
        joins.append("prefill")
        if len(joins) == 1:
            raise RuntimeError("abort P join failed")

    streams["prefill"].synchronize = prefill_join
    router = SimpleNamespace(restore=lambda: events.append(("restore",)))
    group = SimpleNamespace(destroy=lambda: events.append(("destroy",)))
    worker._collective_router = router
    worker._prefill_groups = [group]
    monkeypatch.setattr(
        NPUWorker, "shutdown", lambda self: events.append(("native_shutdown",))
    )
    with pytest.raises(RuntimeError, match="abort P join failed"):
        worker.shutdown()
    assert worker._prefill_task is task and task.tasks["prefill"].done
    assert worker._streams == streams and worker.decode_runner is decode
    assert worker._collective_router is router and worker._prefill_groups == [group]
    assert not worker._cancelled_ack
    assert not getattr(worker, "_muxpd_shutdown_complete", False)
    assert ("restore",) not in events and ("native_shutdown",) not in events
    # A transient join failure must not turn retry into KeyError('prefill') or
    # silently skip native cleanup via a prematurely set complete flag.
    worker.shutdown()
    worker.shutdown()
    assert len(joins) == 3  # failed abort, successful abort, parent group teardown
    assert worker._prefill_task is None and worker._cancelled_ack == {"p"}
    assert not worker._prefill_groups and worker._collective_router is None
    assert not worker._streams and worker.decode_runner is None
    assert events.count(("native_shutdown",)) == 1
    assert events.index(("restore",)) < events.index(("destroy",))
    assert events.index(("destroy",)) < events.index(("native_shutdown",))


def enable_fake_prefill_compile(worker, events):
    controller = SimpleNamespace(compile_full_prefill=True, active=False)

    @contextmanager
    def context():
        assert not controller.active
        controller.active = True
        events.append(("compile_enter",))
        try:
            yield
        finally:
            controller.active = False
            events.append(("compile_exit",))

    controller.full_prefill_compile_context = context
    worker._decode_graph_controller = controller
    return controller


def caller_records(events):
    return [
        event for event in events if event[0] == "event" and event[1][0] == "caller"
    ]


@pytest.mark.integration
@pytest.mark.parametrize("simulation", ["graph"], indirect=True)
def test_consecutive_graph_decode_skips_only_redundant_preparation_fence(simulation):
    worker, events = simulation
    worker.profiler = None
    caller = worker._execution_backend.capture()
    for iteration in range(3):
        start = len(events)
        worker.execute_model(schedule(decode=("d",)))
        receipt = worker.sample_tokens(None)
        step_events = events[start:]
        assert len(caller_records(step_events)) == (1 if iteration == 0 else 0)
        assert len([e for e in step_events if e[0] == "event"]) == (
            3 if iteration == 0 else 2
        )
        assert receipt.req_ids == ["d"]
        assert receipt.muxpd_prefill_progress_source == "none"
        assert worker._execution_backend.capture() == caller
        if iteration:
            assert not any(e[0] == "wait" and e[1] == "decode" for e in step_events)
    assert worker.muxpd_stats["decode_preparation_fences_skipped"] == 2


@pytest.mark.integration
@pytest.mark.parametrize("simulation", ["graph"], indirect=True)
def test_decode_prefill_decode_transitions_keep_preparation_fence(simulation):
    worker, events = simulation
    worker.profiler = None
    outputs = [
        schedule(decode=("d",)),
        schedule(new_p=("p",), pending=("p",), finish=True),
        schedule(decode=("d",)),
        schedule(decode=("d",)),
    ]
    for index, output in enumerate(outputs):
        start = len(events)
        worker.execute_model(output)
        worker.sample_tokens(None)
        assert len(caller_records(events[start:])) == (0 if index == 3 else 1)
    assert worker.muxpd_stats["decode_preparation_fences_skipped"] == 1


@pytest.mark.integration
@pytest.mark.parametrize("simulation", ["graph"], indirect=True)
def test_decode_reuses_latest_published_caller_tail_and_native_block_zero(
    simulation, monkeypatch
):
    worker, events = simulation
    worker.profiler = None
    worker.execute_model(schedule(decode=("d",)))
    worker.sample_tokens(None)
    # A supported external operation publishes its dependency through the
    # backend. The optimization must retain this tail, not assume it is D.
    worker._execution_backend.fence()
    caller_tail = worker._execution_backend._tail
    native = worker.decode_runner.execute_model

    def execute(output):
        assert worker._execution_backend.capture().stream is worker._streams["decode"]
        assert output.new_block_ids_to_zero == [42]
        events.append(("native_zero_on_decode",))
        return native(output)

    monkeypatch.setattr(worker.decode_runner, "execute_model", execute)
    start = len(events)
    output = schedule(decode=("d",))
    output.new_block_ids_to_zero = [42]
    worker.execute_model(output)
    worker.sample_tokens(None)
    step_events = events[start:]
    assert not caller_records(step_events)
    assert ("wait", "decode", caller_tail) in step_events
    assert step_events.index(("wait", "decode", caller_tail)) < step_events.index(
        ("native_zero_on_decode",)
    )


@pytest.mark.integration
@pytest.mark.parametrize("simulation", ["graph"], indirect=True)
def test_even_inactive_profiler_keeps_decode_preparation_fence(simulation):
    worker, events = simulation
    worker.profiler = None
    worker.execute_model(schedule(decode=("d",)))
    worker.sample_tokens(None)
    worker.profiler = SimpleNamespace(_active=False, step=lambda: None)
    start = len(events)
    worker.execute_model(schedule(decode=("d",)))
    worker.sample_tokens(None)
    assert len(caller_records(events[start:])) == 1
    assert not worker.muxpd_stats.get("decode_preparation_fences_skipped", 0)


@pytest.mark.integration
@pytest.mark.parametrize("simulation", ["graph"], indirect=True)
@pytest.mark.parametrize("failure", ["execute", "sample"])
def test_decode_preparation_skip_preserves_failure_and_retry_protocol(
    simulation, monkeypatch, failure
):
    worker, events = simulation
    worker.profiler = None
    worker.execute_model(schedule(decode=("d",)))
    worker.sample_tokens(None)
    caller = worker._execution_backend.capture()

    def fail(*args):
        raise ValueError("native failure")

    start = len(events)
    if failure == "execute":
        native = worker.decode_runner.execute_model
        monkeypatch.setattr(worker.decode_runner, "execute_model", fail)
        with pytest.raises(ValueError, match="native failure"):
            worker.execute_model(schedule(decode=("d",)))
        assert not worker._awaiting_sample and not worker._decode_only_fast_path
        assert worker._execution_backend.capture() == caller
        assert not caller_records(events[start:])
        monkeypatch.setattr(worker.decode_runner, "execute_model", native)
        start = len(events)
        worker.execute_model(schedule(decode=("d",)))
        worker.sample_tokens(None)
        assert len(caller_records(events[start:])) == 1
    else:
        worker.execute_model(schedule(decode=("d",)))
        monkeypatch.setattr(worker.decode_runner, "sample_tokens", fail)
        with pytest.raises(ValueError, match="native failure"):
            worker.sample_tokens(None)
        assert worker._awaiting_sample
        assert worker._execution_backend.capture() == caller
        assert not caller_records(events[start:])
        with pytest.raises(RuntimeError, match="sampler ACK"):
            worker.execute_model(schedule(decode=("d",)))


@pytest.mark.integration
@pytest.mark.parametrize("simulation", ["graph"], indirect=True)
@pytest.mark.parametrize("raw_hooks", [False, True])
@pytest.mark.parametrize("quota", [None, 3, 4])
@pytest.mark.parametrize("timed", [False, True])
def test_compiled_fresh_full_prefill_records_return_and_preserves_sampler_lane(
    simulation, monkeypatch, raw_hooks, quota, timed
):
    worker, events = simulation
    controller = enable_fake_prefill_compile(worker, events)
    if timed:
        enable_fake_slice_timing(worker, monkeypatch)
    native = worker.model_runner.execute_model

    def execute(output):
        assert controller.active
        assert list(output.num_scheduled_tokens) == ["p"]
        if raw_hooks:
            return native(output)
        worker.model_runner.execute_model_state = output
        events.append(("compiled_return",))
        return None

    monkeypatch.setattr(worker.model_runner, "execute_model", execute)
    sample = worker.model_runner.sample_tokens

    def sample_prefill(grammar):
        assert not controller.active
        assert worker._execution_backend.capture().stream is worker._streams["prefill"]
        assert worker._prefill_task.tasks["prefill"].done
        events.append(("prefill_sample",))
        return sample(grammar)

    monkeypatch.setattr(worker.model_runner, "sample_tokens", sample_prefill)
    output = schedule(new_p=("p",), pending=("p",), decode=("d",), finish=quota is None)
    output.muxpd_layers_per_slice = quota or 1
    worker.execute_model(output)
    assert not controller.active and worker._prefill_full_forward_return
    assert worker._prefill_task.tasks["prefill"].done
    assert worker._prefill_task.tasks["prefill"].completed_layers == (
        3 if raw_hooks else 0
    )
    assert worker.model_runner.sample_count == 0
    receipt = worker.sample_tokens(None)
    assert receipt.req_ids == ["d", "p"]
    assert receipt.muxpd_prefill_layers_completed == 3
    assert receipt.muxpd_prefill_progress_source == "full_forward_return"
    assert receipt.muxpd_completed_prefill_ids == {"p"}
    timing = receipt.muxpd_prefill_slice_timing
    if timed:
        assert timing["mode"] == "compiled_full_forward"
        assert timing["layer_delta"] == 3
        assert timing["first_slice"] and timing["device_complete"]
    else:
        assert timing is None
    assert worker.model_runner.sample_count == 1
    assert events.index(("compile_exit",)) < events.index(("prefill_sample",))
    assert worker._prefill_task is None


@pytest.mark.integration
@pytest.mark.parametrize("simulation", ["graph"], indirect=True)
@pytest.mark.parametrize("initial_pause", [False, True])
def test_existing_prefill_never_switches_to_compiled_fx(simulation, initial_pause):
    worker, events = simulation
    enable_fake_prefill_compile(worker, events)
    worker.execute_model(schedule(new_p=("p",), pending=("p",), pause=initial_pause))
    first = worker.sample_tokens(None)
    assert first.muxpd_prefill_layers_completed == (0 if initial_pause else 1)
    worker.execute_model(schedule(pending=("p",), finish=True))
    receipt = worker.sample_tokens(None)
    assert receipt.muxpd_prefill_progress_source == "layer_hooks"
    assert receipt.muxpd_prefill_layers_completed == 3
    assert ("compile_enter",) not in events


@pytest.mark.integration
@pytest.mark.parametrize("simulation", ["graph"], indirect=True)
@pytest.mark.parametrize("all_cancelled", [False, True])
def test_cancelled_new_prefill_uses_existing_hook_and_ack_protocol(
    simulation, all_cancelled
):
    worker, events = simulation
    enable_fake_prefill_compile(worker, events)
    pending = ("p",) if all_cancelled else ("p", "other")
    worker.execute_model(
        schedule(new_p=pending, pending=pending, cancel=("p",), finish=True)
    )
    receipt = worker.sample_tokens(None)
    assert ("compile_enter",) not in events
    assert receipt.muxpd_prefill_progress_source == "layer_hooks"
    assert receipt.muxpd_prefill_layers_completed == (0 if all_cancelled else 3)
    assert receipt.muxpd_cancelled_prefill_ids == {"p"}
    assert receipt.muxpd_completed_prefill_ids == (
        set() if all_cancelled else {"other"}
    )
    assert worker.model_runner.sample_count == (0 if all_cancelled else 1)


@pytest.mark.integration
@pytest.mark.parametrize("simulation", ["graph"], indirect=True)
@pytest.mark.parametrize(
    "failure", ["raises", "partial_hooks", "no_state", "immediate_output"]
)
def test_compiled_prefill_failure_never_fabricates_progress_receipt(
    simulation, monkeypatch, failure
):
    worker, events = simulation
    controller = enable_fake_prefill_compile(worker, events)

    def execute(output):
        if failure == "raises":
            raise ValueError("compiled forward failed")
        if failure == "partial_hooks":
            worker._interleaver.checkpoint(final=True)
        if failure != "no_state":
            worker.model_runner.execute_model_state = output
        return object() if failure == "immediate_output" else None

    monkeypatch.setattr(worker.model_runner, "execute_model", execute)
    with pytest.raises((ValueError, RuntimeError)):
        worker.execute_model(schedule(new_p=("p",), pending=("p",), finish=True))
    assert not controller.active and not worker._prefill_full_forward_return
    assert not worker._awaiting_sample
    assert worker._iteration_prefill_layers_completed == 0
    with pytest.raises(RuntimeError, match="executed iteration"):
        worker.sample_tokens(None)
