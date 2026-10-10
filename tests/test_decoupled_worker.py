"""Persistent forward and cancellation protocol without allocating model weights."""

import importlib
import os
from types import SimpleNamespace

import pytest


@pytest.fixture(params=["fenced", "overlap"])
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
    worker._iteration_cohort_id = None
    worker._iteration_prefill_layers_completed = 0
    worker._iteration_prefill_layers_total = 0
    worker._cancelled_ack = set()
    worker._interleaver = None
    worker._pending_lanes = []
    worker._pending_order = []
    worker._layers_per_slice = 1
    worker._decoder_layer_count = 3
    worker._stream_mode = request.param
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

    if worker._stream_mode == "overlap":
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
    assert [
        (event[0], event[-1]) for event in events if event[0] in ("decode", "prefill")
    ] == [
        ("decode", 0),
        ("prefill", 0),
        ("decode", 0),
        ("prefill", 1),
        ("decode", 0),
        ("prefill", 2),
    ]


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
        if worker._stream_mode == "overlap":
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
