"""Deferred native output ownership and synchronous P receipts with CPU fakes."""

from types import SimpleNamespace

import pytest
from test_decoupled_worker import schedule
from test_decoupled_worker import simulation as simulation


@pytest.fixture
def async_worker(simulation):
    from vllm.v1.outputs import AsyncModelRunnerOutput

    from vllm_muxpd.async_worker import AsyncDecoupledMuxPDNPUWorker

    worker, events = simulation
    worker.__class__ = AsyncDecoupledMuxPDNPUWorker
    native_sample = worker.decode_runner.sample_tokens

    class Deferred(AsyncModelRunnerOutput):
        def __init__(self, output):
            self.output = output
            self.consumed = False

        def get_output(self):
            assert not self.consumed
            self.consumed = True
            events.append(("materialize", tuple(self.output.req_ids)))
            return self.output

    def sample(grammar):
        result = native_sample(grammar)
        events.append(("async_submit", tuple(result.req_ids)))
        return Deferred(result)

    worker.decode_runner.sample_tokens = sample
    return worker, events


def test_decode_outputs_survive_next_transaction_without_materialization(async_worker):
    from vllm.v1.outputs import AsyncModelRunnerOutput

    worker, events = async_worker
    assert worker.execute_model(schedule(decode=("d1",))) is None
    first = worker.sample_tokens(None)
    assert isinstance(first, AsyncModelRunnerOutput)
    assert not worker._awaiting_sample
    assert not any(event[0] == "materialize" for event in events)
    assert worker.execute_model(schedule(decode=("d2", "d3"))) is None
    second = worker.sample_tokens(None)
    assert not any(event[0] == "materialize" for event in events)
    # Mutable worker fields now describe a different transaction. Neither
    # deferred result may read them on the native output thread.
    worker._iteration_cohort_id = 900
    worker._iteration_prefill_layers_completed = 777
    output1, output2 = first.get_output(), second.get_output()
    assert output1.req_ids == ["d1"]
    assert output2.req_ids == ["d2", "d3"]
    for output in (output1, output2):
        assert output.muxpd_prefill_cohort_id is None
        assert output.muxpd_prefill_layers_completed == 0
        assert output.muxpd_prefill_layers_total == 0
        assert output.muxpd_prefill_progress_source == "none"
    with pytest.raises(RuntimeError, match="already consumed"):
        first.get_output()


def test_async_decode_keeps_caller_dependency_for_future_kv_zeroing(async_worker):
    worker, events = async_worker
    if worker._stream_mode != "overlap":
        pytest.skip("caller join is specific to the overlap backend")
    worker.execute_model(schedule(decode=("d",)))
    deferred = worker.sample_tokens(None)
    tail = worker._execution_backend._tails["decode"]
    assert ("wait", "caller", tail) in events
    assert not any(event[0] == "materialize" for event in events)
    deferred.get_output()


def test_mixed_decode_materializes_before_original_prefill_ack(async_worker):
    from vllm.v1.outputs import ModelRunnerOutput

    worker, events = async_worker
    worker.execute_model(schedule(new_p=("p",), pending=("p",), decode=("d",)))
    result = worker.sample_tokens(None)
    assert isinstance(result, ModelRunnerOutput)
    assert result.req_ids == ["d"]
    assert result.muxpd_prefill_layers_completed == 1
    assert ("materialize", ("d",)) in events
    assert worker._prefill_task is not None
    worker._abort_prefill()


def test_zero_token_prefill_completion_returns_its_sample_from_execute(async_worker):
    from vllm.v1.outputs import ModelRunnerOutput

    worker, _ = async_worker
    worker.execute_model(schedule(new_p=("p",), pending=("p",), decode=("d",)))
    worker.sample_tokens(None)
    continuation = schedule(pending=("p",), finish=True)
    assert continuation.total_num_scheduled_tokens == 0
    result = worker.execute_model(continuation)
    assert isinstance(result, ModelRunnerOutput)
    assert result.req_ids == ["p"]
    assert result.muxpd_completed_prefill_ids == frozenset({"p"})
    assert worker._prefill_task is None
    assert not worker._awaiting_sample
    with pytest.raises(RuntimeError, match="requires an executed iteration"):
        worker.sample_tokens(None)


def test_zero_token_full_cancel_returns_ack_from_execute(async_worker):
    worker, _ = async_worker
    worker.execute_model(schedule(new_p=("p",), pending=("p",), decode=("d",)))
    worker.sample_tokens(None)
    cancellation = schedule(pending=("p",), cancel=("p",))
    result = worker.execute_model(cancellation)
    assert result.req_ids == []
    assert result.muxpd_cancelled_prefill_ids == frozenset({"p"})
    assert worker._prefill_task is None
    assert not worker._awaiting_sample


def test_recomputed_prefill_restores_real_tokens_from_scheduler(async_worker):
    worker, _ = async_worker
    state = SimpleNamespace(num_prompt_tokens=3, output_token_ids=[-1, -1])
    worker.model_runner.requests["p"] = state
    output = schedule(new_p=("p",), pending=("p",))
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["p"], num_output_tokens=[2], all_token_ids={"p": [1, 2, 3, 8, 9]}
    )
    worker._restore_prefill_history(output)
    assert state.output_token_ids == [8, 9]
    assert worker.decode_runner.requests["p"].output_token_ids == [8, 9]


def test_recomputed_prefill_rejects_missing_history_before_mutation(async_worker):
    worker, _ = async_worker
    state = SimpleNamespace(num_prompt_tokens=3, output_token_ids=[-1])
    worker.model_runner.requests["p"] = state
    output = schedule(new_p=("p",), pending=("p",))
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["p"], num_output_tokens=[1], all_token_ids={}
    )
    with pytest.raises(RuntimeError, match="complete prefill history"):
        worker._restore_prefill_history(output)
    assert state.output_token_ids == [-1]
