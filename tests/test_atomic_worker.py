"""One-RPC transactions using native EngineCore and CPU lane simulations."""

from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_decoupled_worker import schedule
from test_decoupled_worker import simulation as simulation


@pytest.fixture
def atomic_worker(simulation):
    from vllm_muxpd.atomic_worker import AtomicDecoupledMuxPDNPUWorker

    worker, events = simulation
    worker.__class__ = AtomicDecoupledMuxPDNPUWorker
    return worker, events


def engine_step(worker, output):
    """Use the pinned engine's dispatch rather than recreating its branching."""
    from vllm.v1.engine.core import EngineCore

    calls = []
    retired = []

    def execute_model(scheduler_output, *, non_block):
        assert non_block
        calls.append("execute_model")
        future = Future()
        future.set_result(worker.execute_model(scheduler_output))
        return future

    def sample_tokens(grammar):
        pytest.fail("the engine must not issue a second sampling RPC")

    def update_from_output(scheduler_output, model_output):
        assert scheduler_output is output
        retired.append(model_output)
        return {0: model_output}

    engine = SimpleNamespace(
        scheduler=SimpleNamespace(
            has_requests=lambda: True,
            schedule=lambda throttle: output,
            get_grammar_bitmask=lambda scheduler_output: None,
            update_from_output=update_from_output,
        ),
        model_executor=SimpleNamespace(
            execute_model=execute_model, sample_tokens=sample_tokens
        ),
        _should_throttle_prefills=lambda: False,
        log_error_detail=lambda scheduler_output: nullcontext(),
        log_iteration_details=lambda scheduler_output: nullcontext(),
        _process_aborts_queue=lambda: calls.append("process_aborts"),
    )
    result, executed = EngineCore.step(engine)
    assert calls == ["execute_model", "process_aborts"]
    assert len(retired) == 1 and result == {0: retired[0]}
    assert executed == (output.total_num_scheduled_tokens > 0)
    assert not worker._awaiting_sample and not worker._pending_lanes
    return retired[0]


@pytest.mark.integration
def test_native_engine_samples_decode_once_per_rpc(atomic_worker):
    worker, events = atomic_worker
    for _ in range(3):
        result = engine_step(worker, schedule(decode=("d",)))
        assert result.req_ids == ["d"]
        assert result.sampled_token_ids == [[7]]
        assert result.muxpd_prefill_cohort_id is None
        assert result.muxpd_prefill_layers_completed == 0
    assert worker.decode_runner.sample_count == 3
    assert worker.model_runner.sample_count == 0
    assert events.count(("profiler",)) == 3


@pytest.mark.integration
def test_native_engine_retires_mixed_progress_and_final_prefill(atomic_worker):
    worker, _ = atomic_worker
    caller = worker._execution_backend.capture()
    for index in range(3):
        result = engine_step(
            worker,
            schedule(
                new_p=("p",) if index == 0 else (),
                pending=("p",),
                decode=("d",),
            ),
        )
        assert result.req_ids == (["d"] if index < 2 else ["d", "p"])
        assert result.muxpd_prefill_layers_completed == index + 1
        assert result.muxpd_prefill_layers_total == 3
        assert result.muxpd_completed_prefill_ids == (
            frozenset() if index < 2 else frozenset({"p"})
        )
        assert worker._execution_backend.capture() == caller
    assert worker.decode_runner.sample_count == 3
    assert worker.model_runner.sample_count == 1
    assert worker._prefill_task is None


@pytest.mark.integration
def test_zero_token_continuation_still_returns_progress_receipt(atomic_worker):
    worker, _ = atomic_worker
    first = engine_step(worker, schedule(new_p=("p",), pending=("p",)))
    assert first.req_ids == [] and first.muxpd_prefill_layers_completed == 1
    second = engine_step(worker, schedule(pending=("p",)))
    assert second.req_ids == [] and second.muxpd_prefill_layers_completed == 2
    final = engine_step(worker, schedule(pending=("p",), finish=True))
    assert final.req_ids == ["p"]
    assert final.muxpd_completed_prefill_ids == {"p"}
    assert worker.model_runner.sample_count == 1
    assert worker.decode_runner.sample_count == 0


@pytest.mark.integration
def test_cancellation_ack_still_follows_stream_synchronization(atomic_worker):
    worker, events = atomic_worker
    engine_step(worker, schedule(new_p=("p",), pending=("p",)))
    task = worker._prefill_task
    result = engine_step(worker, schedule(pending=("p",), cancel=("p",), pause=True))
    assert result.req_ids == []
    assert result.muxpd_cancelled_prefill_ids == {"p"}
    assert not result.muxpd_completed_prefill_ids
    assert ("sync",) in events
    assert task.tasks["prefill"].done
    assert worker._prefill_task is None
    assert worker.model_runner.sample_count == 0


@pytest.mark.integration
def test_duplicate_explicit_sampling_is_rejected(atomic_worker):
    worker, _ = atomic_worker
    worker.execute_model(schedule(decode=("d",)))
    with pytest.raises(RuntimeError, match="requires an executed iteration"):
        worker.sample_tokens(None)
    assert worker.decode_runner.sample_count == 1


@pytest.mark.integration
@pytest.mark.parametrize("phase", ["execute", "sample"])
def test_native_failure_propagates_and_restores_caller(
    atomic_worker, monkeypatch, phase
):
    worker, _ = atomic_worker
    caller = worker._execution_backend.capture()

    def fail(*args):
        raise ValueError("native failure")

    monkeypatch.setattr(
        worker.decode_runner,
        "execute_model" if phase == "execute" else "sample_tokens",
        fail,
    )
    with pytest.raises(ValueError, match="native failure"):
        worker.execute_model(schedule(decode=("d",)))
    assert worker._execution_backend.capture() == caller
    assert worker._awaiting_sample == (phase == "sample")
    if phase == "sample":
        with pytest.raises(RuntimeError, match="sampler ACK"):
            worker.execute_model(schedule(decode=("d",)))
