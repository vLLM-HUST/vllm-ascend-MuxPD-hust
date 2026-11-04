"""Exercise bounded asynchronous decode with the native CPU KV allocator."""

import time
from collections import defaultdict, deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import test_decoupled_scheduler as base
from test_decoupled_scheduler import make_scheduler as make_scheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus


@pytest.fixture
def make_async_scheduler(make_scheduler, monkeypatch):
    from vllm_muxpd.async_scheduler import AsyncDecoupledMuxPDScheduler

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    def construct(**kwargs):
        config = kwargs["vllm_config"]
        config.scheduler_config.async_scheduling = True
        config.scheduler_config.scheduler_cls = (
            "vllm_muxpd.async_scheduler.AsyncDecoupledMuxPDScheduler"
        )
        config.scheduler_config.batch_admission_policy_config[
            "async_decode_pipeline"
        ] = True
        config.parallel_config.worker_cls = (
            "vllm_muxpd.async_worker.AsyncDecoupledMuxPDNPUWorker"
        )
        return AsyncDecoupledMuxPDScheduler(**kwargs)

    monkeypatch.setattr(base, "DecoupledMuxPDScheduler", construct)
    return make_scheduler


class QueueDriver:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.pending = []

    def schedule(self):
        tokens = defaultdict(int)
        for _, output in self.pending:
            for rid, count in output.num_scheduled_tokens.items():
                tokens[rid] += count
        result = self.scheduler.schedule_batch(
            frozenset(batch_id for batch_id, _ in self.pending),
            in_flight_tokens=dict(tokens),
        )
        if result is not None:
            self.pending.append(result)
            return result[1]
        return None

    def retire(self, **kwargs):
        batch_id, output = self.pending[0]
        result = self.scheduler.update_from_output(
            output, base.receipt(output, **kwargs)
        )
        self.scheduler.on_batch_complete(batch_id)
        self.pending.pop(0)
        return result

    def start_decode(self, rid="d", *, tokens=8, max_tokens=32):
        req = base.request(rid, tokens=tokens, max_tokens=max_tokens)
        self.scheduler.add_request(req)
        output = self.schedule()
        assert output.muxpd_finish_prefill
        assert req.num_output_placeholders == 0
        self.retire(completed={rid})
        return req


def test_prefill_first_token_is_transferred_before_async_decode(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    req = driver.start_decode()
    assert req.num_output_placeholders == 0
    output = driver.schedule()
    assert output.num_scheduled_tokens == {"d": 1}
    assert output.scheduled_cached_reqs.all_token_ids["d"] == list(req.all_token_ids)
    assert output.scheduled_cached_reqs.all_token_ids["d"][-1] == 2
    assert req.num_output_placeholders == 1
    driver.retire()
    assert req.num_output_tokens == 2 and req.num_output_placeholders == 0


def test_two_decode_batches_pipeline_and_retire_fifo(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    req = driver.start_decode()
    first = driver.schedule()
    second = driver.schedule()
    assert first is not second
    assert first.num_scheduled_tokens == second.num_scheduled_tokens == {"d": 1}
    assert req.num_output_placeholders == 2
    assert req.num_computed_tokens == req.num_prompt_tokens + 2
    assert req.num_output_tokens == 1
    assert driver.schedule() is None
    assert base.output_ids(driver.retire()) == {"d"}
    assert req.num_output_tokens == 2 and req.num_output_placeholders == 1
    assert base.output_ids(driver.retire()) == {"d"}
    assert req.num_output_tokens == 3 and req.num_output_placeholders == 0
    assert not scheduler._async_steps


def test_new_prefill_drains_both_decode_batches_before_admission(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    decode = driver.start_decode()
    driver.schedule()
    driver.schedule()
    prefill = base.request("p", tokens=64)
    scheduler.add_request(prefill)
    assert driver.schedule() is None
    assert prefill.num_computed_tokens == 0
    driver.retire()
    assert driver.schedule() is None
    driver.retire()
    output = driver.schedule()
    assert output.muxpd_prefill_request_ids == {"p"}
    assert output.num_scheduled_tokens == {"d": 1, "p": 64}
    assert prefill.num_output_placeholders == 0
    assert decode.num_output_placeholders == 1
    assert driver.schedule() is None
    driver.retire()
    assert driver.schedule().muxpd_pending_prefill_ids == {"p"}
    assert driver.schedule() is None
    driver.retire()


def test_zero_token_prefill_completion_is_not_lost(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    driver.start_decode()
    scheduler.add_request(base.request("p", tokens=64))
    first = driver.schedule()
    assert first.muxpd_prefill_request_ids == {"p"}
    driver.retire()
    scheduler.finish_requests("d", RequestStatus.FINISHED_ABORTED)
    continuation = driver.schedule()
    assert continuation.total_num_scheduled_tokens == 0
    assert continuation.muxpd_finish_prefill
    assert continuation.muxpd_pending_prefill_ids == {"p"}
    assert driver.schedule() is None
    assert base.output_ids(driver.retire(completed={"p"})) == {"p"}
    assert scheduler.requests["p"].num_output_tokens == 1
    assert scheduler.requests["p"].num_output_placeholders == 0


def test_max_tokens_accounts_for_in_flight_outputs(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    req = driver.start_decode(max_tokens=3)
    assert driver.schedule().num_scheduled_tokens == {"d": 1}
    assert driver.schedule().num_scheduled_tokens == {"d": 1}
    driver.retire()
    assert req.num_output_tokens == 2
    extra = driver.schedule()
    assert extra is None or "d" not in extra.num_scheduled_tokens
    driver.retire()
    assert req.num_output_tokens == 3
    assert req.num_output_placeholders == 0
    assert "d" not in scheduler.requests
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_abort_holds_decode_blocks_until_last_ack_and_rejects_id_reuse(
    make_async_scheduler,
):
    scheduler = make_async_scheduler(trace=False, num_blocks=12)
    driver = QueueDriver(scheduler)
    driver.start_decode(tokens=32)
    driver.schedule()
    driver.schedule()
    pool = scheduler.kv_cache_manager.block_pool
    before = pool.get_num_free_blocks()
    scheduler.finish_requests("d", RequestStatus.FINISHED_ABORTED)
    assert "d" not in scheduler.requests
    assert pool.get_num_free_blocks() == before
    with pytest.raises(ValueError):
        scheduler.add_request(base.request("d", tokens=8))
    assert not base.output_ids(driver.retire())
    assert pool.get_num_free_blocks() == before
    with pytest.raises(ValueError):
        scheduler.add_request(base.request("d", tokens=8))
    assert not base.output_ids(driver.retire())
    assert pool.get_num_free_blocks() > before
    replacement = base.request("d", tokens=8)
    scheduler.add_request(replacement)
    output = driver.schedule()
    assert output.muxpd_prefill_request_ids == {"d"}
    assert replacement.num_output_tokens == 0
    driver.retire(completed={"d"})
    assert replacement.num_output_tokens == 1


def test_reordered_receipt_does_not_consume_pending_fifo(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    req = driver.start_decode()
    first = driver.schedule()
    second = driver.schedule()
    with pytest.raises(RuntimeError):
        scheduler.update_from_output(second, base.receipt(second))
    assert len(scheduler._async_steps) == 2
    assert req.num_output_tokens == 1 and req.num_output_placeholders == 2
    assert driver.pending[0][1] is first
    driver.retire()
    driver.retire()
    assert req.num_output_tokens == 3 and req.num_output_placeholders == 0


def test_mismatched_engine_queue_is_rejected_before_scheduling(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    req = driver.start_decode()
    driver.schedule()
    computed = req.num_computed_tokens
    with pytest.raises(RuntimeError):
        scheduler.schedule_batch(frozenset())
    assert req.num_computed_tokens == computed
    assert req.num_output_placeholders == 1
    driver.retire()


def test_cancelled_prefill_requires_zero_token_ack(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    driver.start_decode()
    scheduler.add_request(base.request("p", tokens=64))
    driver.schedule()
    driver.retire()
    scheduler.finish_requests(("d", "p"), RequestStatus.FINISHED_ABORTED)
    held = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    output = driver.schedule()
    assert output.total_num_scheduled_tokens == 0
    assert output.muxpd_cancel_prefill_request_ids == {"p"}
    assert scheduler.has_requests()
    assert driver.schedule() is None
    driver.retire(cancelled={"p"})
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() > held
    assert not scheduler._prefill_cohort
    cleanup = driver.schedule()
    assert cleanup.total_num_scheduled_tokens == 0
    assert cleanup.finished_req_ids == {"p"}
    driver.retire()
    assert not scheduler.has_requests()


def test_preempted_decode_drains_before_full_history_prefill(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False, num_blocks=12)
    driver = QueueDriver(scheduler)
    req = driver.start_decode(tokens=32)
    driver.schedule()
    driver.schedule()
    pool = scheduler.kv_cache_manager.block_pool
    before = pool.get_num_free_blocks()
    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    assert req.status == RequestStatus.PREEMPTED
    assert pool.get_num_free_blocks() == before
    assert driver.schedule() is None
    driver.retire()
    assert req.num_output_placeholders == 1
    assert pool.get_num_free_blocks() == before
    assert driver.schedule() is None
    driver.retire()
    assert req.num_output_placeholders == 0
    assert req.num_output_tokens == 3
    assert pool.get_num_free_blocks() > before
    output = driver.schedule()
    assert output.muxpd_prefill_request_ids == {"d"}
    assert output.num_scheduled_tokens == {"d": 35}
    cached = output.scheduled_cached_reqs
    assert cached.resumed_req_ids == {"d"}
    assert cached.all_token_ids["d"] == list(req.all_token_ids)
    assert cached.all_token_ids["d"][-3:] == [2, 2, 2]
    assert all(token >= 0 for token in cached.all_token_ids["d"])
    driver.retire(completed={"d"})
    assert req.num_output_placeholders == 0
    assert req.num_output_tokens == 4


def test_existing_pinned_prefill_cannot_bypass_preemption_guard_as_resumed(
    make_async_scheduler, monkeypatch
):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    driver.start_decode()
    scheduler.add_request(base.request("p", tokens=64))
    driver.schedule()
    driver.retire()
    cohort = scheduler._prefill_cohort
    original = scheduler._schedule_while_prefilling

    def report_illegal_preemption(now):
        output = original(now)
        assert not output.muxpd_prefill_request_ids
        output.preempted_req_ids = set(output.preempted_req_ids or ()) | {"p"}
        output.scheduled_cached_reqs.resumed_req_ids.add("p")
        return output

    monkeypatch.setattr(
        scheduler, "_schedule_while_prefilling", report_illegal_preemption
    )
    with pytest.raises(RuntimeError, match="preempted a pinned prefill"):
        driver.schedule()
    assert scheduler._prefill_cohort is cohort
    assert cohort.pending_ids == {"p"}
    assert not driver.pending


def test_skipped_waiting_also_drains_before_new_transaction(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    driver.start_decode()
    driver.schedule()
    pending = base.request("p", tokens=64)
    scheduler.add_request(pending)
    scheduler.waiting.remove_request(pending)
    scheduler.skipped_waiting.add_request(pending)
    assert not scheduler.waiting and scheduler.skipped_waiting
    assert driver.schedule() is None
    assert pending.num_computed_tokens == 0
    driver.retire()
    output = driver.schedule()
    assert output.muxpd_prefill_request_ids == {"p"}
    assert pending.num_computed_tokens == 64


def test_keep_pause_rejects_queued_decode_before_mutating_state(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    req = driver.start_decode()
    driver.schedule()
    driver.schedule()
    with pytest.raises(RuntimeError, match="drain"):
        scheduler.set_pause_state(PauseState.PAUSED_ALL)
    assert scheduler.pause_state == PauseState.UNPAUSED
    assert req.num_output_placeholders == 2
    driver.retire()
    driver.retire()
    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    assert scheduler.pause_state == PauseState.PAUSED_ALL


def test_reset_running_rejects_queued_decode_before_freeing_kv(make_async_scheduler):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    req = driver.start_decode()
    driver.schedule()
    pool = scheduler.kv_cache_manager.block_pool
    before = pool.get_num_free_blocks()
    computed = req.num_computed_tokens
    with pytest.raises(RuntimeError, match="drain"):
        scheduler.reset_prefix_cache(reset_running_requests=True)
    assert pool.get_num_free_blocks() == before
    assert req.num_computed_tokens == computed
    assert req.num_output_placeholders == 1
    assert req.status == RequestStatus.RUNNING
    driver.retire()
    scheduler.reset_prefix_cache(reset_running_requests=True)


@pytest.mark.parametrize("cancel_prefill", [False, True])
def test_native_engine_queue_retires_zero_token_control_without_sample_rpc(
    make_async_scheduler, cancel_prefill
):
    scheduler = make_async_scheduler(trace=False)
    driver = QueueDriver(scheduler)
    driver.start_decode()
    scheduler.add_request(base.request("p", tokens=64))
    driver.schedule()
    driver.retire()
    scheduler.finish_requests("d", RequestStatus.FINISHED_ABORTED)
    if cancel_prefill:
        scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    assert not driver.pending
    calls = []

    def execute_model(output, *, non_block):
        assert non_block and output.total_num_scheduled_tokens == 0
        assert output.muxpd_pending_prefill_ids == {"p"}
        calls.append("execute_model")
        future = Future()
        future.set_result(
            base.receipt(
                output,
                completed=() if cancel_prefill else {"p"},
                cancelled={"p"} if cancel_prefill else (),
            )
        )
        return future

    def forbidden_sample(*args, **kwargs):
        pytest.fail("zero-token control must use execute_model's receipt")

    engine = SimpleNamespace(
        scheduler=scheduler,
        batch_queue=deque(),
        batch_queue_size=2,
        is_ec_consumer=True,
        is_pooling_model=False,
        model_executor=SimpleNamespace(
            execute_model=execute_model, sample_tokens=forbidden_sample
        ),
        _should_throttle_prefills=lambda: False,
        log_error_detail=lambda output: nullcontext(),
        log_iteration_details=lambda output: nullcontext(),
        _process_aborts_queue=lambda: calls.append("process_aborts"),
    )
    assert EngineCore.step_with_batch_queue(engine) == (None, False)
    assert len(engine.batch_queue) == len(scheduler._async_steps) == 1
    result, executed = EngineCore.step_with_batch_queue(engine)
    assert not executed
    assert base.output_ids(result) == (set() if cancel_prefill else {"p"})
    assert calls == ["execute_model", "process_aborts"]
    assert not engine.batch_queue and not scheduler._async_steps
    assert scheduler._async_completed_batch is None
    assert scheduler._prefill_cohort is None
    if not cancel_prefill:
        assert scheduler.requests["p"].num_output_tokens == 1
        assert scheduler.requests["p"].num_output_placeholders == 0
