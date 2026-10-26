"""Experimental native batch queue for pure decode, with drained P transitions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from .decoupled_scheduler import DecoupledMuxPDScheduler


@dataclass(frozen=True)
class _QueuedStep:
    batch_id: str
    step: Any


class AsyncDecoupledMuxPDScheduler(DecoupledMuxPDScheduler):
    """Pipeline two D batches; admit each P/control transaction after a drain.

    Prefill still owns its original full-prompt allocation and synchronous ACK.
    Decode alone uses native one-token output placeholders. The queue stores
    complete retirement records rather than reusing the synchronous step slot.
    """

    def __init__(self, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        if vllm_config.use_v2_model_runner:
            raise ValueError("Async MuxPD supports only the V1 model runner")
        if (vllm_config.scheduler_config.batch_admission_policy_config or {}).get(
            "async_decode_pipeline"
        ) is not True:
            raise ValueError("Async MuxPD requires async_decode_pipeline=true")
        self._async_steps: deque[_QueuedStep] = deque()
        self._async_sequence = 0
        self._async_completed_batch: str | None = None
        self._async_retiring_decodes: frozenset[str] = frozenset()
        self._async_handoff_ids: set[str] = set()
        self._async_held_decode_blocks: dict[str, list[Any]] = {}
        self._async_held_decode_sessions: dict[str, str | None] = {}
        super().__init__(vllm_config, *args, **kwargs)

    def _queued_decode_ids(self) -> set[str]:
        return {
            rid
            for queued in self._async_steps
            for rid in queued.step.decode_output.num_scheduled_tokens
        }

    def add_request(self, request):
        if request.request_id in self._queued_decode_ids():
            raise ValueError("Request ID still has an in-flight decode output")
        super().add_request(request)

    def _muxpd_snapshots(self):
        snapshots = self._make_batch_request_snapshots()
        eligible = []
        for snapshot in snapshots:
            request = self.requests[snapshot.request_id]
            if self._is_blocked_waiting_status(request.status):
                if not getattr(self, self._native_contract.blocked_method)(request):
                    continue
            if request.spec_token_ids:
                raise RuntimeError("Async MuxPD does not support speculative tokens")
            # EngineCore in_flight_tokens counts scheduled prompt tokens too.
            # Only actual D output placeholders extend the generated context.
            eligible.append(
                replace(
                    snapshot,
                    num_output_tokens=(
                        snapshot.num_output_tokens + request.num_output_placeholders
                    ),
                )
            )
        return tuple(eligible)

    def _update_after_schedule(self, output):
        decode_ids = {
            rid
            for rid in output.num_scheduled_tokens
            if (request := self.requests[rid]).num_output_tokens > 0
            and request.num_computed_tokens
            >= request.num_tokens + request.num_output_placeholders - 1
        }
        Scheduler._update_after_schedule(self, output)
        for rid in decode_ids:
            if output.num_scheduled_tokens[rid] != 1:
                raise RuntimeError("Async MuxPD requires one token per decode")
            self.requests[rid].num_output_placeholders += 1

    def _update_request_with_output(self, request, new_token_ids):
        if request.request_id not in self._async_retiring_decodes:
            return Scheduler._update_request_with_output(self, request, new_token_ids)
        if request.async_tokens_to_discard > 0:
            request.async_tokens_to_discard -= 1
            return [], False
        status = request.status
        tokens, stopped = Scheduler._update_request_with_output(
            self, request, new_token_ids
        )
        request.num_output_placeholders -= len(tokens)
        if request.num_output_placeholders < 0:
            raise RuntimeError("Async MuxPD decode placeholder underflow")
        if status == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return tokens, stopped

    def schedule_batch(
        self,
        in_flight_batch_ids: frozenset[str],
        throttle_prefills: bool = False,
        *,
        in_flight_tokens: dict[str, int] | None = None,
    ):
        del in_flight_tokens
        if self._async_completed_batch is not None:
            raise RuntimeError("Async MuxPD requires the previous batch completion")
        if in_flight_batch_ids != frozenset(q.batch_id for q in self._async_steps):
            raise RuntimeError("Async MuxPD engine queue and scheduler disagree")
        if self._async_steps and (
            len(self._async_steps) >= 2
            or self._prefill_cohort is not None
            or self.waiting
            or self.skipped_waiting
            or any(q.step.cohort_id is not None for q in self._async_steps)
        ):
            return None
        output = self.schedule(throttle_prefills)
        step = self._decoupled_step
        if step is None or step.output is not output:
            raise RuntimeError("Async MuxPD did not create a retirement record")
        if self._async_steps and output.muxpd_prefill_cohort_id is not None:
            raise RuntimeError("Async MuxPD admitted prefill before draining decode")
        # Global previous-step membership does not imply D-lane membership.
        # P's first completed token must be supplied explicitly when D starts.
        cached = output.scheduled_cached_reqs
        handoffs = self._async_handoff_ids.intersection(cached.req_ids)
        handoffs.update(set(output.muxpd_prefill_request_ids) & set(cached.req_ids))
        if handoffs:
            cached.all_token_ids.update(
                {rid: self.requests[rid].all_token_ids.copy() for rid in handoffs}
            )
            self._async_handoff_ids.difference_update(handoffs)
        self._async_sequence += 1
        batch_id = f"muxpd-async-{self._async_sequence}"
        self._async_steps.append(_QueuedStep(batch_id, step))
        self._decoupled_step = None
        return batch_id, output

    def update_from_output(self, scheduler_output, model_runner_output):
        if not self._async_steps:
            raise RuntimeError("Async MuxPD received an unknown output")
        queued = self._async_steps[0]
        if queued.step.output is not scheduler_output:
            raise RuntimeError("Async MuxPD requires FIFO output retirement")
        self._decoupled_step = queued.step
        self._async_retiring_decodes = frozenset(
            queued.step.decode_output.num_scheduled_tokens
        )
        try:
            result = super().update_from_output(scheduler_output, model_runner_output)
        finally:
            self._async_retiring_decodes = frozenset()
        self._async_handoff_ids.update(
            set(model_runner_output.muxpd_completed_prefill_ids) & self.requests.keys()
        )
        self._async_steps.popleft()
        self._async_completed_batch = queued.batch_id
        self._release_decode_blocks()
        return result

    def on_batch_complete(self, batch_id):
        if batch_id != self._async_completed_batch:
            raise RuntimeError("Async MuxPD received an unknown batch completion")
        self._async_completed_batch = None

    def on_batch_abort(self, batch_id):
        # The engine treats worker failures as fatal. Keep allocations and
        # records owned until native shutdown; never manufacture a safe ACK.
        if batch_id not in {q.batch_id for q in self._async_steps} and (
            batch_id != self._async_completed_batch
        ):
            raise RuntimeError("Async MuxPD received an unknown batch failure")

    def set_pause_state(self, pause_state):
        if pause_state == PauseState.PAUSED_ALL and self._async_steps:
            raise RuntimeError("Async MuxPD must drain before keep-pause")
        return super().set_pause_state(pause_state)

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        if reset_running_requests and self._async_steps:
            raise RuntimeError(
                "Async MuxPD must drain before resetting running requests"
            )
        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def _free_request_blocks(self, request):
        rid = request.request_id
        if request.is_finished():
            self._async_handoff_ids.discard(rid)
        if rid not in self._queued_decode_ids():
            return super()._free_request_blocks(request)
        if request.is_finished():
            # Preserve the parent's host lifecycle cleanup while retaining the
            # physical blocks until the last queued D output acknowledges them.
            self._arrival_monotonic.pop(rid, None)
            self._prefill_first_decode_debt.discard(rid)
            self._decode_completed_at.pop(rid, None)
        if rid not in self._async_held_decode_blocks:
            self._async_held_decode_blocks[rid] = (
                self.kv_cache_manager.pop_blocks_for_free(request)
            )
            self._async_held_decode_sessions[rid] = getattr(request, "session_id", None)
        return self._native_contract.held_result(self._async_held_decode_blocks[rid])

    def _release_decode_blocks(self):
        live = self._queued_decode_ids()
        for rid in tuple(self._async_held_decode_blocks):
            if rid in live:
                continue
            blocks = self._async_held_decode_blocks[rid]
            try:
                if blocks:
                    self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
            except BaseException as error:
                self._kv_reclaim_error = error
                raise
            del self._async_held_decode_blocks[rid]
            session = self._async_held_decode_sessions.pop(rid)
            self._native_contract.emit_reclaimed(rid, session, len(blocks))
