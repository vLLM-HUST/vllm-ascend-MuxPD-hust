"""MuxPD scheduler extension for HUST vLLM's batch-admission API 1.1."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.batch_admission import (
    BATCH_ADMISSION_POLICY_API_VERSION,
    BatchRequest,
)
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .config import validate_engine_config
from .deployment import managed_runtime_preflight
from .evidence import emit_runtime_effective
from .native_compat import scheduler_contract
from .policy import Decision, MuxPDPolicy, remaining_tokens, request_phase

logger = init_logger(__name__)


class MuxPDScheduler(Scheduler):
    """Select homogeneous full-prefill/decode work, then use native allocation.

    Both EngineCore entry paths are overridden: synchronous EngineCore.step
    calls schedule(), whereas queued execution can call schedule_batch().
    Policy or invariant failures are fatal and never restore mixed batching.
    """

    def __init__(self, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        managed_runtime_preflight()
        if BATCH_ADMISSION_POLICY_API_VERSION != "1.1":
            raise RuntimeError("MuxPD requires HUST batch-admission API 1.1")
        self._native_contract = scheduler_contract(
            Scheduler,
            SchedulerOutput,
            ModelRunnerOutput,
            CachedRequestData,
            NewRequestData,
        )
        validate_engine_config(vllm_config)
        super().__init__(vllm_config, *args, **kwargs)
        if self.need_mamba_block_aligned_split:
            raise ValueError(
                "MuxPD cannot run with Mamba block-aligned prefill splitting"
            )
        self.muxpd_policy = MuxPDPolicy.from_config(
            self.scheduler_config.batch_admission_policy_config or {},
            vllm_config,
        )
        if (
            type(self) is MuxPDScheduler
            and self.muxpd_policy.config.execution != "serial"
        ):
            raise ValueError(
                f"execution={self.muxpd_policy.config.execution!r} requires "
                "the matching MuxPD scheduler and worker"
            )
        self.muxpd_stats: Counter[str] = Counter()
        # Detailed diagnostic records exist only when tracing is enabled.
        self.last_muxpd_step: dict[str, Any] | None = None
        self._muxpd_pending: tuple[SchedulerOutput, float, Decision, int] | None = None
        self._muxpd_trace = None
        self._muxpd_rejected: list[Request] = []
        trace_path = self.muxpd_policy.config.trace_path
        if trace_path:
            path = Path(trace_path.format(pid=os.getpid())).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._muxpd_trace = path.open("a", encoding="utf-8", buffering=1)

    def add_request(self, request: Request) -> None:
        if getattr(request, "replay_start", 0) != 0:
            raise ValueError("MuxPD does not support external partial KV replay")
        if request.has_encoder_inputs or request.mm_features:
            raise ValueError(
                "MuxPD currently supports text-only requests; "
                "encoder inputs may split prefills"
            )
        if request.pooling_params is not None or request.resumable:
            raise ValueError(
                "MuxPD supports generation without streaming-input continuation"
            )
        if request.spec_token_ids or request.num_output_placeholders:
            raise ValueError(
                "MuxPD does not support speculative or asynchronous requests"
            )
        if request.num_tokens >= self.max_model_len:
            raise ValueError(
                "MuxPD prompt must leave at least one token within max_model_len"
            )
        reason = self._unsupported_reason(request)
        if reason is not None:
            # EngineCore treats add_request and worker exceptions as fatal.
            # Finish only this request with a request-scoped error instead.
            logger.warning("MuxPD rejected request %s: %s", request.request_id, reason)
            request.status = RequestStatus.FINISHED_ERROR
            self._muxpd_rejected.append(request)
            return
        super().add_request(request)

    def _unsupported_reason(self, request: Request) -> str | None:
        params = request.sampling_params
        if params is None:
            return "generation requires sampling parameters"
        if params.logprobs is not None or params.prompt_logprobs is not None:
            return "logprobs are not supported"
        if params.structured_outputs is not None:
            return "structured outputs are not supported"
        return None

    def has_requests(self) -> bool:
        return super().has_requests() or bool(self._muxpd_rejected)

    def _finish_outputs(self, result: dict) -> dict:
        self._record_effective_outputs(result)
        rejected, self._muxpd_rejected = self._muxpd_rejected, []
        for request in rejected:
            row = EngineCoreOutput(
                request_id=request.request_id,
                new_token_ids=[],
                finish_reason=request.get_finished_reason(),
                trace_headers=request.trace_headers,
            )
            packet = result.get(request.client_index)
            if packet is None:
                packet = result[request.client_index] = EngineCoreOutputs(outputs=[row])
            else:
                packet.outputs.append(row)
            if self.finished_req_ids_dict is not None:
                # Same lifetime signal native _free_request sends to DP clients.
                finished = packet.finished_requests
                packet.finished_requests = (finished or set()) | {request.request_id}
        return result

    def _muxpd_snapshots(self) -> tuple[BatchRequest, ...]:
        snapshots = self._make_batch_request_snapshots()
        eligible: list[BatchRequest] = []
        for snapshot in snapshots:
            request = self.requests[snapshot.request_id]
            if self._is_blocked_waiting_status(request.status):
                if not getattr(self, self._native_contract.blocked_method)(request):
                    continue
            if request.spec_token_ids or request.num_output_placeholders:
                raise RuntimeError(
                    "MuxPD encountered unsupported in-flight or speculative tokens"
                )
            eligible.append(snapshot)
        return tuple(eligible)

    def _muxpd_native_schedule(
        self,
        decision: Decision,
        snapshots: tuple[BatchRequest, ...],
    ) -> SchedulerOutput:
        by_id = {request.request_id: request for request in snapshots}
        if self._batch_eligible_request_ids is not None:
            raise RuntimeError("MuxPD cannot nest an external admission allowlist")
        self._batch_eligible_request_ids = frozenset(decision.request_ids)
        try:
            output = super().schedule(throttle_prefills=False)
        finally:
            self._batch_eligible_request_ids = None
        for rid, tokens in output.num_scheduled_tokens.items():
            if rid not in decision.request_ids:
                raise RuntimeError(
                    f"MuxPD native scheduler escaped its allowlist: {rid}"
                )
            request = by_id[rid]
            if request_phase(request) != decision.phase:
                raise RuntimeError("MuxPD detected a mixed prefill/decode batch")
            expected = remaining_tokens(request)
            if tokens != expected:
                raise RuntimeError(
                    f"MuxPD refuses partial {decision.phase}: {rid!r} "
                    f"scheduled {tokens}, expected {expected}"
                )
            if decision.phase == "decode" and tokens != 1:
                raise RuntimeError(
                    "MuxPD decode must contain exactly one token per request"
                )
        return output

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if throttle_prefills:
            raise RuntimeError("MuxPD does not support DP prefill throttling")
        if self._muxpd_pending is not None:
            raise RuntimeError(
                "MuxPD requires update_from_output before scheduling another batch"
            )
        started = time.monotonic()
        snapshots = self._muxpd_snapshots()
        if self._pause_state == PauseState.PAUSED_ALL:
            decision = Decision(None, (), "paused_all")
        else:
            # Native PAUSED_NEW permits existing running work only.
            if self._pause_state != PauseState.UNPAUSED:
                snapshots = tuple(r for r in snapshots if r.state == "running")
            slots = self.max_num_running_reqs - len(self.running)
            decision = self.muxpd_policy.choose(
                snapshots,
                started,
                available_prefill_slots=max(0, slots),
            )
        output = self._muxpd_native_schedule(decision, snapshots)
        # A complete prompt can be temporarily blocked by occupied KV blocks.
        # Keep native allocation/preemption semantics, then progress decodes so
        # they can finish and release capacity. Preserve lifecycle notifications
        # produced by the first (empty) native scheduling pass.
        if not output.total_num_scheduled_tokens and decision.phase == "prefill":
            retry_snapshots = self._muxpd_snapshots()
            retry_decision = self.muxpd_policy.choose(
                retry_snapshots,
                started,
                available_prefill_slots=0,
                force_decode=True,
            )
            if retry_decision.request_ids:
                first_output = output
                snapshots, decision = retry_snapshots, retry_decision
                output = self._muxpd_native_schedule(decision, snapshots)
                output.finished_req_ids.update(first_output.finished_req_ids)
                output.preempted_req_ids.update(first_output.preempted_req_ids)
                self.muxpd_stats["prefill_capacity_deferrals"] += 1
            elif not self.running and decision.request_ids:
                raise RuntimeError(
                    "MuxPD cannot fit a complete prefill into available KV cache; "
                    "increase KV capacity or reduce max_model_len/input length"
                )
        self.muxpd_stats["steps"] += 1
        if output.total_num_scheduled_tokens:
            assert decision.phase is not None
            self.muxpd_policy.commit(
                decision.phase, tuple(output.num_scheduled_tokens), started
            )
            self.muxpd_stats[f"{decision.phase}_batches"] += 1
            self.muxpd_stats[f"{decision.phase}_tokens"] += (
                output.total_num_scheduled_tokens
            )
            by_id = {r.request_id: r for r in snapshots}
            contexts = sum(
                by_id[rid].context_length for rid in output.num_scheduled_tokens
            )
            self._muxpd_pending = (output, started, decision, contexts)
        else:
            self.muxpd_stats["empty_steps"] += 1
        self.muxpd_stats["preemptions"] += len(output.preempted_req_ids)
        self._record_muxpd_step(output, decision, snapshots, started)
        return output

    def schedule_batch(
        self,
        in_flight_batch_ids: frozenset[str],
        throttle_prefills: bool = False,
        *,
        in_flight_tokens: dict[str, int] | None = None,
    ) -> tuple[str | None, SchedulerOutput]:
        if in_flight_batch_ids or any((in_flight_tokens or {}).values()):
            raise RuntimeError("MuxPD supports one synchronous batch in flight")
        output = self.schedule(throttle_prefills)
        batch_id = (
            f"muxpd-{self.muxpd_stats['steps']}"
            if output.total_num_scheduled_tokens
            else None
        )
        return batch_id, output

    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_runner_output: Any
    ) -> Any:
        pending = self._muxpd_pending
        if (pending is not None and pending[0] is not scheduler_output) or (
            scheduler_output.total_num_scheduled_tokens and pending is None
        ):
            raise RuntimeError(
                "MuxPD received an output for an unknown or reordered batch"
            )
        result = super().update_from_output(scheduler_output, model_runner_output)
        if pending is not None:
            _, started, decision, contexts = pending
            completed = time.monotonic()
            duration_ms = 1000 * (completed - started)
            assert decision.phase is not None
            self.muxpd_policy.observe_batch(
                decision.phase,
                duration_ms,
                scheduler_output.total_num_scheduled_tokens,
                contexts,
            )
            if decision.phase == "decode":
                self.muxpd_policy.last_decode_at = completed
            self.muxpd_stats["completed_batches"] += 1
            self._muxpd_pending = None
            if self._muxpd_trace is not None:
                self._write_trace(
                    "complete",
                    completed,
                    phase=decision.phase,
                    duration_ms=duration_ms,
                    prefill_scale=self.muxpd_policy.prefill_scale,
                    decode_scale=self.muxpd_policy.decode_scale,
                )
        return self._finish_outputs(result)

    def _write_trace(self, event: str, now: float, **fields: Any) -> dict[str, Any]:
        """Append one JSONL record; callers check that tracing is enabled."""
        record = {
            "event": event,
            "time": now,
            "pid": os.getpid(),
            "step": self.muxpd_stats["steps"],
            **fields,
        }
        self._muxpd_trace.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def _record_effective_outputs(self, result: dict) -> None:
        # Count only outputs actually retired by native scheduling. A sampled
        # token discarded after a host abort is not an effective scheduler step.
        rows = [row for batch in result.values() for row in batch.outputs]
        tokens = sum(len(row.new_token_ids) for row in rows)
        requests = sum(bool(row.new_token_ids) for row in rows)
        if tokens:
            emit_runtime_effective(
                self,
                role="scheduler",
                request_count=requests,
                output_token_count=tokens,
            )
            emit_runtime_effective(
                self.muxpd_policy,
                role="admission",
                request_count=requests,
                output_token_count=tokens,
            )

    def _record_muxpd_step(
        self,
        output: SchedulerOutput,
        decision: Decision,
        snapshots: tuple[BatchRequest, ...],
        now: float,
    ) -> None:
        if self._muxpd_trace is None:
            return
        by_id = {r.request_id: r for r in snapshots}
        scheduled = output.num_scheduled_tokens
        self.last_muxpd_step = self._write_trace(
            "schedule",
            now,
            phase=decision.phase,
            request_ids=list(scheduled),
            scheduled_tokens=dict(scheduled),
            expected_tokens={rid: remaining_tokens(by_id[rid]) for rid in scheduled},
            computed_before={rid: by_id[rid].num_computed_tokens for rid in scheduled},
            request_phases={rid: request_phase(by_id[rid]) for rid in scheduled},
            preempted_req_ids=sorted(output.preempted_req_ids),
            finished_req_ids=sorted(output.finished_req_ids),
            policy=asdict(decision),
        )

    def export_muxpd_stats(self) -> dict[str, int]:
        return dict(self.muxpd_stats)

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            if self._muxpd_trace is not None:
                self._muxpd_trace.close()
                self._muxpd_trace = None
