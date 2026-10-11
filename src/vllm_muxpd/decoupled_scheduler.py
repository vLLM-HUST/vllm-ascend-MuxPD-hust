"""Keep full-prefill state alive while retiring decode tokens each iteration.

This is cooperative layer execution, not a claim of concurrent device kernels.
One prefill cohort owns its KV allocation until completion or cancellation ACK.
The normal synchronous EngineCore.step path must execute every returned output,
including zero-token continuation and cancellation-control steps.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from .multiplex_scheduler import LayeredMuxPDScheduler
from .policy import Decision, remaining_tokens, request_phase
from .scheduler import MuxPDScheduler

_GUARD_ESTIMATE_CACHE_LIMIT = 256


def _output_for_update(output: SchedulerOutput, ids: set[str]) -> SchedulerOutput:
    """Select the exact original token costs for a native result retirement.

    Request-registration and lifecycle messages were already sent to the worker;
    they must not be replayed while retiring the two independent phase results.
    Native request retirement uses this token map and the original model-output
    row indices. Native MFU accounting requires the omitted registration rows,
    so that optional metric is rejected before scheduler initialization.
    """
    tokens = {rid: n for rid, n in output.num_scheduled_tokens.items() if rid in ids}
    return replace(
        output,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=tokens,
        total_num_scheduled_tokens=sum(tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        preempted_req_ids=set(),
        free_encoder_mm_hashes=[],
        new_block_ids_to_zero=None,
    )


@dataclass
class _PrefillCohort:
    sequence: int
    original_output: SchedulerOutput
    pending_ids: set[str]
    arrival_monotonic: float
    completed_layers: int = 0


@dataclass
class _Step:
    output: SchedulerOutput
    decode_output: SchedulerOutput
    cohort_id: int | None
    started_at: float
    expected_layers: int
    calibration_phase: str | None
    deferred_decode_ids: frozenset[str]
    guard_decode_shape: tuple[int, int, int] | None = None


class DecoupledMuxPDScheduler(LayeredMuxPDScheduler):
    """Admit complete prompts once, then independently retire P and D outputs."""

    def __init__(self, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        if vllm_config.observability_config.enable_mfu_metrics:
            raise ValueError(
                "Decoupled MuxPD does not support enable_mfu_metrics; "
                "native accounting cannot represent cross-iteration prefill"
            )
        self._prefill_cohort: _PrefillCohort | None = None
        self._cohort_sequence = 0
        self._decoupled_step: _Step | None = None
        self._held_prefill_blocks: dict[str, list[Any]] = {}
        self._arrival_monotonic: dict[str, float] = {}
        self._prefill_latency: dict[int, tuple[float, int]] = {}
        self._decode_latency_ms: float | None = None
        self._latency_samples = {"prefill": 0, "decode": 0}
        self._prefill_first_decode_debt: set[str] = set()
        self._prefill_first_admission: dict[str, Any] = {}
        # Guard estimates are separate from whole-P (potentially compiled) EWMA.
        # Exact P token-length multisets and D batch/context buckets prevent a
        # small, unrelated calibration shape from silently licensing more work.
        self._guard_prefill_cost: OrderedDict[
            tuple[tuple[int, ...], bool, int], dict[str, tuple[float, int]]
        ] = OrderedDict()
        self._guard_decode_cost: OrderedDict[
            tuple[int, int, int], tuple[float, int]
        ] = OrderedDict()
        self._guard_cache_evictions = {"prefill": 0, "decode": 0}
        self._guard_live_decodes: dict[str, int] = {}
        self._decode_completed_at: dict[str, float] = {}
        # Reuse the common validated allocator setup, without the layered-only
        # execution/worker checks in LayeredMuxPDScheduler.__init__.
        MuxPDScheduler.__init__(self, vllm_config, *args, **kwargs)
        self._decoder_layer_count = getattr(
            vllm_config.model_config.hf_text_config, "num_hidden_layers", None
        )
        if type(self._decoder_layer_count) is not int or self._decoder_layer_count < 1:
            raise ValueError("Decoupled MuxPD requires the model's decoder layer count")
        self._layered_pending = None
        if self.muxpd_policy.config.execution != "decoupled":
            raise ValueError("DecoupledMuxPDScheduler requires execution='decoupled'")
        expected_worker = "vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker"
        if vllm_config.parallel_config.worker_cls != expected_worker:
            raise ValueError("DecoupledMuxPDScheduler requires DecoupledMuxPDNPUWorker")
        if self.defer_block_free or self.connector is not None:
            raise ValueError("Decoupled MuxPD does not support native deferred frees")
        if self.enable_return_routed_experts:
            raise ValueError("Decoupled MuxPD does not support routed expert outputs")
        if self.lora_config is not None:
            raise ValueError("Decoupled MuxPD does not support LoRA")

    def add_request(self, request: Request) -> None:
        if not math.isfinite(request.arrival_time):
            raise ValueError("MuxPD request arrival_time must be finite")
        if request.use_structured_output:
            raise ValueError("Decoupled MuxPD does not support structured outputs")
        if request.lora_request is not None:
            raise ValueError("Decoupled MuxPD does not support LoRA")
        if (
            self._prefill_cohort
            and request.request_id in self._prefill_cohort.pending_ids
        ):
            raise ValueError("Request ID is still awaiting prefill retirement")
        super().add_request(request)
        # Native Request uses epoch seconds. Convert once to the scheduler clock
        # while retaining time already spent queued before engine admission.
        self._arrival_monotonic[request.request_id] = time.monotonic() - max(
            0.0, time.time() - request.arrival_time
        )

    def _cancelled_prefill_ids(self) -> set[str]:
        if self._prefill_cohort is None:
            return set()
        return self._prefill_cohort.pending_ids - self.requests.keys()

    def _free_request_blocks(self, request: Request) -> None:
        if request.is_finished():
            self._arrival_monotonic.pop(request.request_id, None)
            self._prefill_first_decode_debt.discard(request.request_id)
            self._decode_completed_at.pop(request.request_id, None)
        cohort = self._prefill_cohort
        if cohort is not None and request.request_id in cohort.pending_ids:
            # Remove native ownership immediately, but retain the block refcounts
            # until the worker confirms the suspended forward no longer uses them.
            if request.request_id in self._held_prefill_blocks:
                raise RuntimeError("Prefill blocks were detached more than once")
            self._held_prefill_blocks[request.request_id] = (
                self.kv_cache_manager.pop_blocks_for_free(request)
            )
            return
        super()._free_request_blocks(request)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        if (
            self._prefill_cohort
            and request.request_id in self._prefill_cohort.pending_ids
        ):
            raise RuntimeError("Cannot preempt a pending decoupled prefill")
        super()._preempt_request(request, timestamp)
        self._prefill_first_decode_debt.discard(request.request_id)
        self._decode_completed_at.pop(request.request_id, None)

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        if self._prefill_cohort is not None and reset_running_requests:
            raise RuntimeError("Cannot reset running requests with a pending prefill")
        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def set_pause_state(self, pause_state: PauseState) -> None:
        if pause_state == PauseState.PAUSED_ALL and self._prefill_cohort is not None:
            # Native keep/sleep may clear KV or unload model memory immediately
            # after this transition. A suspended forward still references both.
            # Reject before mutating pause state; wait/abort can drain normally.
            raise RuntimeError(
                "Cannot keep-pause with a pending decoupled prefill; "
                "drain or abort the cohort first"
            )
        super().set_pause_state(pause_state)

    def has_requests(self) -> bool:
        # Aborted requests have left native queues, but their pinned KV and the
        # worker's suspended forward still require an explicit cancellation ACK.
        return super().has_requests() or bool(self._cancelled_prefill_ids())

    def schedule_batch(self, *args, **kwargs):
        raise RuntimeError(
            "Decoupled MuxPD requires synchronous EngineCore.step; "
            "the batch queue cannot retire zero-token prefill progress"
        )

    def _schedule_while_prefilling(self, now: float) -> SchedulerOutput:
        cohort = self._prefill_cohort
        assert cohort is not None
        original_running = self.running
        original_limit = self.max_num_running_reqs
        pending_ids = cohort.pending_ids
        self.running = [r for r in original_running if r.request_id not in pending_ids]
        # Cancelled-but-unacknowledged prefill members still reserve a worker slot.
        self.max_num_running_reqs = max(0, original_limit - len(pending_ids))
        try:
            snapshots = tuple(
                r
                for r in self._muxpd_snapshots()
                if r.state == "running"
                and request_phase(r) == "decode"
                and remaining_tokens(r) > 0
            )
            if self._pause_state == PauseState.PAUSED_ALL:
                snapshots = ()
            if self.muxpd_policy.config.decode_budget_guard:
                self._guard_live_decodes = {
                    r.request_id: r.context_length for r in snapshots
                }
            selected = snapshots[
                : min(
                    self.max_num_running_reqs,
                    self.max_num_scheduled_tokens,
                )
            ]
            if any(remaining_tokens(r) != 1 for r in selected):
                raise RuntimeError(
                    "Decoupled decode must contain one token per request"
                )
            output = self._schedule_layered_selection(
                tuple(r.request_id for r in selected),
                snapshots,
            )
        finally:
            # Keep native decode preemption/removal, but put the protected prefill
            # requests back in their original positions for finish_requests().
            surviving = {r.request_id: r for r in self.running}
            for request in original_running:
                if (
                    request.request_id in pending_ids
                    and request.request_id in self.requests
                ):
                    surviving[request.request_id] = request
            original_ids = {r.request_id for r in original_running}
            self.running = [
                surviving[r.request_id]
                for r in original_running
                if r.request_id in surviving
            ] + [r for rid, r in surviving.items() if rid not in original_ids]
            self.max_num_running_reqs = original_limit
        output.muxpd_prefill_request_ids = frozenset()
        output.muxpd_layers_per_slice = self.muxpd_policy.config.layers_per_slice
        self.muxpd_stats["steps"] += 1
        self.muxpd_stats["preemptions"] += len(output.preempted_req_ids or ())
        if output.total_num_scheduled_tokens:
            self.muxpd_policy.commit("decode", tuple(output.num_scheduled_tokens), now)
        decision = Decision(
            "decode" if output.total_num_scheduled_tokens else None,
            tuple(output.num_scheduled_tokens),
            "pending_prefill_decode_progress",
        )
        self._record_muxpd_step(output, decision, snapshots, now)
        return output

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if throttle_prefills or self._decoupled_step is not None:
            raise RuntimeError("Decoupled MuxPD requires one synchronous worker step")
        started = time.monotonic()
        self._guard_live_decodes = {}
        self._prefill_first_admission = {
            "enabled": self.muxpd_policy.config.prefill_first,
            "decode_budget_guard": self.muxpd_policy.config.decode_budget_guard,
            "reason": "pending_cohort" if self._prefill_cohort else "unchanged",
            "decode_debt_ids": sorted(self._prefill_first_decode_debt),
            "deferred_decode_ids": [],
        }
        if self._prefill_cohort is None:
            output = LayeredMuxPDScheduler.schedule(self, throttle_prefills=False)
            self._layered_pending = None
            new_prefill_ids = set(output.muxpd_prefill_request_ids)
            if new_prefill_ids:
                self._cohort_sequence += 1
                self._prefill_cohort = _PrefillCohort(
                    self._cohort_sequence,
                    _output_for_update(output, new_prefill_ids),
                    new_prefill_ids,
                    min(self._arrival_monotonic[rid] for rid in new_prefill_ids),
                )
        else:
            output = self._schedule_while_prefilling(started)
            new_prefill_ids = set()
        decode_ids = set(output.num_scheduled_tokens) - new_prefill_ids
        if self.muxpd_policy.config.decode_budget_guard:
            running_ids = {r.request_id for r in self.running}
            self._guard_live_decodes = {
                rid: context
                for rid, context in self._guard_live_decodes.items()
                if rid in running_ids
            }
        deferred_decode_ids = frozenset()
        if self._prefill_first_admission["reason"] in (
            "deadline_prefill_first",
            "full_budget_prefill_first",
            "token_budget_prefill_first",
        ):
            if new_prefill_ids and not decode_ids:
                deferred_decode_ids = frozenset(
                    self._prefill_first_admission["deferred_decode_ids"]
                )
            else:
                # The complete prompt could not acquire KV. The shared native
                # fallback has already progressed decode without advancing P.
                self._prefill_first_admission["reason"] = "prefill_kv_fallback"
                self._prefill_first_admission["deferred_decode_ids"] = []
        cohort = self._prefill_cohort
        paused = self._pause_state == PauseState.PAUSED_ALL
        # Native finish_requests removes host state immediately. Its worker-side
        # cleanup notification must wait: the suspended prefill still owns these
        # rows. The separate cancellation control requests a safe ACK instead.
        if cohort is not None:
            output.finished_req_ids.difference_update(cohort.pending_ids)
            if set(output.preempted_req_ids or ()) & cohort.pending_ids:
                raise RuntimeError("Native scheduler preempted a pinned prefill")
        output.muxpd_execution = "decoupled"
        output.muxpd_pending_prefill_ids = frozenset(
            cohort.pending_ids if cohort else ()
        )
        output.muxpd_cancel_prefill_request_ids = frozenset(
            self._cancelled_prefill_ids()
        )
        output.muxpd_prefill_cohort_id = cohort.sequence if cohort else None
        output.muxpd_pause_prefill = paused
        output.muxpd_finish_prefill = bool(
            cohort is not None
            and not paused
            and not (
                self._guard_live_decodes
                if self.muxpd_policy.config.decode_budget_guard
                else decode_ids
            )
        )
        quota_info = self._apply_layer_quota(output, started)
        output.muxpd_prefill_layers_completed_before = (
            cohort.completed_layers if cohort else 0
        )
        output.muxpd_prefill_layers_total = self._decoder_layer_count if cohort else 0
        fully_cancelled = bool(
            cohort
            and set(output.muxpd_cancel_prefill_request_ids) == cohort.pending_ids
        )
        expected_layers = 0
        if cohort:
            expected_layers = cohort.completed_layers
            if not paused and not fully_cancelled:
                expected_layers = (
                    self._decoder_layer_count
                    if output.muxpd_finish_prefill
                    else min(
                        self._decoder_layer_count,
                        expected_layers + output.muxpd_layers_per_slice,
                    )
                )
        calibration_phase = (
            "prefill"
            if new_prefill_ids
            and not decode_ids
            and expected_layers == self._decoder_layer_count
            else "decode"
            if cohort is None and decode_ids
            else None
        )
        self._decoupled_step = _Step(
            output,
            _output_for_update(output, decode_ids),
            output.muxpd_prefill_cohort_id,
            started,
            expected_layers,
            calibration_phase,
            deferred_decode_ids,
            self._decode_shape(
                [self._guard_live_decodes[rid] for rid in sorted(decode_ids)]
            )
            if self.muxpd_policy.config.decode_budget_guard and decode_ids
            else None,
        )
        if self._muxpd_trace is not None:
            self._trace_decoupled(
                "decoupled_schedule",
                cohort_id=output.muxpd_prefill_cohort_id,
                new_prefill_ids=sorted(new_prefill_ids),
                pending_prefill_ids=sorted(output.muxpd_pending_prefill_ids),
                cancelled_prefill_ids=sorted(output.muxpd_cancel_prefill_request_ids),
                decode_ids=sorted(decode_ids),
                finish_prefill=output.muxpd_finish_prefill,
                pause_prefill=paused,
                layers_per_slice=output.muxpd_layers_per_slice,
                layers_completed_before=output.muxpd_prefill_layers_completed_before,
                layers_total=output.muxpd_prefill_layers_total,
                compile_full_prefill=self.muxpd_policy.config.compile_full_prefill,
                quota=quota_info,
                admission=self._prefill_first_admission,
            )
        return output

    def _adjust_phase_selection(self, prefills, decodes, snapshots, now):
        """Return an urgent full P before a separate, guaranteed decode turn.

        Excluding D here is essential: Scheduler.schedule has not yet advanced
        any native token counters or allocated this iteration's KV blocks.
        """
        config = self.muxpd_policy.config
        if not (config.prefill_first or config.decode_budget_guard):
            return prefills, decodes
        all_decodes = [r for r in snapshots if request_phase(r) == "decode"][
            : min(self.max_num_running_reqs, self.max_num_scheduled_tokens)
        ]
        if config.decode_budget_guard:
            self._guard_live_decodes = {
                r.request_id: r.context_length for r in all_decodes
            }
        if self._prefill_first_decode_debt:
            # The base selector may have chosen P-only for a prompt that fills
            # the token budget. Debt still takes precedence over that fallback.
            if all_decodes:
                self._prefill_first_admission["reason"] = "repay_decode"
                return [], all_decodes
            return prefills, decodes
        if not prefills:
            return prefills, decodes
        if not decodes:
            # The base selector already chose P-only because a complete prompt
            # cannot coexist with D within the token budget. Such a turn must
            # incur the same debt, even if P's first token finishes the request
            # and immediately frees a slot for the next queued long prompt.
            if all_decodes:
                self._prefill_first_admission.update(
                    reason="full_budget_prefill_first",
                    deferred_decode_ids=[request.request_id for request in all_decodes],
                )
            return prefills, decodes
        if not config.prefill_first:
            return prefills, decodes
        tokens = sum(remaining_tokens(request) for request in prefills)
        base_quota, work_quota = self._base_layer_quota(tokens)
        self._prefill_first_admission["preview"] = {"work_quota": work_quota}
        if config.decode_budget_guard:
            _, guard = self._decode_budget_quota(
                tuple(sorted(remaining_tokens(r) for r in prefills)),
                self._decoder_layer_count,
                self._decoder_layer_count,
                min(self._arrival_monotonic[r.request_id] for r in prefills),
                now,
            )
            self._prefill_first_admission["preview"]["decode_budget"] = guard
            if guard["cap_layers"] < self._decoder_layer_count:
                self._prefill_first_admission["reason"] = "decode_budget_retain_slicing"
                return prefills, decodes
        if (
            self.muxpd_policy.config.prefill_layer_token_budget
            and base_quota >= self._decoder_layer_count
        ):
            self._prefill_first_admission.update(
                reason="token_budget_prefill_first",
                deferred_decode_ids=[request.request_id for request in decodes],
                preview={"work_quota": work_quota},
            )
            return prefills, []
        if not self._prefill_latency or self._decode_latency_ms is None:
            self._prefill_first_admission["reason"] = "uncalibrated"
            return prefills, decodes
        quota, estimate = self._deadline_layer_quota(
            tokens,
            0,
            min(self._arrival_monotonic[request.request_id] for request in prefills),
            now,
        )
        if quota != self._decoder_layer_count:
            self._prefill_first_admission["reason"] = "retain_layer_slicing"
            return prefills, decodes
        self._prefill_first_admission.update(
            reason="deadline_prefill_first",
            deferred_decode_ids=[request.request_id for request in decodes],
            preview=estimate,
        )
        return prefills, []

    def _estimated_prefill_ms(self, tokens: int) -> float | None:
        if not self._prefill_latency:
            return None
        # Learn launch overhead instead of assuming a constant cost per token.
        # Prefer a measured cohort at least as large; extrapolate conservatively
        # only above the largest measured token count. No NVIDIA constants.
        measured = sorted(self._prefill_latency.values(), key=lambda sample: sample[1])
        duration, measured_tokens = next(
            (sample for sample in measured if sample[1] >= tokens), measured[-1]
        )
        return duration * max(1.0, tokens / measured_tokens)

    def _apply_layer_quota(self, output, now: float) -> dict:
        info = self._apply_desired_layer_quota(output, now)
        if not self.muxpd_policy.config.decode_budget_guard:
            return info
        cohort = self._prefill_cohort
        if cohort is None:
            info["decode_budget"] = None
            return info
        remaining = self._decoder_layer_count - cohort.completed_layers
        quota, guard = self._decode_budget_quota(
            tuple(sorted(cohort.original_output.num_scheduled_tokens.values())),
            remaining,
            output.muxpd_layers_per_slice,
            cohort.arrival_monotonic,
            now,
        )
        guard["desired_reason"] = info["reason"]
        guard["selected_decode_ids"] = sorted(
            set(output.num_scheduled_tokens) - set(output.muxpd_prefill_request_ids)
        )
        guard["control_step"] = bool(
            output.muxpd_pause_prefill
            or set(output.muxpd_cancel_prefill_request_ids) == cohort.pending_ids
        )
        info["decode_budget"] = guard
        info["reason"] = "decode_budget_quota"
        output.muxpd_layers_per_slice = quota
        # No-selected-D is not no-live-D: an oversized complete prompt can use
        # the entire token admission budget while existing D waits for one turn.
        output.muxpd_finish_prefill = quota == remaining and not guard["control_step"]
        return info

    @staticmethod
    def _decode_shape(contexts: list[int]) -> tuple[int, int, int] | None:
        if not contexts:
            return None
        return (
            len(contexts),
            (sum(contexts) - 1).bit_length(),
            (max(contexts) - 1).bit_length(),
        )

    def _decode_budget_quota(self, p_shape, remaining, desired, arrival, now):
        """Cap P work using local observations; never promise a hard SLO.

        First-layer startup and continuation costs have distinct observations.
        Prefer completed lane spans; retain a separate conservative host-step
        fallback that includes D twice. Neither is a cross-rank worst-case bound.
        Compiled whole-P timings cannot train the raw layer estimator. Unknown
        shapes and infeasible budgets still advance one P layer.
        """
        config = self.muxpd_policy.config
        contexts = self._guard_live_decodes
        d_shape = self._decode_shape(list(contexts.values()))
        first_slice = remaining == self._decoder_layer_count
        first_sample, first_source = self._raw_slice_estimate(p_shape, True)
        p_sample, p_source = self._raw_slice_estimate(p_shape, False)
        d_sample = self._guard_cache_read(self._guard_decode_cost, d_shape)
        ages = {
            rid: max(0.0, (now - self._decode_completed_at[rid]) * 1000)
            for rid in contexts
            if rid in self._decode_completed_at
        }
        p_ms = config.estimate_safety_factor * p_sample[0] if p_sample else None
        first_ms = (
            config.estimate_safety_factor * first_sample[0] if first_sample else None
        )
        d_ms = config.estimate_safety_factor * d_sample[0] if d_sample else None
        cold = []
        if not p_sample:
            cold.append("unmeasured_raw_continuation_shape")
        if first_slice and not first_sample:
            cold.append("unmeasured_raw_first_layer_shape")
        if not d_sample:
            cold.append("unmeasured_decode_batch_context_bucket")
        if set(ages) != set(contexts):
            cold.append("unknown_decode_retirement_time")
        age = max(ages.values(), default=0.0)
        slack = config.tpot_slo_ms - age - d_ms if d_ms is not None else None
        calibrated = not cold
        if not contexts:
            mode, cap, quota = "no_live_decode", remaining, remaining
        elif not calibrated:
            mode, cap, quota = "cold_start", 1, 1
        else:
            mode = "measured_slice_budget"
            cap = min(
                remaining,
                max(1, 1 + math.floor((slack - first_ms) / p_ms))
                if first_slice
                else max(1, math.floor(max(0.0, slack) / p_ms)),
            )
            quota = min(remaining, desired, cap)
        cohort_age = max(0.0, (now - arrival) * 1000)
        projected = (
            (first_ms + (remaining - 1) * p_ms if first_slice else remaining * p_ms)
            + math.ceil(remaining / quota) * d_ms
            if calibrated
            else None
        )
        ttft_feasible = (
            cohort_age + projected <= config.ttft_slo_ms
            if projected is not None
            else False
            if cohort_age >= config.ttft_slo_ms
            else None
        )
        return quota, {
            "version": 1,
            "mode": mode,
            "calibrated": calibrated,
            "cold_reasons": cold,
            "estimate_source": "observed_max_raw_slice_spans_plus_solo_D",
            "estimate_scope": (
                "lane spans exclude cross-rank residual; "
                "host fallback includes D twice; not a hard SLO bound"
            ),
            "active_decode_ids": sorted(contexts),
            "active_decode_context_tokens": contexts.copy(),
            "decode_age_ms": ages,
            "decode_shape": list(d_shape) if d_shape else None,
            "prefill_shape_tokens": list(p_shape),
            "first_slice": first_slice,
            "first_layer_samples": first_sample[1] if first_sample else 0,
            "first_layer_estimate_source": first_source,
            "prefill_samples": p_sample[1] if p_sample else 0,
            "prefill_estimate_source": p_source,
            "decode_samples": d_sample[1] if d_sample else 0,
            "estimated_raw_layer_ms": p_ms,
            "estimated_first_layer_ms": first_ms,
            "estimated_solo_decode_ms": d_ms,
            "decode_budget_ms": config.tpot_slo_ms,
            "decode_slack_ms": slack,
            "desired_layers_per_slice": desired,
            "cap_layers": cap,
            "selected_layers_per_slice": quota,
            "minimum_progress_exceeds_budget": bool(
                contexts and calibrated and slack < (first_ms if first_slice else p_ms)
            ),
            "cohort_age_ms": cohort_age,
            "ttft_budget_ms": config.ttft_slo_ms,
            "estimated_remaining_ttft_ms": projected,
            "ttft_feasible": ttft_feasible,
            "estimate_cache": self._guard_cache_stats(),
        }

    @staticmethod
    def _guard_cache_read(cache, key):
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    def _guard_cache_store(self, cache, key, value, phase):
        if key not in cache and len(cache) >= _GUARD_ESTIMATE_CACHE_LIMIT:
            cache.popitem(last=False)
            self._guard_cache_evictions[phase] += 1
        cache[key] = value
        cache.move_to_end(key)

    def _guard_cache_stats(self):
        return {
            "entry_limit_per_mapping": _GUARD_ESTIMATE_CACHE_LIMIT,
            "entries": {
                "prefill": len(self._guard_prefill_cost),
                "decode": len(self._guard_decode_cost),
            },
            "evictions": self._guard_cache_evictions.copy(),
        }

    def _raw_slice_estimate(self, shape, first):
        samples = (
            self._guard_cache_read(
                self._guard_prefill_cost, (shape, first, 1 if first else 0)
            )
            or {}
        )
        # Keep the two estimands separate. A completed lane measurement takes
        # precedence; the first cold host proxy must not poison all later slices.
        for source in ("completed_lane_span", "completed_host_step_proxy"):
            if source in samples:
                return samples[source], source
        return None, None

    def _apply_desired_layer_quota(self, output, now: float) -> dict:
        config = self.muxpd_policy.config
        cohort = self._prefill_cohort
        calibrated = bool(self._prefill_latency) and self._decode_latency_ms is not None
        info = {
            "enabled": config.adaptive_layers,
            "calibrated": calibrated,
            "latency_samples": self._latency_samples.copy(),
            "reason": "fixed_quota" if not config.adaptive_layers else "uncalibrated",
            "estimate_source": "completed_pure_phase_host_round_trip_ewma",
            "estimate_scope": "deadline heuristic; not device overlap or SLO guarantee",
            "work_quota": None,
        }
        if not cohort:
            return info
        base_quota, work_quota = self._base_layer_quota(
            cohort.original_output.total_num_scheduled_tokens
        )
        info["work_quota"] = work_quota
        remaining = self._decoder_layer_count - cohort.completed_layers
        cancelled = set(output.muxpd_cancel_prefill_request_ids)
        control_step = output.muxpd_pause_prefill or cancelled == cohort.pending_ids
        if config.prefill_layer_token_budget:
            output.muxpd_layers_per_slice = min(remaining, base_quota)
            if not control_step and output.muxpd_layers_per_slice == remaining:
                output.muxpd_finish_prefill = True
            info["reason"] = "control_step" if control_step else "token_budget_quota"
        if not config.adaptive_layers or not calibrated:
            return info
        quota, estimate = self._deadline_layer_quota(
            cohort.original_output.total_num_scheduled_tokens,
            cohort.completed_layers,
            cohort.arrival_monotonic,
            now,
        )
        if control_step:
            info["reason"] = "control_step"
        else:
            output.muxpd_layers_per_slice = quota
            if quota == remaining:
                output.muxpd_finish_prefill = True
            info["reason"] = (
                "deadline_finish"
                if estimate["allowed_decode_rounds"] == 1
                else "deadline_quota"
            )
        info.update(estimate)
        return info

    def _base_layer_quota(self, tokens: int) -> tuple[int, dict]:
        """Use the original cohort's complete P work, never its remaining layers.

        The budget measures layer-token work, not KV/token admission capacity.
        Integer division rounds down. The configured layer floor may exceed the
        budget, so this heuristic is neither a hard work cap nor a timing bound.
        Recomputed P includes already-generated tokens that must be replayed.
        """
        config = self.muxpd_policy.config
        budget = config.prefill_layer_token_budget
        quota = config.layers_per_slice
        if budget:
            quota = max(quota, min(self._decoder_layer_count, budget // tokens))
        return quota, {
            "source": (
                "prefill_layer_token_budget"
                if budget
                else "configured_layers_per_slice"
            ),
            "prefill_layer_token_budget": budget,
            "cohort_full_prefill_tokens": tokens,
            "configured_layers_per_slice": config.layers_per_slice,
            "base_layers_per_slice": quota,
        }

    def _deadline_layer_quota(self, tokens, completed_layers, arrival_monotonic, now):
        """Pure deadline calculation shared by preview and allocated cohorts."""
        config = self.muxpd_policy.config
        remaining = self._decoder_layer_count - completed_layers
        full_ms = self._estimated_prefill_ms(tokens)
        assert full_ms is not None and self._decode_latency_ms is not None
        p_ms = (
            config.estimate_safety_factor
            * full_ms
            * (remaining / self._decoder_layer_count)
        )
        d_ms = config.estimate_safety_factor * self._decode_latency_ms
        age_ms = max(0.0, (now - arrival_monotonic) * 1000)
        slack = config.ttft_slo_ms - age_ms - p_ms
        rounds = max(1, math.floor(slack / d_ms))
        base_quota, work_quota = self._base_layer_quota(tokens)
        quota = min(remaining, max(base_quota, math.ceil(remaining / rounds)))
        return quota, {
            "cohort_age_ms": age_ms,
            "estimated_remaining_prefill_ms": p_ms,
            "estimated_decode_ms": d_ms,
            "allowed_decode_rounds": rounds,
            "work_quota": work_quota,
            "decode_slo_feasible_without_prefill": (
                self._decode_latency_ms <= config.tpot_slo_ms
            ),
        }

    def _observe_latency(self, phase: str, duration_ms: float, tokens: int) -> None:
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            return
        alpha = self.muxpd_policy.config.ewma_alpha
        if phase == "prefill":
            bucket = (tokens - 1).bit_length()
            old, measured_tokens = self._prefill_latency.get(
                bucket, (duration_ms, tokens)
            )
            anchor = max(measured_tokens, tokens)
            old *= anchor / measured_tokens
            duration_ms *= anchor / tokens
            sample = min(max(duration_ms, old / 4), old * 4)
            self._prefill_latency[bucket] = (
                (1 - alpha) * old + alpha * sample,
                anchor,
            )
        else:
            old = self._decode_latency_ms
            if old is None:
                self._decode_latency_ms = duration_ms
            else:
                sample = min(max(duration_ms, old / 4), old * 4)
                self._decode_latency_ms = (1 - alpha) * old + alpha * sample
        self._latency_samples[phase] += 1

    @staticmethod
    def _receipt_ids(model_output, name: str) -> set[str]:
        value = getattr(model_output, name, None)
        if not isinstance(value, (set, frozenset, list, tuple)) or any(
            not isinstance(rid, str) for rid in value
        ):
            raise RuntimeError(f"Missing or invalid worker receipt: {name}")
        return set(value)

    def _validate_receipt(self, step: _Step, model_output):
        if not hasattr(model_output, "muxpd_prefill_cohort_id") or (
            model_output.muxpd_prefill_cohort_id != step.cohort_id
        ):
            raise RuntimeError("Unknown/reordered prefill cohort receipt")
        completed = self._receipt_ids(model_output, "muxpd_completed_prefill_ids")
        cancelled = self._receipt_ids(model_output, "muxpd_cancelled_prefill_ids")
        cohort = self._prefill_cohort
        pending = cohort.pending_ids if cohort else set()
        if completed & cancelled or not (completed | cancelled) <= pending:
            raise RuntimeError("Prefill receipt contains duplicate/unknown completion")
        if not cancelled <= self._cancelled_prefill_ids():
            raise RuntimeError("Worker cancelled a live prefill request")
        alive = pending & self.requests.keys()
        if completed & alive and (completed & alive) != alive:
            raise RuntimeError("Worker must complete the live prefill cohort together")
        if step.output.muxpd_finish_prefill and not alive <= completed:
            raise RuntimeError("Worker did not finish an unopposed prefill")
        if step.output.muxpd_pause_prefill and completed & alive:
            raise RuntimeError("Worker advanced a paused prefill")
        if pending and set(step.output.muxpd_cancel_prefill_request_ids) == pending:
            if completed | cancelled != pending:
                raise RuntimeError(
                    "Worker did not acknowledge full cohort cancellation"
                )
        expected_samples = set(step.decode_output.num_scheduled_tokens) | completed
        if (
            len(model_output.req_ids) != len(set(model_output.req_ids))
            or set(model_output.req_ids) != expected_samples
            or model_output.req_id_to_index
            != {rid: index for index, rid in enumerate(model_output.req_ids)}
            or len(model_output.sampled_token_ids) != len(model_output.req_ids)
            or any(len(tokens) != 1 for tokens in model_output.sampled_token_ids)
        ):
            raise RuntimeError(
                "Worker samples do not match decode/completed prefill IDs"
            )
        if any(
            getattr(model_output, name, None) is not None
            for name in (
                "routed_experts",
                "kv_connector_output",
                "ec_connector_output",
            )
        ):
            raise RuntimeError("Unsupported metadata in decoupled worker output")
        progress = getattr(model_output, "muxpd_prefill_layers_completed", None)
        total = getattr(model_output, "muxpd_prefill_layers_total", None)
        expected_total = self._decoder_layer_count if cohort else 0
        if (
            type(progress) is not int
            or type(total) is not int
            or total != expected_total
            or progress != step.expected_layers
        ):
            raise RuntimeError("Invalid prefill layer progress receipt")
        if completed and progress != total:
            raise RuntimeError("Prefill completed before all decoder layers")
        if cohort and progress == total and completed | cancelled != pending:
            raise RuntimeError("Final prefill layer requires whole cohort retirement")
        source = getattr(model_output, "muxpd_prefill_progress_source", None)
        if type(source) is not str or source not in (
            "none",
            "layer_hooks",
            "full_forward_return",
        ):
            raise RuntimeError("Missing or invalid prefill progress source")
        if (source == "none") != (cohort is None):
            raise RuntimeError(
                "Prefill progress source does not match cohort ownership"
            )
        if source == "full_forward_return" and not (
            self.muxpd_policy.config.compile_full_prefill
            and set(step.output.muxpd_prefill_request_ids) == pending
            and step.output.muxpd_prefill_layers_completed_before == 0
            and not step.output.muxpd_pause_prefill
            and not step.output.muxpd_cancel_prefill_request_ids
            and step.expected_layers == total == progress
            and completed == pending
            and not cancelled
        ):
            raise RuntimeError(
                "Full-forward progress requires a fresh, complete prefill transaction"
            )
        return completed, cancelled, progress, source

    @staticmethod
    def _merge_outputs(first, second):
        latest_stats = next(
            (
                row.scheduler_stats
                for row in second.values()
                if row.scheduler_stats is not None
            ),
            None,
        )
        if latest_stats is not None:
            for row in first.values():
                row.scheduler_stats = None
        for client, row in second.items():
            if client not in first:
                first[client] = row
                continue
            target = first[client]
            target.outputs.extend(row.outputs)
            if row.finished_requests:
                target.finished_requests = (
                    set(target.finished_requests or ()) | row.finished_requests
                )
            if row.scheduler_stats is not None:
                target.scheduler_stats = row.scheduler_stats
            target.timestamp = max(target.timestamp, row.timestamp)
        return first

    def update_from_output(self, scheduler_output, model_runner_output):
        step = self._decoupled_step
        if step is None or step.output is not scheduler_output:
            raise RuntimeError("Unknown/reordered decoupled worker transaction")
        completed, cancelled, progress, source = self._validate_receipt(
            step, model_runner_output
        )
        slice_timing = self._validate_slice_timing(step, model_runner_output, source)
        completed_at = time.monotonic()
        duration_ms = (completed_at - step.started_at) * 1000
        cohort = self._prefill_cohort
        calibration_ids = (
            set(cohort.original_output.num_scheduled_tokens)
            if step.calibration_phase == "prefill" and cohort
            else set(step.decode_output.num_scheduled_tokens)
        )
        valid_latency_sample = bool(
            step.calibration_phase
            and not cancelled
            and not step.output.muxpd_cancel_prefill_request_ids
            and calibration_ids <= self.requests.keys()
        )
        guard_sample = self._observe_guard_latency(
            step, cohort, progress, source, duration_ms, cancelled, slice_timing
        )
        # Validation precedes every mutation. An ACK proves the worker can no
        # longer access this member's buffers, including late-aborted completions.
        for rid in completed | cancelled:
            blocks = self._held_prefill_blocks.pop(rid, ())
            if blocks:
                self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
        if cohort is not None:
            cohort.completed_layers = progress
            cohort.pending_ids.difference_update(completed | cancelled)
        # Also clean worker metadata after a completion raced with a host abort.
        # Repeating an acknowledged cancellation's cleanup is idempotent.
        self.finished_req_ids.update((completed | cancelled) - self.requests.keys())
        result = Scheduler.update_from_output(
            self,
            step.decode_output,
            model_runner_output,
        )
        if completed:
            assert cohort is not None
            prefill_output = _output_for_update(cohort.original_output, completed)
            result = self._merge_outputs(
                result,
                Scheduler.update_from_output(
                    self,
                    prefill_output,
                    model_runner_output,
                ),
            )
        if cohort is not None and not cohort.pending_ids:
            self._prefill_cohort = None
            self.muxpd_stats["retired_prefill_cohorts"] += 1
        self._prefill_first_decode_debt.difference_update(
            step.decode_output.num_scheduled_tokens
        )
        self._prefill_first_decode_debt.update(
            # Allocation may have preempted a decode, or its ID may have been
            # cancelled and reused while P ran. Only still-running D owes a turn.
            step.deferred_decode_ids & {request.request_id for request in self.running}
        )
        self._decoupled_step = None
        self.muxpd_stats["completed_worker_steps"] += 1
        self.muxpd_stats["completed_prefill_requests"] += len(completed)
        self.muxpd_stats["cancelled_prefill_requests"] += len(cancelled)
        if step.decode_output.total_num_scheduled_tokens:
            self.muxpd_policy.last_decode_at = completed_at
        if self.muxpd_policy.config.decode_budget_guard:
            for rid in set(step.decode_output.num_scheduled_tokens) | completed:
                if rid in self.requests:
                    self._decode_completed_at[rid] = completed_at
        if valid_latency_sample:
            self._observe_latency(
                step.calibration_phase,
                duration_ms,
                cohort.original_output.total_num_scheduled_tokens
                if step.calibration_phase == "prefill"
                else step.decode_output.total_num_scheduled_tokens,
            )
        if self._muxpd_trace is not None:
            self._trace_decoupled(
                "decoupled_complete",
                cohort_id=step.cohort_id,
                completed_prefill_ids=sorted(completed),
                cancelled_prefill_ids=sorted(cancelled),
                decode_ids=list(step.decode_output.num_scheduled_tokens),
                duration_ms=duration_ms,
                layers_completed=progress,
                layers_total=model_runner_output.muxpd_prefill_layers_total,
                progress_source=source,
                latency_sample_phase=step.calibration_phase
                if valid_latency_sample
                else None,
                decode_debt_ids=sorted(self._prefill_first_decode_debt),
                **(
                    {
                        "decode_budget_sample": guard_sample,
                        "prefill_slice_timing": slice_timing,
                        "estimate_cache": self._guard_cache_stats(),
                    }
                    if self.muxpd_policy.config.decode_budget_guard
                    else {}
                ),
            )
        return result

    def _validate_slice_timing(self, step, model_output, source):
        """Validate optional lane timing before estimator or lifecycle mutation."""
        timing = getattr(model_output, "muxpd_prefill_slice_timing", None)
        if timing is None:
            return None
        cohort = self._prefill_cohort
        if (
            not self.muxpd_policy.config.decode_budget_guard
            or cohort is None
            or step.output.muxpd_pause_prefill
            or step.output.muxpd_cancel_prefill_request_ids
            or not isinstance(timing, dict)
        ):
            raise RuntimeError("Unexpected prefill slice timing receipt")
        delta = step.expected_layers - cohort.completed_layers
        for name, expected in (
            ("version", 1),
            ("layer_delta", delta),
            ("prefill_tokens", cohort.original_output.total_num_scheduled_tokens),
        ):
            if type(timing.get(name)) is not int or timing[name] != expected:
                raise RuntimeError(f"Invalid prefill slice timing {name}")
        expected_first = cohort.completed_layers == 0
        if (
            type(timing.get("first_slice")) is not bool
            or timing["first_slice"] != expected_first
            or timing.get("mode")
            != (
                "compiled_full_forward"
                if source == "full_forward_return"
                else "raw_layers"
            )
            or type(timing.get("device_complete")) is not bool
        ):
            raise RuntimeError("Invalid prefill slice timing execution mode")
        for name in ("host_submit_ms", "device_elapsed_ms"):
            value = timing.get(name)
            if value is None and name == "device_elapsed_ms":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
            ):
                raise RuntimeError(f"Invalid prefill slice timing {name}")
        if timing["device_complete"] != (timing.get("device_elapsed_ms") is not None):
            raise RuntimeError("Prefill slice device timing lacks completion")
        if timing["device_complete"] and not (
            step.decode_output.num_scheduled_tokens
            or step.expected_layers == self._decoder_layer_count
        ):
            raise RuntimeError("Prefill slice has no existing sampling completion")
        return timing.copy()

    def _observe_guard_latency(
        self, step, cohort, progress, source, duration, cancelled, timing
    ):
        if not self.muxpd_policy.config.decode_budget_guard:
            return None
        involved = set(step.decode_output.num_scheduled_tokens)
        if cohort:
            involved.update(cohort.original_output.num_scheduled_tokens)
        if (
            not math.isfinite(duration)
            or duration <= 0
            or cancelled
            or step.output.muxpd_cancel_prefill_request_ids
            or step.output.muxpd_pause_prefill
            or not involved <= self.requests.keys()
        ):
            return {"phase": None, "reason": "invalid_cancelled_or_paused_sample"}
        if cohort is None and step.guard_decode_shape:
            old, count = self._guard_cache_read(
                self._guard_decode_cost, step.guard_decode_shape
            ) or (0.0, 0)
            self._guard_cache_store(
                self._guard_decode_cost,
                step.guard_decode_shape,
                (max(old, duration), count + 1),
                "decode",
            )
            return {"phase": "decode", "shape": list(step.guard_decode_shape)}
        if cohort:
            delta = progress - cohort.completed_layers
            sampled = bool(step.decode_output.num_scheduled_tokens) or (
                progress == self._decoder_layer_count
            )
            if (
                source == "layer_hooks"
                and delta > 0
                and delta < self._decoder_layer_count
                and sampled
            ):
                shape = tuple(
                    sorted(cohort.original_output.num_scheduled_tokens.values())
                )
                first = cohort.completed_layers == 0
                # q>1 first slices remain separate observations. Their entire
                # span must never become the fixed cost of a q=1 first layer.
                key = (shape, first, delta if first else 0)
                sample_source = (
                    "completed_lane_span"
                    if timing is not None and timing["device_complete"]
                    else "completed_host_step_proxy"
                )
                span = (
                    max(timing["host_submit_ms"], timing["device_elapsed_ms"])
                    if sample_source == "completed_lane_span"
                    else duration
                )
                cost = span if first else span / delta
                if cost <= 0:
                    return {"phase": None, "reason": "zero_slice_span"}
                observations = (
                    self._guard_cache_read(self._guard_prefill_cost, key) or {}
                )
                old, count = observations.get(sample_source, (0.0, 0))
                observations[sample_source] = (max(old, cost), count + 1)
                self._guard_cache_store(
                    self._guard_prefill_cost, key, observations, "prefill"
                )
                return {
                    "phase": "raw_prefill_layer",
                    "shape": list(shape),
                    "layer_delta": delta,
                    "first_slice": first,
                    "measurement_source": sample_source,
                    "sample_span_ms": span,
                    "sample_cost_ms": cost,
                    "host_step_ms_per_layer": duration / delta,
                    "host_step_minus_lane_span_ms": (
                        max(0.0, duration - span)
                        if sample_source == "completed_lane_span"
                        else None
                    ),
                }
        return {"phase": None, "reason": "no_completed_raw_slice_or_solo_decode"}

    def _trace_decoupled(self, event: str, **fields) -> None:
        if self._muxpd_trace is not None:
            self._muxpd_trace.write(
                json.dumps(
                    {
                        "event": event,
                        "pid": os.getpid(),
                        "time": time.monotonic(),
                        "step": self.muxpd_stats["steps"],
                        **fields,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
