"""Homogeneous P/D admission with bounded waiting and online cost estimates.

This is the portable scheduling layer inspired by MuxWise. It does not claim
the paper's hardware spatial multiplexing or kernel-level interference model.
The native scheduler remains responsible for all KV allocation and lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from .config import MuxPDConfig

if TYPE_CHECKING:
    from vllm.v1.core.sched.batch_admission import BatchRequest

Phase = Literal["prefill", "decode"]


def request_phase(request: BatchRequest) -> Phase:
    # A preempted request retains generated tokens, but must recompute its
    # ENTIRE context. A one-token prompt is still prefill, not decode.
    if request.state == "waiting" or request.num_output_tokens == 0:
        return "prefill"
    if request.num_computed_tokens < request.context_length - 1:
        return "prefill"
    return "decode"


def remaining_tokens(request: BatchRequest) -> int:
    return max(0, request.context_length - request.num_computed_tokens)


@dataclass(frozen=True)
class Decision:
    phase: Phase | None
    request_ids: tuple[str, ...]
    reason: str
    estimated_prefill_ms: float = 0.0
    estimated_decode_ms: float = 0.0
    decode_slack_ms: float = 0.0
    oldest_prefill_wait_ms: float = 0.0


class MuxPDPolicy:
    """Choose one homogeneous batch; mutations are committed after allocation."""

    def __init__(
        self,
        config: MuxPDConfig | None = None,
        token_budget: int = 8192,
        max_num_seqs: int = 256,
    ) -> None:
        if token_budget <= 0 or max_num_seqs <= 0:
            raise ValueError("MuxPD token and sequence budgets must be positive")
        self.config = config or MuxPDConfig()
        self.token_budget = token_budget
        self.max_num_seqs = max_num_seqs
        self.decode_steps = 0
        self.last_decode_at: float | None = None
        self.prefill_wait_since: dict[str, float] = {}
        self.prefill_scale = 1.0
        self.decode_scale = 1.0
        self._batch_sequence = 0
        self.last_decision = Decision(None, (), "empty")

    @classmethod
    def from_config(cls, config: Mapping[str, Any], vllm_config: Any) -> "MuxPDPolicy":
        scheduler = vllm_config.scheduler_config
        implementation = getattr(scheduler, "scheduler_cls", None)
        name = (
            implementation
            if isinstance(implementation, str) or implementation is None
            else f"{implementation.__module__}.{implementation.__qualname__}"
        )
        if name not in (
            "vllm_muxpd.scheduler.MuxPDScheduler",
            "vllm_muxpd.multiplex_scheduler.LayeredMuxPDScheduler",
            "vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler",
            "vllm_muxpd.async_scheduler.AsyncDecoupledMuxPDScheduler",
        ):
            raise ValueError(
                "MuxPDPolicy requires a paired MuxPD scheduler_cls; "
                "the native synchronous scheduler bypasses admission policies"
            )
        return cls(
            MuxPDConfig.from_mapping(config),
            getattr(scheduler, "max_num_scheduled_tokens", None)
            or scheduler.max_num_batched_tokens,
            scheduler.max_num_seqs,
        )

    def estimate_prefill_ms(self, requests: Sequence[BatchRequest]) -> float:
        if not requests:
            return 0.0
        tokens = sum(remaining_tokens(request) for request in requests)
        return (
            self.config.estimate_safety_factor
            * self.prefill_scale
            * (self.config.prefill_fixed_ms + self.config.prefill_token_ms * tokens)
        )

    def estimate_decode_ms(self, requests: Sequence[BatchRequest]) -> float:
        if not requests:
            return 0.0
        return (
            self.config.estimate_safety_factor
            * self.decode_scale
            * (
                self.config.decode_fixed_ms
                + self.config.decode_context_token_ms
                * sum(request.context_length for request in requests)
            )
        )

    def choose(
        self,
        requests: Sequence[BatchRequest],
        now: float,
        *,
        available_prefill_slots: int | None = None,
        force_decode: bool = False,
    ) -> Decision:
        eligible = [
            r for r in requests if r.num_in_flight_tokens == 0 and remaining_tokens(r)
        ]
        prefills = [r for r in eligible if request_phase(r) == "prefill"]
        decodes = [r for r in eligible if request_phase(r) == "decode"]
        prefill_ids = {r.request_id for r in prefills}
        self.prefill_wait_since = {
            rid: timestamp
            for rid, timestamp in self.prefill_wait_since.items()
            if rid in prefill_ids
        }
        for request in prefills:
            self.prefill_wait_since.setdefault(request.request_id, now)
        slots = (
            self.max_num_seqs
            if available_prefill_slots is None
            else available_prefill_slots
        )
        selected_prefills: list[BatchRequest] = []
        budget = self.config.max_prefill_batch_tokens or self.token_budget
        budget = min(budget, self.token_budget)
        # Preserve snapshot/native traversal order and head-of-line fairness.
        # The soft batching cap never prevents one complete prompt from fitting.
        for request in prefills:
            cost = remaining_tokens(request)
            if cost > self.token_budget:
                raise ValueError(
                    f"MuxPD full prefill {request.request_id!r} exceeds token budget"
                )
            if len(selected_prefills) >= self.config.max_prefill_requests:
                break
            if request.state == "waiting" and slots <= 0:
                continue
            if selected_prefills and cost > budget:
                break
            selected_prefills.append(request)
            budget -= cost
            if request.state == "waiting":
                slots -= 1
        selected_decodes = decodes[: min(self.token_budget, self.max_num_seqs)]
        for request in selected_decodes:
            if remaining_tokens(request) != 1:
                raise RuntimeError("MuxPD decode must schedule exactly one token")
        p_ms = self.estimate_prefill_ms(selected_prefills)
        d_ms = self.estimate_decode_ms(selected_decodes)
        wait_ms = max(
            (
                1000 * max(0, now - self.prefill_wait_since[r.request_id])
                for r in selected_prefills
            ),
            default=0.0,
        )
        gap_ms = (
            0.0
            if self.last_decode_at is None
            else 1000 * max(0, now - self.last_decode_at)
        )
        slack_ms = self.config.tpot_slo_ms - gap_ms - d_ms
        phase: Phase | None
        if not selected_prefills and not selected_decodes:
            phase, reason = None, "no_schedulable_work"
        elif not selected_prefills or force_decode:
            phase, reason = (
                ("decode", "decode_only")
                if selected_decodes
                else (None, "decode_unavailable")
            )
        elif not selected_decodes:
            phase, reason = "prefill", "prefill_only"
        elif self.decode_steps >= self.config.max_decode_steps:
            phase, reason = "prefill", "decode_step_bound"
        elif wait_ms >= self.config.max_prefill_wait_ms:
            phase, reason = "prefill", "prefill_wait_bound"
        elif wait_ms + p_ms >= self.config.ttft_slo_ms:
            phase, reason = "prefill", "ttft_deadline"
        elif self.last_decode_at is not None and p_ms <= slack_ms:
            phase, reason = "prefill", "fits_decode_slack"
        else:
            phase, reason = "decode", "decode_priority"
        selected = (
            selected_prefills
            if phase == "prefill"
            else selected_decodes
            if phase == "decode"
            else []
        )
        self.last_decision = Decision(
            phase,
            tuple(r.request_id for r in selected),
            reason,
            p_ms,
            d_ms,
            slack_ms,
            wait_ms,
        )
        return self.last_decision

    def commit(self, phase: Phase, request_ids: Sequence[str], now: float) -> None:
        if phase == "decode":
            self.decode_steps += 1
            self.last_decode_at = now
        else:
            self.decode_steps = 0
            for rid in request_ids:
                self.prefill_wait_since.pop(rid, None)

    def observe_batch(
        self,
        phase: Phase,
        duration_ms: float,
        tokens: int,
        contexts: int,
    ) -> None:
        if duration_ms <= 0:
            return
        if phase == "prefill":
            base = self.config.prefill_fixed_ms + self.config.prefill_token_ms * tokens
            old = self.prefill_scale
        else:
            base = (
                self.config.decode_fixed_ms
                + self.config.decode_context_token_ms * contexts
            )
            old = self.decode_scale
        # Bound extreme single samples (e.g. warmup) while still adapting.
        sample = min(max(duration_ms / base, old / 4), old * 4)
        new = (1 - self.config.ewma_alpha) * old + self.config.ewma_alpha * sample
        if phase == "prefill":
            self.prefill_scale = new
        else:
            self.decode_scale = new

    def admit_batch(self, context: Any) -> Any:
        from vllm.v1.core.sched.batch_admission import BatchAdmission

        if context.in_flight_batch_ids or context.max_concurrent_batches != 1:
            raise RuntimeError("MuxPD supports one synchronous batch in flight")
        decision = self.choose(context.requests, context.now)
        if not decision.request_ids:
            return None
        self._batch_sequence += 1
        return BatchAdmission(
            f"muxpd-{decision.phase}-{self._batch_sequence}", decision.request_ids
        )
