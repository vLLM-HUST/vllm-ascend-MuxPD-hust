"""Composite scheduling for separate, cooperatively executed P/D model batches.

The host receives one lifecycle transaction. The worker executes each phase with
an independent runner and input batch; prompt tokens are never chunked.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler

from .policy import Decision, remaining_tokens, request_phase
from .scheduler import MuxPDScheduler


class LayeredMuxPDScheduler(MuxPDScheduler):
    """Keep native KV ownership while sending isolated P/D lanes to the worker.

    Both lanes finish before the host retires the transaction. This first
    integration does not let decode take multiple steps during one prefill.
    """

    def __init__(self, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        if self.muxpd_policy.config.execution != "layered":
            raise ValueError("LayeredMuxPDScheduler requires execution='layered'")
        worker = vllm_config.parallel_config.worker_cls
        if worker != "vllm_muxpd.worker.MuxPDNPUWorker":
            raise ValueError("LayeredMuxPDScheduler requires MuxPDNPUWorker")
        self._layered_pending = None

    def schedule(self, throttle_prefills: bool = False):
        if throttle_prefills or self._layered_pending is not None:
            raise RuntimeError("Layered MuxPD requires one synchronous transaction")
        now = time.monotonic()
        snapshots = self._muxpd_snapshots()
        if self._pause_state == PauseState.PAUSED_ALL:
            snapshots = ()
        elif self._pause_state != PauseState.UNPAUSED:
            snapshots = tuple(r for r in snapshots if r.state == "running")
        snapshots = tuple(
            r for r in snapshots
            if r.num_in_flight_tokens == 0 and remaining_tokens(r) > 0
        )
        p_candidates = tuple(r for r in snapshots if request_phase(r) == "prefill")
        d_candidates = tuple(r for r in snapshots if request_phase(r) == "decode")
        slots = max(0, self.max_num_running_reqs - len(self.running))
        policy = self.muxpd_policy
        primary = policy.choose(snapshots, now, available_prefill_slots=slots)
        budget = self.max_num_scheduled_tokens
        # Reserve every active decode slot, then admit only complete prefills.
        # If the oldest full prompt cannot coexist, bounded waiting permits a
        # P-only transaction rather than silently cutting that prompt.
        decodes = list(d_candidates[: min(self.max_num_running_reqs, budget)])
        for request in decodes:
            if remaining_tokens(request) != 1:
                raise RuntimeError("Layered MuxPD decode requires one token")
        budget -= len(decodes)
        prefills = []
        soft_budget = policy.config.max_prefill_batch_tokens or budget
        for request in p_candidates:
            if request.state == "waiting" and slots == 0:
                continue
            if len(prefills) >= policy.config.max_prefill_requests:
                break
            cost = remaining_tokens(request)
            if cost > budget or (prefills and cost > soft_budget):
                break
            prefills.append(request)
            budget -= cost
            soft_budget -= cost
            if request.state == "waiting":
                slots -= 1
        if not prefills and primary.phase == "prefill":
            ids = set(primary.request_ids)
            prefills = [r for r in p_candidates if r.request_id in ids]
            decodes = []
        selected = tuple(r.request_id for r in (*prefills, *decodes))
        output = self._schedule_layered_selection(selected, snapshots)
        if selected and not output.total_num_scheduled_tokens:
            # A P-only turn may need KV blocks still held by live decodes.
            # Progress those decodes instead of repeatedly returning the same
            # empty P-only transaction. Keep lifecycle side effects from both
            # native scheduling passes visible to the worker.
            retry_snapshots = self._muxpd_snapshots()
            retry = policy.choose(
                retry_snapshots, now, available_prefill_slots=0,
                force_decode=True,
            )
            if retry.request_ids:
                first_output = output
                snapshots = retry_snapshots
                output = self._schedule_layered_selection(
                    retry.request_ids, snapshots,
                )
                output.finished_req_ids.update(first_output.finished_req_ids)
                output.preempted_req_ids.update(first_output.preempted_req_ids)
                self.muxpd_stats["prefill_capacity_deferrals"] += 1
            if not output.total_num_scheduled_tokens and not self.running:
                raise RuntimeError(
                    "Layered MuxPD cannot fit a complete prefill into available "
                    "KV cache; increase KV capacity or reduce input length"
                )
        by_id = {r.request_id: r for r in snapshots}
        p_ids = frozenset(
            rid
            for rid in output.num_scheduled_tokens
            if request_phase(by_id[rid]) == "prefill"
        )
        d_ids = frozenset(output.num_scheduled_tokens) - p_ids
        output.muxpd_prefill_request_ids = p_ids
        output.muxpd_layers_per_slice = policy.config.layers_per_slice
        phase = (
            "multiplex" if p_ids and d_ids else "prefill" if p_ids
            else "decode" if d_ids else None
        )
        output.muxpd_phase = phase
        decision = Decision(
            phase, tuple(output.num_scheduled_tokens), "separate_model_batches"
        )
        self.muxpd_stats["steps"] += 1
        self.muxpd_stats["preemptions"] += len(output.preempted_req_ids)
        if output.total_num_scheduled_tokens:
            self._layered_pending = (output, now, by_id)
            self.muxpd_stats[f"{phase}_transactions"] += 1
            if d_ids:
                policy.commit("decode", tuple(d_ids), now)
            if p_ids:
                policy.commit("prefill", tuple(p_ids), now)
        else:
            self.muxpd_stats["empty_steps"] += 1
        self._record_muxpd_step(output, decision, snapshots, now)
        return output

    def _schedule_layered_selection(self, selected, snapshots):
        """Apply native lifecycle changes and verify full per-request costs."""
        if self._batch_eligible_request_ids is not None:
            raise RuntimeError("Nested MuxPD admission is unsupported")
        self._batch_eligible_request_ids = frozenset(selected)
        try:
            output = Scheduler.schedule(self, throttle_prefills=False)
        finally:
            self._batch_eligible_request_ids = None
        by_id = {r.request_id: r for r in snapshots}
        for rid, count in output.num_scheduled_tokens.items():
            if rid not in selected or count != remaining_tokens(by_id[rid]):
                raise RuntimeError("Layered MuxPD refused a partial/unknown request")
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        pending = self._layered_pending
        if (
            pending is not None and pending[0] is not scheduler_output
        ) or (pending is None and scheduler_output.total_num_scheduled_tokens):
            raise RuntimeError("Unknown/reordered layered MuxPD transaction")
        result = Scheduler.update_from_output(
            self, scheduler_output, model_runner_output
        )
        if pending is not None:
            now = time.monotonic()
            duration_ms = (now - pending[1]) * 1000
            self._layered_pending = None
            self.muxpd_stats["completed_transactions"] += 1
            if scheduler_output.muxpd_phase in ("decode", "multiplex"):
                self.muxpd_policy.last_decode_at = now
            if self._muxpd_trace is not None:
                self._muxpd_trace.write(
                    json.dumps(
                        {
                            "event": "complete",
                            "time": now,
                            "pid": os.getpid(),
                            "step": self.muxpd_stats["steps"],
                            "phase": scheduler_output.muxpd_phase,
                            "duration_ms": duration_ms,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        return result
