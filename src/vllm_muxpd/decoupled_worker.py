"""Persistent prefill forwards with independently completed decode iterations."""

from __future__ import annotations

import sys
from dataclasses import replace
from functools import partial
from typing import Any

import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_muxpd.lanes import LayerInterleaver
from vllm_muxpd.slice_timing import PrefillSliceTimer
from vllm_muxpd.worker import (
    MuxPDNPUWorker,
    _merge_outputs,
    _select_requests,
)


class DecoupledMuxPDNPUWorker(MuxPDNPUWorker):
    """Keep one complete-prompt cohort suspended between engine iterations.

    Decode can finish while a prefill Python stack remains suspended. The
    default backend fences complete decoder slices; stream_mode='overlap'
    uses independent streams and gates only communication dependencies.
    """

    execution_mode = "decoupled"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._prefill_task: LayerInterleaver | None = None
        self._prefill_cohort_id: int | None = None
        self._prefill_order: list[str] = []
        self._prefill_cancelled: set[str] = set()
        self._prefill_compiled = False
        self._prefill_full_forward_return = False
        self._prefill_num_tokens = 0
        self._iteration_prefill_timer: PrefillSliceTimer | None = None
        self._awaiting_sample = False
        self._iteration_cohort_id: int | None = None
        self._iteration_prefill_layers_completed = 0
        self._iteration_prefill_layers_total = 0
        self._iteration_prefill_progress_source = "none"
        self._cancelled_ack: set[str] = set()
        self._decode_only_fast_path = False
        super().__init__(*args, **kwargs)

    def _clear_prefill(self) -> None:
        self._prefill_task = None
        self._prefill_cohort_id = None
        self._prefill_order = []
        self._prefill_cancelled = set()
        self._prefill_compiled = False
        self._prefill_full_forward_return = False
        self._prefill_num_tokens = 0

    def _snapshot_prefill_progress(self) -> None:
        if self._prefill_task is not None:
            task = self._prefill_task.tasks["prefill"]
            complete_return = task.done and getattr(
                self, "_prefill_full_forward_return", False
            )
            self._iteration_prefill_layers_completed = (
                self._decoder_layer_count if complete_return else task.completed_layers
            )
            self._iteration_prefill_layers_total = self._decoder_layer_count
            self._iteration_prefill_progress_source = (
                "group_forward_return"
                if getattr(self, "_prefill_groups_enabled", False)
                else "full_forward_return"
                if complete_return
                else "layer_hooks"
            )

    def _execute_compiled_prefill(self, output: SchedulerOutput, controller: Any):
        """A fresh full forward may use FX, but never D's ACL graph."""
        task = self._prefill_task.tasks["prefill"]
        if task.completed_layers != 0 or self._stream_mode != "fenced":
            raise RuntimeError("Compiled prefill requires a fresh fenced forward")
        with controller.full_prefill_compile_context():
            result = self.model_runner.execute_model(output)
        # A compiled FX call normally elides Python post-hooks. If native falls
        # back to raw Python, require every layer hook rather than masking a
        # partial forward with a fabricated full layer count.
        if task.completed_layers not in (0, self._decoder_layer_count):
            raise RuntimeError("Compiled prefill returned with incomplete layer hooks")
        if result is not None or self.model_runner.execute_model_state is None:
            raise RuntimeError(
                "Compiled prefill requires a complete deferred native sampler"
            )
        self._prefill_full_forward_return = True
        return result

    def _run_lane(
        self, interleaver: LayerInterleaver, lane: str, layer_budget: int | None
    ) -> None:
        self._interleaver = interleaver
        try:
            if lane == "prefill" and getattr(self, "_prefill_compiled", False):
                # This path was admitted with a full quota. Even a raw Python
                # fallback must not suspend inside the compilation context.
                layer_budget = None
            if (
                lane == "prefill"
                and getattr(self, "_prefill_slice_timing_enabled", False)
                and not self._prefill_cancelled
            ):
                if self._stream_mode != "fenced":
                    raise RuntimeError("MuxPD slice timing requires fenced execution")
                timer = PrefillSliceTimer(
                    torch.npu.Event(enable_timing=True),
                    torch.npu.Event(enable_timing=True),
                    self._streams["prefill"],
                    start_layer=interleaver.tasks[lane].completed_layers,
                    prefill_tokens=self._prefill_num_tokens,
                    compiled=getattr(self, "_prefill_compiled", False),
                )
                self._iteration_prefill_timer = timer
                try:
                    interleaver.step(
                        lane,
                        sys.maxsize if layer_budget is None else layer_budget,
                        observer=timer,
                    )
                except BaseException:
                    # Match run()'s full-forward unwind behavior. Partial steps
                    # keep their existing fail-closed engine-error semantics.
                    if layer_budget is None:
                        interleaver.close()
                    raise
                return
            if layer_budget is None:
                interleaver.run((lane,), sys.maxsize)
            else:
                interleaver.step(lane, layer_budget)
        finally:
            self._interleaver = None

    def _execute_decode_only(self, output: SchedulerOutput, decodes: set[str]) -> None:
        """Run a complete D forward in its own lane without a resumable stack."""
        cached = output.scheduled_cached_reqs
        if (
            all(req.req_id in decodes for req in output.scheduled_new_reqs)
            and all(rid in decodes for rid in cached.req_ids)
            and cached.resumed_req_ids <= decodes
            and cached.all_token_ids.keys() <= decodes
        ):
            # Lifecycle work was applied to both runners above. Every request
            # belongs to D, so retain the native per-request arrays instead of
            # allocating sliced copies of all of them.
            lane_output = replace(
                output,
                finished_req_ids=set(),
                preempted_req_ids=(output.preempted_req_ids or set()) & decodes,
            )
        else:
            lane_output = _select_requests(output, decodes, carry_side_effects=True)
        backend = self._execution_backend
        caller_state = backend.capture()
        try:
            backend.restore(self._lane_states["decode"])
            result = self.decode_runner.execute_model(lane_output)
            backend.fence()
        finally:
            backend.restore(caller_state)
        if result is not None:
            raise RuntimeError("MuxPD requires a deferred native decode sampler")
        self._pending_lanes = ["decode"]
        self._pending_order = list(lane_output.num_scheduled_tokens)
        self.muxpd_stats["decode_batches"] += 1
        self._decode_only_fast_path = True
        self._awaiting_sample = True

    def _advance_lanes(
        self,
        decode: LayerInterleaver | None,
        prefill_budget: int | None,
        pause_prefill: bool,
    ) -> None:
        prefill = None if pause_prefill else self._prefill_task
        if decode is None:
            if prefill is not None:
                self._run_lane(prefill, "prefill", prefill_budget)
            return
        if prefill is None or self._stream_mode == "fenced":
            self._run_lane(decode, "decode", None)
            if prefill is not None:
                self._run_lane(prefill, "prefill", prefill_budget)
            return

        if self._overlap_communication == "isolated":
            # Independent HCCL groups remove the cross-lane collective chain.
            # Launch captured D first so its device work can overlap raw P's
            # host submission. Native D replay/update synchronization remains
            # inside execute_model; sampling still follows both submissions.
            # Eager D retains P-first ordering. Both paths use two lane steps,
            # without a host greenlet round trip at every decoder layer.
            if self._decode_graph_enabled:
                self._run_lane(decode, "decode", None)
                self._run_lane(prefill, "prefill", prefill_budget)
            else:
                self._run_lane(prefill, "prefill", prefill_budget)
                self._run_lane(decode, "decode", None)
            return

        # Putting an entire decode network ahead of P would make P's first
        # collective wait for D's final collective. Alternate full layers so
        # independent compute can overlap around per-group communication gates.
        task = prefill.tasks["prefill"]
        remaining = prefill_budget if prefill_budget is not None else sys.maxsize
        while not decode.tasks["decode"].done:
            if task.done or remaining == 0:
                self._run_lane(decode, "decode", None)
                break
            self._run_lane(decode, "decode", 1)
            before = task.completed_layers
            self._run_lane(prefill, "prefill", 1)
            remaining -= task.completed_layers - before
        if not task.done and remaining > 0:
            self._run_lane(prefill, "prefill", remaining)

    @torch.inference_mode()
    def _abort_prefill(self) -> None:
        if self._prefill_task is None:
            return
        self._snapshot_prefill_progress()
        self._prefill_task.close()
        # An ACK permits scheduler KV reuse. Host stack teardown alone cannot
        # establish that previously launched cache writes have finished.
        self._streams["prefill"].synchronize()
        self._cancelled_ack.update(self._prefill_order)
        self._retire_finished(set(self._prefill_order))
        self.model_runner.execute_model_state = None
        self._clear_prefill()

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> None:
        if self._awaiting_sample:
            raise RuntimeError("MuxPD requires sampler ACK before another iteration")
        previous_completed_decode_only = self._decode_only_fast_path
        self._step_profiler()
        if getattr(scheduler_output, "muxpd_execution", None) != "decoupled":
            raise ValueError("Persistent MuxPD worker requires the decoupled scheduler")
        new_prefills = set(scheduler_output.muxpd_prefill_request_ids)
        pending = set(scheduler_output.muxpd_pending_prefill_ids)
        cancelled = set(scheduler_output.muxpd_cancel_prefill_request_ids)
        cohort_id = scheduler_output.muxpd_prefill_cohort_id
        scheduled = set(scheduler_output.num_scheduled_tokens)
        if not new_prefills <= scheduled or not cancelled <= pending:
            raise ValueError("Invalid MuxPD prefill/cancellation request IDs")
        if new_prefills and self._prefill_task is not None:
            raise RuntimeError("MuxPD supports only one active prefill cohort")
        expected_pending = set(self._prefill_order) | new_prefills
        if pending != expected_pending:
            raise RuntimeError("MuxPD scheduler and worker disagree on pinned prefills")
        if pending and (
            cohort_id is None
            or (self._prefill_task is not None and cohort_id != self._prefill_cohort_id)
        ):
            raise RuntimeError("MuxPD prefill cohort identifier mismatch")
        protected = set(self._prefill_order) & (
            set(scheduler_output.finished_req_ids)
            | set(scheduler_output.preempted_req_ids or ())
            | scheduler_output.scheduled_cached_reqs.resumed_req_ids
        )
        if protected:
            raise RuntimeError("MuxPD cannot retire or preempt a pinned prefill")
        self._validate_requests(scheduler_output)
        self._retire_finished(scheduler_output.finished_req_ids)
        self._invalidate_batch_rows(
            set(scheduler_output.preempted_req_ids or ())
            | scheduler_output.scheduled_cached_reqs.resumed_req_ids
            | {req.req_id for req in scheduler_output.scheduled_new_reqs}
        )
        self._iteration_cohort_id = cohort_id
        self._iteration_prefill_layers_completed = 0
        self._iteration_prefill_layers_total = 0
        self._iteration_prefill_progress_source = "none"
        self._iteration_prefill_timer = None
        self._cancelled_ack = set()
        self._decode_only_fast_path = False
        self._pending_lanes = []
        self._pending_order = []
        decodes = scheduled - new_prefills
        if decodes & pending:
            raise RuntimeError("A pinned prefill cannot simultaneously enter decode")

        decode_only = bool(
            decodes
            and not new_prefills
            and self._prefill_task is None
            and not pending
            and not cancelled
            and cohort_id is None
        )
        reuse_decode_dependency = (
            decode_only
            and previous_completed_decode_only
            and getattr(self, "_decode_graph_enabled", False)
            and self._stream_mode == "fenced"
            and getattr(self, "profiler", None) is None
        )
        if reuse_decode_dependency:
            # The preceding D-only sampler completed successfully. Lifecycle
            # edits above touch only host rows; native block zeroing happens
            # inside execute_model on D. Preserve the latest backend tail: D
            # FIFO suffices when it belongs to D, otherwise restore(D) still
            # waits for its actual producer. Do not publish an empty caller
            # event that creates a redundant caller -> D dependency.
            # Custom collective RPCs that submit caller-stream device work
            # must publish it through backend.fence(); bypassing that stream
            # protocol is unsupported. Profiling always uses the full path.
            execution_output = scheduler_output
            key = "decode_preparation_fences_skipped"
            self.muxpd_stats[key] = self.muxpd_stats.get(key, 0) + 1
        else:
            execution_output = self._begin_execution(scheduler_output)

        if decode_only:
            self._execute_decode_only(execution_output, decodes)
            return None

        budget = None
        if (
            pending
            and not scheduler_output.muxpd_pause_prefill
            and not (pending <= (cancelled | self._prefill_cancelled))
        ):
            budget = scheduler_output.muxpd_layers_per_slice
            if type(budget) is not int or budget < 1:
                raise ValueError("MuxPD layer budget must be a positive integer")
            if scheduler_output.muxpd_finish_prefill:
                budget = None
            if getattr(self, "_prefill_groups_enabled", False) and budget not in (
                None,
                4,
            ):
                raise ValueError("Prefill FX groups require an atomic four-layer quota")

        if new_prefills:
            output = _select_requests(
                execution_output, new_prefills, carry_side_effects=not decodes
            )
            task = LayerInterleaver(self._execution_backend)
            controller = getattr(self, "_decode_graph_controller", None)
            self._prefill_compiled = bool(
                controller is not None
                and controller.compile_full_prefill
                and not scheduler_output.muxpd_pause_prefill
                and not cancelled
                and (budget is None or budget >= self._decoder_layer_count)
            )
            self._prefill_full_forward_return = False
            if self._prefill_compiled:
                forward = partial(self._execute_compiled_prefill, output, controller)
            else:
                forward = partial(self.model_runner.execute_model, output)
            task.add(
                "prefill",
                forward,
                self._lane_states["prefill"],
            )
            self._prefill_task = task
            self._prefill_order = list(output.num_scheduled_tokens)
            self._prefill_cohort_id = cohort_id
            self._prefill_num_tokens = output.total_num_scheduled_tokens
            self.muxpd_stats["prefill_batches"] += 1
        self._prefill_cancelled.update(cancelled)
        if self._prefill_task is not None and self._prefill_cancelled == pending:
            self._abort_prefill()

        decode_task = None
        if decodes:
            output = _select_requests(
                execution_output, decodes, carry_side_effects=True
            )
            task = LayerInterleaver(self._execution_backend)
            task.add(
                "decode",
                lambda output=output: self.decode_runner.execute_model(output),
                self._lane_states["decode"],
            )
            decode_task = task
            decode_order = list(output.num_scheduled_tokens)

        self._advance_lanes(decode_task, budget, scheduler_output.muxpd_pause_prefill)
        self._snapshot_prefill_progress()

        if decode_task is not None:
            if decode_task.tasks["decode"].result is not None:
                raise RuntimeError("MuxPD requires a deferred native decode sampler")
            self._pending_lanes.append("decode")
            self._pending_order.extend(decode_order)
            self.muxpd_stats["decode_batches"] += 1

        if self._prefill_task is not None and not scheduler_output.muxpd_pause_prefill:
            if self._prefill_task.tasks["prefill"].done:
                if self._prefill_task.tasks["prefill"].result is not None:
                    raise RuntimeError(
                        "MuxPD requires a deferred native prefill sampler"
                    )
                self._pending_lanes.append("prefill")
                self._pending_order.extend(self._prefill_order)
        self._awaiting_sample = True
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: Any) -> ModelRunnerOutput:
        if not self._awaiting_sample:
            raise RuntimeError("MuxPD sample_tokens requires an executed iteration")
        completed = set()
        cancelled = self._cancelled_ack.copy()
        prefill_completed = "prefill" in self._pending_lanes
        sampled_lane = bool(self._pending_lanes)
        if prefill_completed:
            cancelled.update(self._prefill_cancelled)
            completed.update(set(self._prefill_order) - cancelled)
        output = super().sample_tokens(grammar_output)
        if cancelled:
            keep = [req_id for req_id in output.req_ids if req_id not in cancelled]
            filtered = ModelRunnerOutput(
                req_ids=keep,
                req_id_to_index={req_id: i for i, req_id in enumerate(keep)},
                sampled_token_ids=[
                    output.sampled_token_ids[output.req_id_to_index[req_id]]
                    for req_id in keep
                ],
                num_nans_in_logits={
                    req_id: value
                    for req_id, value in (output.num_nans_in_logits or {}).items()
                    if req_id not in cancelled
                }
                or None,
            )
            output = _merge_outputs([filtered], keep)
            self._retire_finished(cancelled)
        output.muxpd_prefill_cohort_id = self._iteration_cohort_id
        output.muxpd_completed_prefill_ids = frozenset(completed)
        output.muxpd_cancelled_prefill_ids = frozenset(cancelled)
        # Enqueued layer boundaries are progress evidence, not device completion
        # or permission to free KV. Only the completion/cancellation ACK is that.
        output.muxpd_prefill_layers_completed = self._iteration_prefill_layers_completed
        output.muxpd_prefill_layers_total = self._iteration_prefill_layers_total
        output.muxpd_prefill_progress_source = self._iteration_prefill_progress_source
        timer = getattr(self, "_iteration_prefill_timer", None)
        output.muxpd_prefill_slice_timing = (
            timer.receipt(
                self._iteration_prefill_layers_completed,
                # Fenced samplers wait for the P slice tail before their native
                # synchronous D2H. Empty transactions have no such barrier.
                completion_barrier=sampled_lane and self._stream_mode == "fenced",
            )
            if timer is not None
            else None
        )
        if prefill_completed:
            self._clear_prefill()
        self._awaiting_sample = False
        return output

    def shutdown(self) -> None:
        if self._prefill_task is not None:
            # Failed teardown or synchronization leaves the cohort owned here.
            # Parent cleanup must not clear its streams/model before a retry can
            # safely finish the abort and release the suspended task's state.
            self._abort_prefill()
        super().shutdown()
