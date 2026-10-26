"""Opt-in native asynchronous D output, with synchronous P ownership."""

from __future__ import annotations

from typing import Any

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from .decoupled_worker import DecoupledMuxPDNPUWorker
from .worker import _merge_outputs


class _AsyncDecodeOutput(AsyncModelRunnerOutput):
    """An immutable transaction snapshot consumed by the native output thread."""

    def __init__(self, native: AsyncModelRunnerOutput, order: tuple[str, ...]):
        self._native = native
        self._order = order
        self._consumed = False

    def get_output(self) -> ModelRunnerOutput:
        if self._consumed:
            raise RuntimeError("Async MuxPD output was already consumed")
        self._consumed = True
        output = _merge_outputs(
            [self._native.get_output()], list(self._order), passthrough=True
        )
        # No worker/runner state is read on the native output thread. A later
        # execute/sample RPC can already have replaced every iteration field.
        output.muxpd_prefill_cohort_id = None
        output.muxpd_completed_prefill_ids = frozenset()
        output.muxpd_cancelled_prefill_ids = frozenset()
        output.muxpd_prefill_layers_completed = 0
        output.muxpd_prefill_layers_total = 0
        output.muxpd_prefill_progress_source = "none"
        output.muxpd_prefill_slice_timing = None
        return output


class AsyncDecoupledMuxPDNPUWorker(DecoupledMuxPDNPUWorker):
    """Keep native D token feedback and defer only pure-D host output copies.

    The scheduler drains its queue before P activity. Mixed iterations still
    materialize D in the owning worker thread before the existing merged ACK;
    only P-free D returns an AsyncModelRunnerOutput to the native output thread.
    """

    def __init__(self, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        policy = vllm_config.scheduler_config.batch_admission_policy_config or {}
        if policy.get("async_decode_pipeline") is not True:
            raise ValueError("Async MuxPD requires async_decode_pipeline=true")
        super().__init__(vllm_config, *args, **kwargs)

    def init_device(self) -> None:
        super().init_device()
        # Both runners allocate native async-capable buffers during startup.
        # P samples synchronously and never leaves a placeholder in its shared
        # request state. D keeps its private async stream and device token cache.
        self.model_runner.use_async_scheduling = False
        if not self.decode_runner.use_async_scheduling:
            raise RuntimeError("Async MuxPD decode runner is not asynchronous")

    def _materialize_lane_output(self, output: Any) -> ModelRunnerOutput:
        if isinstance(output, AsyncModelRunnerOutput):
            return output.get_output()
        return output

    def _restore_prefill_history(self, output):
        """A D request can re-enter synchronous P after native preemption.

        Async D's shared CPU history can contain -1 placeholders even after
        its output was delivered. The scheduler's resumed full history is the
        authority; synchronous P does not run native async row reconstruction.
        """
        prefills = set(output.muxpd_prefill_request_ids)
        cached = output.scheduled_cached_reqs
        for index, rid in enumerate(cached.req_ids):
            if rid not in prefills:
                continue
            state = self.model_runner.requests.get(rid)
            if state is None:
                raise RuntimeError("Async MuxPD resumed prefill has no request state")
            tokens = cached.all_token_ids.get(rid)
            count = cached.num_output_tokens[index]
            if (
                tokens is None
                or len(tokens) != state.num_prompt_tokens + count
                or any(token < 0 for token in tokens)
            ):
                raise RuntimeError("Async MuxPD requires a complete prefill history")
            state.output_token_ids = list(tokens[-count:]) if count else []

    @torch.inference_mode()
    def execute_model(self, scheduler_output):
        self._restore_prefill_history(scheduler_output)
        # The queue engine omits a separate sample RPC for zero-token P/control
        # steps. Such steps still carry a mandatory progress/cancellation ACK.
        super().execute_model(scheduler_output)
        if scheduler_output.total_num_scheduled_tokens == 0:
            return self.sample_tokens(None)
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output):
        if not self._decode_only_fast_path:
            return super().sample_tokens(grammar_output)
        if not self._awaiting_sample or grammar_output is not None:
            raise RuntimeError("Async MuxPD requires one unstructured sample RPC")
        if self._pending_lanes != ["decode"] or self._prefill_task is not None:
            raise RuntimeError("Async MuxPD output requires a pure decode transaction")
        order = tuple(self._pending_order)
        backend = self._execution_backend
        caller = backend.capture()
        try:
            backend.restore(self._lane_states["decode"])
            output = self.decode_runner.sample_tokens(None)
            if not isinstance(output, AsyncModelRunnerOutput):
                raise RuntimeError("Async MuxPD requires native asynchronous sampling")
            backend.fence()
        finally:
            backend.restore(caller)
            if self._stream_mode == "overlap":
                # Keep caller-side block zeroing ordered after every D write.
                # This is a device event dependency, not a host materialization.
                backend.join(["decode"])
        self._pending_lanes = []
        self._pending_order = []
        self._awaiting_sample = False
        return _AsyncDecodeOutput(output, order)
