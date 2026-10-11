"""Opt-in greedy D sampling submission with native retirement left in place.

This module has no Torch import until the real runtime is constructed. The
adapter owns only a D runner's instance methods, pinned buffer and one ticket;
it never changes native request bookkeeping or returns an early worker ACK.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from threading import get_ident
from types import MethodType
from typing import Any, Callable

_MISSING = object()


class NPUSampleCopyRuntime:
    def __init__(self, runner: Any, stream: Any) -> None:
        import torch

        if not runner.pin_memory:
            raise ValueError("MuxPD decode sampling requires pinned CPU memory")
        self.stream = stream
        self.buffer = torch.empty_like(
            runner.sampled_token_ids_pinned_cpu, pin_memory=True
        )
        self.event = torch.npu.Event()

    def enqueue(self, sampled: Any) -> Any:
        view = self.buffer[:sampled.shape[0]]
        view.copy_(sampled, non_blocking=True)
        self.event.record(self.stream)
        return view

    def materialize(self, view: Any) -> list[list[int]]:
        self.event.synchronize()
        return view.tolist()

    def synchronize(self) -> None:
        self.stream.synchronize()


@dataclass
class _SampleTicket:
    submission: int
    state: Any
    metadata: Any
    request_ids: tuple[str, ...]
    request_indices: dict[str, int]
    sampler_output: Any = None
    cpu_view: Any = None
    sample_consumed: bool = False
    copy_consumed: bool = False


class DecodeSamplePipeline:
    """Submit once before P, retire once through the unchanged native sampler.

    Unsupported requests take the native path before any sampling is submitted.
    Once submission starts, every failure poisons the adapter until shutdown;
    retrying sampling could otherwise consume RNG or append token history twice.
    """

    def __init__(
        self,
        runner: Any,
        runtime: Any,
        assert_scope: Callable[[], None],
        *,
        builtin_processor_types: tuple[type, ...],
        diagnostic: Callable[[dict], None] | None = None,
    ) -> None:
        self.runner = runner
        self.runtime = runtime
        self.assert_scope = assert_scope
        self.builtin_processor_types = builtin_processor_types
        self.diagnostic = diagnostic
        self._thread = get_ident()
        self._phase = "idle"
        self._ticket: _SampleTicket | None = None
        self.stats: Counter[str] = Counter()
        self._original_attributes = {
            name: vars(runner).get(name, _MISSING)
            for name in ("_sample", "_to_list", "sample_tokens")
        }
        self._native_sample = runner._sample
        self._native_to_list = runner._to_list
        self._native_sample_tokens = runner.sample_tokens

        def sample(_runner, logits, spec_decode_metadata):
            return self._sample(logits, spec_decode_metadata)

        def to_list(_runner, sampled):
            return self._to_list(sampled)

        def sample_tokens(_runner, grammar_output):
            return self.retire(grammar_output)

        self._installed = {
            "_sample": MethodType(sample, runner),
            "_to_list": MethodType(to_list, runner),
            "sample_tokens": MethodType(sample_tokens, runner),
        }
        for name, method in self._installed.items():
            setattr(runner, name, method)

    @property
    def pending(self) -> bool:
        return self._ticket is not None

    def _check(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("MuxPD decode sampler requires its owner thread")
        if self._phase in ("closed", "poisoned"):
            raise RuntimeError(f"MuxPD decode sampler is {self._phase}")
        for name, method in self._installed.items():
            if getattr(self.runner, name) is not method:
                raise RuntimeError("MuxPD decode sampler method was replaced")

    def assert_idle(self) -> None:
        self._check()
        if self._phase != "idle" or self._ticket is not None:
            raise RuntimeError("MuxPD cannot reuse D before its sample ticket retires")

    def _unsupported(self, state: Any) -> str | None:
        runner = self.runner
        if runner.use_async_scheduling or runner.speculative_config is not None:
            return "async_or_speculative"
        if runner.routed_experts_initialized:
            return "routed_experts"
        if state is None or state.spec_decode_metadata is not None:
            return "missing_or_speculative_state"
        if state.logits is None or runner.kv_connector_output is not None:
            return "logits_or_connector"
        batch = runner.input_batch
        metadata = batch.sampling_metadata
        if metadata.all_greedy is not True or metadata.all_random:
            return "not_all_greedy"
        if metadata.max_num_logprobs is not None or metadata.logprob_token_ids:
            return "logprobs"
        if (
            not metadata.no_penalties
            or metadata.bad_words_token_ids
            or metadata.allowed_token_ids_mask is not None
            or metadata.thinking_budget_state_holder is not None
        ):
            return "stateful_or_constrained_sampling"
        if runner.model_config.logits_processors or any(
            type(processor) not in self.builtin_processor_types
            for processor in metadata.logitsprocs.all
        ):
            # Entry-point plugins may exist even without model_config entries.
            return "custom_logits_processor"
        if runner.num_discarded_requests or batch.sampled_token_ids_cpu is not None:
            return "discard_or_async_feedback"
        ids = tuple(batch.req_ids)
        scheduled = state.scheduler_output.num_scheduled_tokens
        if (
            not ids
            or set(ids) != set(scheduled)
            or any(count != 1 for count in scheduled.values())
        ):
            return "not_one_token_decode"
        for rid in ids:
            params = runner.requests[rid].sampling_params
            if params is None or (
                params.structured_outputs is not None
                or params.logprobs is not None
                or params.prompt_logprobs is not None
            ):
                return "grammar_or_logprobs"
        return None

    def enqueue(self) -> bool:
        """Return false before submission if this whole D batch is unsupported."""
        self._check()
        if self._phase != "idle" or self._ticket is not None:
            raise RuntimeError("MuxPD decode sampler already has an in-flight ticket")
        self.assert_scope()
        state = self.runner.execute_model_state
        reason = self._unsupported(state)
        if reason is not None:
            self.stats["fallback_" + reason] += 1
            if self.diagnostic is not None:
                self.diagnostic({"phase": "fallback", "reason": reason})
            return False
        batch = self.runner.input_batch
        self.stats["submissions"] += 1
        ticket = _SampleTicket(
            self.stats["submissions"], state,
            batch.sampling_metadata, tuple(batch.req_ids),
            batch.req_id_to_index.copy(),
        )
        self._ticket = ticket
        self._phase = "submitting"
        try:
            ticket.sampler_output = self._native_sample(
                state.logits, state.spec_decode_metadata
            )
            sampled = ticket.sampler_output.sampled_token_ids
            if (
                ticket.sampler_output.logprobs_tensors is not None
                or tuple(sampled.shape) != (len(ticket.request_ids), 1)
            ):
                raise RuntimeError("MuxPD greedy sampler returned unsupported output")
            ticket.cpu_view = self.runtime.enqueue(sampled)
            self._phase = "ready"
            self.stats["enqueued"] += 1
            self._diagnose("enqueue", ticket)
            return True
        except BaseException:
            # Device operations may have been submitted before the exception.
            # Keep state/output/buffer references until a successful stream join.
            self._phase = "poisoned"
            raise

    def _sample(self, logits: Any, spec_decode_metadata: Any) -> Any:
        self._check()
        if self._ticket is None:
            return self._native_sample(logits, spec_decode_metadata)
        ticket = self._ticket
        if (
            self._phase != "retiring"
            or ticket.sample_consumed
            or logits is not ticket.state.logits
            or spec_decode_metadata is not ticket.state.spec_decode_metadata
        ):
            raise RuntimeError(
                "MuxPD decode sample ticket identity/consumption mismatch"
            )
        ticket.sample_consumed = True
        return ticket.sampler_output

    def _to_list(self, sampled: Any) -> list[list[int]]:
        self._check()
        if self._ticket is None:
            return self._native_to_list(sampled)
        ticket = self._ticket
        if (
            self._phase != "retiring"
            or not ticket.sample_consumed
            or ticket.copy_consumed
            or sampled is not ticket.sampler_output.sampled_token_ids
        ):
            raise RuntimeError("MuxPD decode copy ticket identity/consumption mismatch")
        self.assert_scope()
        result = self.runtime.materialize(ticket.cpu_view)
        ticket.copy_consumed = True
        return result

    def retire(self, grammar_output: Any) -> Any:
        self._check()
        if self._ticket is None:
            return self._native_sample_tokens(grammar_output)
        ticket = self._ticket
        try:
            self.assert_scope()
            batch = self.runner.input_batch
            if (
                self._phase != "ready"
                or grammar_output is not None
                or self.runner.execute_model_state is not ticket.state
                or batch.sampling_metadata is not ticket.metadata
                or tuple(batch.req_ids) != ticket.request_ids
                or batch.req_id_to_index != ticket.request_indices
            ):
                raise RuntimeError("MuxPD decode retirement state changed before ACK")
            self._phase = "retiring"
            # Native code still clears ExecuteModelState, records sampling_done,
            # adjusts discarded generators, appends token IDs, constructs the
            # output and submits hybrid feedback at its original boundary.
            output = self._native_sample_tokens(grammar_output)
            if (
                not ticket.sample_consumed
                or not ticket.copy_consumed
                or self.runner.execute_model_state is not None
            ):
                raise RuntimeError("MuxPD native sampler did not consume its ticket")
            self.stats["retired"] += 1
            self._diagnose("retire", ticket)
            self._ticket = None
            self._phase = "idle"
            return output
        except BaseException:
            self._phase = "poisoned"
            raise

    def _diagnose(self, phase: str, ticket: _SampleTicket) -> None:
        if self.diagnostic is not None:
            self.diagnostic({
                "phase": phase, "submission": ticket.submission,
                "request_ids": list(ticket.request_ids),
                "private_copy_waited": ticket.copy_consumed,
                "native_retirement_completed": phase == "retire",
                "counts": dict(self.stats),
            })

    def close(self) -> None:
        """Join before releasing a failed/in-flight ticket; retry a failed join."""
        if self._phase == "closed":
            return
        if get_ident() != self._thread:
            raise RuntimeError("MuxPD decode sampler requires its owner thread")
        if self._ticket is not None:
            self.runtime.synchronize()
        for name, method in self._installed.items():
            if getattr(self.runner, name) is method:
                original = self._original_attributes[name]
                if original is _MISSING:
                    delattr(self.runner, name)
                else:
                    setattr(self.runner, name, original)
        self._ticket = None
        self._phase = "closed"
        self._installed.clear()
        self._original_attributes.clear()
        self._native_sample = self._native_to_list = self._native_sample_tokens = None
        self.runner = self.runtime = None
        self.assert_scope = lambda: None
        self.diagnostic = None
