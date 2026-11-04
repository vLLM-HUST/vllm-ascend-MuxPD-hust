"""Opt-in slice observations; no additional device synchronization."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any


class PrefillSliceTimer:
    """Record a lane interval, read it only after an existing completion barrier.

    NPU event elapsed time includes gaps while the host submits work. It is not
    a sum of kernel durations, nor the decode delay or end-to-end step latency.
    The start event follows the lane's incoming dependency, excluding the wait
    for previously submitted decode work. A compiled mode labels permission to
    run a full compiled forward, not proof that native actually hit an FX graph.
    """

    def __init__(
        self,
        start_event: Any,
        end_event: Any,
        stream: Any,
        *,
        start_layer: int,
        prefill_tokens: int,
        compiled: bool,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.start_event = start_event
        self.end_event = end_event
        self.stream = stream
        self.start_layer = start_layer
        self.prefill_tokens = prefill_tokens
        self.compiled = compiled
        self.clock = clock
        self._started: float | None = None
        self._host_submit_ms: float | None = None

    def start(self) -> None:
        if self._started is not None:
            raise RuntimeError("MuxPD slice timer cannot be reused")
        self._started = self.clock()
        self.start_event.record(self.stream)

    def stop(self) -> None:
        if self._started is None or self._host_submit_ms is not None:
            raise RuntimeError("MuxPD slice timer has no active interval")
        self.end_event.record(self.stream)
        self._host_submit_ms = (self.clock() - self._started) * 1000

    def receipt(
        self, completed_layers: int, *, completion_barrier: bool
    ) -> dict[str, Any]:
        if self._host_submit_ms is None:
            raise RuntimeError("MuxPD slice timer has not stopped")
        # An empty sampler transaction does not synchronize an unfinished P.
        # Never wait for it here, and never assume host layer progress proves
        # device completion. query() is nonblocking and is an additional check.
        complete = completion_barrier and self.end_event.query()
        elapsed = (
            self.start_event.elapsed_time(self.end_event) if complete else None
        )
        return {
            "version": 1,
            "layer_delta": completed_layers - self.start_layer,
            "first_slice": self.start_layer == 0,
            "prefill_tokens": self.prefill_tokens,
            "mode": "compiled_full_forward" if self.compiled else "raw_layers",
            "host_submit_ms": self._host_submit_ms,
            "device_elapsed_ms": elapsed,
            "device_complete": complete,
        }
