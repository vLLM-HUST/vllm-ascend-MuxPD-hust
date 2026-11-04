"""Experimental building blocks for independent Ascend compute streams.

Nothing is installed on import. Callers must retain the worker's A2, no-MC2,
no-FlashComm constraints. Prefill remains raw eager; opt-in decode graphs require
the graph controller and isolated communication groups. Collective call order
must be identical on every rank. This module does not set core quotas or claim
hardware overlap without a device profile.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from threading import get_ident
from typing import Any, Protocol


class StreamRuntime(Protocol):
    def current_stream(self) -> Any: ...

    def set_stream(self, stream: Any) -> None: ...

    def record_event(self, stream: Any) -> Any: ...

    def wait_event(self, stream: Any, event: Any) -> None: ...


class TorchNPURuntime:
    """Lazy bridge; constructing this object does not create an NPU stream."""

    def __init__(self) -> None:
        import torch

        self.npu = torch.npu

    def current_stream(self) -> Any:
        return self.npu.current_stream()

    def set_stream(self, stream: Any) -> None:
        self.npu.set_stream(stream)

    def record_event(self, stream: Any) -> Any:
        return stream.record_event()

    def wait_event(self, stream: Any, event: Any) -> None:
        stream.wait_event(event)


_MISSING = object()


class CollectiveGates:
    """Serialize only collectives sharing a device process group.

    When the invoking stream changes, the wrapper waits before the collective.
    Same-stream calls are ordered by stream FIFO. Every call still records its
    completion after the native synchronous collective API has joined internal
    HCCL work to that stream, so a later stream switch sees the latest tail.
    Compute after the collective is outside this fence. This is deliberately
    instance-local and reversible.
    """

    METHODS = ("all_reduce", "all_gather", "reduce_scatter")

    def __init__(self, runtime: StreamRuntime) -> None:
        self.runtime = runtime
        self._thread = get_ident()
        self._installed: list[tuple[Any, str, Any, Any]] = []
        self._tails: dict[int, tuple[Any, Any]] = {}
        self._failed = False
        self.calls = 0

    def _check_thread(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("MuxPD collective gates require one host thread")

    def _wrap(self, function: Any, group_key: int) -> Any:
        @wraps(function)
        def gated(*args: Any, **kwargs: Any) -> Any:
            self._check_thread()
            if self._failed:
                raise RuntimeError("MuxPD collective gate is poisoned after a failure")
            stream = self.runtime.current_stream()
            previous = self._tails.get(group_key)
            try:
                if previous is not None:
                    previous_stream, previous_event = previous
                    if stream != previous_stream:
                        self.runtime.wait_event(stream, previous_event)
                result = function(*args, **kwargs)
                self._tails[group_key] = (stream, self.runtime.record_event(stream))
            except BaseException:
                self._failed = True
                raise
            self.calls += 1
            return result

        gated._muxpd_collective_gate = self
        return gated

    def install(self, communicators: Iterable[Any]) -> "CollectiveGates":
        self._check_thread()
        if self._installed or self._failed:
            raise RuntimeError("MuxPD collective gates cannot be installed twice")
        seen: set[int] = set()
        try:
            for communicator in communicators:
                if id(communicator) in seen:
                    continue
                seen.add(id(communicator))
                group = getattr(communicator, "device_group", None)
                if group is None:
                    raise ValueError("MuxPD communicator requires a device_group")
                for name in self.METHODS:
                    original = getattr(communicator, name, None)
                    if not callable(original):
                        raise ValueError(f"MuxPD communicator requires {name}()")
                    if hasattr(original, "_muxpd_collective_gate"):
                        raise RuntimeError(
                            "A MuxPD collective gate is already installed"
                        )
                    previous = vars(communicator).get(name, _MISSING)
                    wrapper = self._wrap(original, id(group))
                    setattr(communicator, name, wrapper)
                    self._installed.append((communicator, name, previous, wrapper))
        except BaseException as primary:
            try:
                self.restore()
            except BaseException as cleanup:
                if hasattr(primary, "add_note"):
                    primary.add_note(f"MuxPD gate rollback also failed: {cleanup!r}")
            raise
        return self

    def restore(self) -> None:
        """Remove our wrappers; never overwrite another actor's replacement."""
        self._check_thread()
        errors = []
        for index in reversed(range(len(self._installed))):
            communicator, name, previous, wrapper = self._installed[index]
            try:
                if getattr(communicator, name, None) is not wrapper:
                    raise RuntimeError(f"MuxPD gate methods changed externally: {name}")
                if previous is _MISSING:
                    delattr(communicator, name)
                else:
                    setattr(communicator, name, previous)
            except BaseException as error:
                errors.append(error)
            else:
                self._installed.pop(index)
        if errors:
            raise errors[0]
        self._tails.clear()


@dataclass(frozen=True)
class LaneState:
    forward_context: Any
    stream: Any
    workspace_manager: Any
    lane: str | None = None


class NPUOverlapBackend:
    """Lane-local streams, contexts and workspaces for ``LayerInterleaver``.

    ``begin_batch`` precedes each host transaction. If given a preparation lane,
    that lane's initial checkpoint releases the other lane, covering global
    new-block zeroing carried by its runner. With ``preparation_lane=None``, the
    caller must enqueue all shared preparation first; each lane waits only on
    that caller event. Later checkpoints merely record the lane's tail. ``join``
    is explicit; ordinary context restoration never inserts a whole-layer
    cross-lane dependency.
    """

    def __init__(
        self,
        *,
        runtime: StreamRuntime | None = None,
        forward_context_module: Any = None,
        workspace_module: Any = None,
        ascend_utils_module: Any = None,
    ) -> None:
        if forward_context_module is None:
            import vllm.forward_context as forward_context_module
        if workspace_module is None:
            import vllm.v1.worker.workspace as workspace_module
        if ascend_utils_module is None:
            import vllm_ascend.utils as ascend_utils_module

        self.runtime = runtime or TorchNPURuntime()
        self._forward = forward_context_module
        self._workspace = workspace_module
        self._ascend_utils = ascend_utils_module
        self._thread = get_ident()
        self._active_lane: str | None = None
        self._states: dict[str, LaneState] = {}
        self._tails: dict[str, Any] = {}
        self._initial_event: Any = None
        self._preparation_event: Any = None
        self._preparation_lane: str | None = None
        self._primed: set[str] = set()

    def _check_thread(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("MuxPD stream overlap requires one host thread")

    @property
    def active_lane(self) -> str | None:
        """Current lane for communicator routing; ``None`` is caller work."""
        self._check_thread()
        return self._active_lane

    def lane_state(
        self, name: str, stream: Any, *, workspace_manager: Any = None
    ) -> LaneState:
        """Register a lane, optionally retaining its profiled/captured manager.

        Graph P retains the owner's native profiled workspace; graph D uses
        its independent manager during KV initialization, capture and replay.
        Ordinary eager lanes allocate their own managers as before.
        """
        self._check_thread()
        if not name or name in self._states:
            raise ValueError(f"MuxPD lane must have a unique nonempty name: {name!r}")
        if any(stream == state.stream for state in self._states.values()):
            raise ValueError("MuxPD overlap lanes require independent streams")
        manager = workspace_manager
        if manager is None:
            manager = self._workspace.WorkspaceManager(stream.device, num_ubatches=1)
        if any(manager is state.workspace_manager for state in self._states.values()):
            raise ValueError(
                "MuxPD overlap lanes require independent workspace managers"
            )
        state = LaneState(None, stream, manager, name)
        self._states[name] = state
        return state

    def begin_batch(self, preparation_lane: str | None) -> None:
        """Publish shared preparation before restoring new or suspended lanes.

        ``None`` is for preparation already enqueued on the caller stream. In
        that mode runner shards must not repeat global block zeroing. A retained
        lane's existing work is ordered by its own stream; the new initial event
        does not join the other lane's outstanding tail.
        """
        self._check_thread()
        if self._active_lane is not None:
            raise RuntimeError("MuxPD must begin a batch from the caller context")
        if preparation_lane is not None and preparation_lane not in self._states:
            raise ValueError("MuxPD preparation lane is not registered")
        self._initial_event = self.runtime.record_event(self.runtime.current_stream())
        self._preparation_lane = preparation_lane
        self._preparation_event = None
        self._primed.clear()

    def capture(self) -> LaneState:
        self._check_thread()
        return LaneState(
            self._forward._forward_context,
            self.runtime.current_stream(),
            self._workspace._manager,
            self._active_lane,
        )

    def restore(self, state: LaneState) -> None:
        self._check_thread()
        if state.lane is not None:
            registered = self._states.get(state.lane)
            if registered is None or registered.stream != state.stream:
                raise ValueError("MuxPD cannot restore an unregistered lane stream")
            if state.workspace_manager is not registered.workspace_manager:
                raise ValueError("MuxPD cannot share workspace managers between lanes")
            if self._initial_event is None:
                raise RuntimeError("MuxPD begin_batch must precede lane execution")
            if state.lane not in self._primed:
                if (
                    self._preparation_lane is None
                    or state.lane == self._preparation_lane
                ):
                    barrier = self._initial_event
                else:
                    barrier = self._preparation_event
                    if barrier is None:
                        raise RuntimeError(
                            "MuxPD preparation lane must reach its first fence "
                            "before another lane starts"
                        )
                self.runtime.wait_event(state.stream, barrier)
                self._primed.add(state.lane)
        self._forward._forward_context = state.forward_context
        self.runtime.set_stream(state.stream)
        self._ascend_utils._CURRENT_STREAM = state.stream
        self._workspace._manager = state.workspace_manager
        self._active_lane = state.lane

    def fence(self) -> None:
        self._check_thread()
        lane = self._active_lane
        if lane is None:
            raise RuntimeError("MuxPD lane fence requires an active lane")
        event = self.runtime.record_event(self.runtime.current_stream())
        self._tails[lane] = event
        if lane == self._preparation_lane and self._preparation_event is None:
            self._preparation_event = event

    def join(self, lanes: Iterable[str] | None = None) -> None:
        """Order caller work after completed slices; this does not block the CPU."""
        self._check_thread()
        if self._active_lane is not None:
            raise RuntimeError("MuxPD join must run from the caller context")
        names = tuple(self._tails) if lanes is None else tuple(lanes)
        unknown = set(names) - self._states.keys()
        if unknown:
            raise ValueError(f"MuxPD cannot join unknown lanes: {sorted(unknown)}")
        stream = self.runtime.current_stream()
        for name in names:
            event = self._tails.get(name)
            if event is not None:
                self.runtime.wait_event(stream, event)
