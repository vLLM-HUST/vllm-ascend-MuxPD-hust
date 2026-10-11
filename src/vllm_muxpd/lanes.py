"""Cooperative decoder execution; no worker threads or device dependencies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import get_ident
from typing import Any, Protocol

from greenlet import GreenletExit, getcurrent, greenlet


class ExecutionBackend(Protocol):
    """State belongs to a lane; fences protect shared device workspaces."""

    def capture(self) -> Any: ...

    def restore(self, state: Any) -> None: ...

    def fence(self) -> None: ...


class SliceObserver(Protocol):
    """Optional observations made inside the lane and its existing fences."""

    def start(self) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class LayerBoundary:
    lane: str
    completed_layers: int


class CooperativeTask:
    """A forward whose Python stack survives decoder-layer boundaries."""

    def __init__(self, name: str, function: Callable[[], Any], state: Any) -> None:
        self.name = name
        self.state = state
        self.completed_layers = 0
        self.result: Any = None
        self.coroutine = greenlet(function, parent=getcurrent())

    @property
    def done(self) -> bool:
        return self.coroutine.dead


class LayerInterleaver:
    """Advance separate forwards deterministically on one host thread.

    A model's decoder post-hooks call ``checkpoint``. ``step`` returns at
    most after ``layer_budget`` such calls, preserving the live Python stack.
    A backend fence runs at each yield and forward completion. The initial
    Ascend backend implements a full layer fence, not concurrent kernels.
    """

    def __init__(self, backend: ExecutionBackend) -> None:
        self.backend = backend
        self.tasks: dict[str, CooperativeTask] = {}
        self.active: CooperativeTask | None = None
        self._target_layer = 0
        self._owner_thread = get_ident()
        self._slice_observer: SliceObserver | None = None

    def _check_thread(self) -> None:
        if get_ident() != self._owner_thread:
            raise RuntimeError("MuxPD cooperative execution requires one host thread")

    def add(self, name: str, function: Callable[[], Any], state: Any) -> None:
        self._check_thread()
        if name in self.tasks:
            raise ValueError(f"Duplicate MuxPD lane: {name}")
        self.tasks[name] = CooperativeTask(name, function, state)

    def checkpoint(self, *, final: bool = False) -> None:
        task = self.active
        if task is None:
            return
        self._check_thread()
        if getcurrent() is not task.coroutine:
            raise RuntimeError("Decoder hook called outside its cooperative task")
        task.completed_layers += 1
        if not final and task.completed_layers >= self._target_layer:
            if self._slice_observer is not None:
                self._slice_observer.stop()
            self.backend.fence()
            task.state = self.backend.capture()
            task.coroutine.parent.switch(
                LayerBoundary(task.name, task.completed_layers)
            )

    def step(
        self,
        name: str,
        layer_budget: int = 1,
        *,
        observer: SliceObserver | None = None,
    ) -> LayerBoundary | None:
        self._check_thread()
        if type(layer_budget) is not int or layer_budget < 1:
            raise ValueError("layer_budget must be a positive integer")
        if self.active is not None:
            raise RuntimeError("Cannot recursively advance MuxPD lanes")
        task = self.tasks[name]
        if task.done:
            return None
        caller_state = self.backend.capture()
        self.active = task
        self._slice_observer = observer
        self._target_layer = task.completed_layers + layer_budget
        try:
            self.backend.restore(task.state)
            if observer is not None:
                observer.start()
            value = task.coroutine.switch()
            if task.done:
                task.result = value
                if observer is not None:
                    observer.stop()
                self.backend.fence()
                task.state = self.backend.capture()
                return None
            if not isinstance(value, LayerBoundary):
                raise RuntimeError("MuxPD forward yielded outside a decoder hook")
            return value
        finally:
            self.active = None
            self._slice_observer = None
            self.backend.restore(caller_state)

    def run(self, order: Iterable[str], layer_budget: int = 1) -> dict[str, Any]:
        names = tuple(order)
        if len(set(names)) != len(names) or set(names) != set(self.tasks):
            raise ValueError("Execution order must include each lane exactly once")
        try:
            while any(not self.tasks[name].done for name in names):
                for name in names:
                    self.step(name, layer_budget)
            return {name: self.tasks[name].result for name in names}
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Unwind suspended contexts deterministically on the owning thread."""
        self._check_thread()
        caller_state = self.backend.capture()
        try:
            for task in self.tasks.values():
                if not task.done:
                    if not task.coroutine:
                        # A never-started stack owns no device state. Restoring
                        # it could violate a backend's initial-preparation fence.
                        task.coroutine.throw(GreenletExit)
                        continue
                    self.backend.restore(task.state)
                    task.coroutine.throw(GreenletExit)
                    self.backend.fence()
        finally:
            self.active = None
            self.backend.restore(caller_state)
