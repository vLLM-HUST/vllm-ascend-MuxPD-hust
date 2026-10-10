"""Route independent P/D forwards to distinct existing process groups.

The worker owns group creation, collective warmup, stream synchronization, and
group destruction. This module changes only communicator method dispatch; it
never replaces vLLM's global TP/EP coordinators or creates device resources.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from threading import get_ident
from typing import Any

_MISSING = object()


class LaneCollectiveRouter:
    """Use a prefill sibling communicator while preserving native decode calls.

    ``backend.active_lane`` must be ``prefill``, ``decode``, or ``None``. Caller
    work (including native warmup) uses the original communicator. Independent
    groups need no cross-lane event gate; the worker must maintain its existing
    caller/lane preparation barriers and use one host thread. Every rank must
    create sibling groups in the same order, with identical rank membership.

    Call ``restore`` only after the worker has synchronized both lane streams.
    Neither original nor sibling groups are destroyed here.
    """

    METHODS = ("all_reduce", "all_gather", "reduce_scatter")

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._thread = get_ident()
        self._installed: list[tuple[Any, str, Any, Any]] = []
        self._failed = False
        self.calls = {"caller": 0, "decode": 0, "prefill": 0}

    def _check_thread(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("MuxPD collective routing requires one host thread")

    def _wrap(self, original: Any, prefill: Any) -> Any:
        @wraps(original)
        def routed(*args: Any, **kwargs: Any) -> Any:
            self._check_thread()
            if self._failed:
                raise RuntimeError(
                    "MuxPD collective router is poisoned after a failure"
                )
            try:
                lane = self._backend.active_lane
                if lane == "prefill":
                    function = prefill
                elif lane in (None, "decode"):
                    function = original
                else:
                    raise RuntimeError(f"Unknown MuxPD collective lane: {lane!r}")
                result = function(*args, **kwargs)
            except BaseException:
                self._failed = True
                raise
            self.calls[lane or "caller"] += 1
            return result

        routed._muxpd_collective_router = self
        return routed

    def install(self, pairs: Iterable[tuple[Any, Any]]) -> "LaneCollectiveRouter":
        """Install ``(native communicator, prefill communicator)`` routes.

        Aliases of one native device group must use one sibling device group.
        Exact repeated pairs are harmless. A sibling may not also be a native
        endpoint: that would permit recursion or accidentally share a route.
        """
        self._check_thread()
        if self._installed or self._failed:
            raise RuntimeError("MuxPD collective router cannot be installed twice")
        pairs = tuple(pairs)
        if not pairs:
            raise ValueError("MuxPD collective router requires communicator pairs")
        native_ids = {id(native) for native, _ in pairs}
        native_group_ids = {
            id(native.device_group)
            for native, _ in pairs
            if getattr(native, "device_group", None) is not None
        }
        seen: dict[int, Any] = {}
        group_routes: dict[int, Any] = {}
        try:
            for native, prefill in pairs:
                if id(prefill) in native_ids:
                    raise ValueError("MuxPD sibling cannot be a native endpoint")
                if id(native) in seen:
                    if seen[id(native)] is not prefill:
                        raise ValueError(
                            "MuxPD native communicator has conflicting routes"
                        )
                    continue
                seen[id(native)] = prefill
                native_group = getattr(native, "device_group", None)
                prefill_group = getattr(prefill, "device_group", None)
                if native_group is None or prefill_group is None:
                    raise ValueError("MuxPD routing requires device_group endpoints")
                if id(prefill_group) in native_group_ids:
                    raise ValueError(
                        "MuxPD prefill requires an independent device_group"
                    )
                previous_group = group_routes.setdefault(
                    id(native_group), prefill_group
                )
                if previous_group is not prefill_group:
                    raise ValueError(
                        "MuxPD group aliases require the same sibling group"
                    )
                for name in self.METHODS:
                    original = getattr(native, name, None)
                    target = getattr(prefill, name, None)
                    if not callable(original) or not callable(target):
                        raise ValueError(f"MuxPD communicator requires {name}()")
                    if any(
                        hasattr(function, marker)
                        for function in (original, target)
                        for marker in (
                            "_muxpd_collective_router",
                            "_muxpd_collective_gate",
                        )
                    ):
                        raise RuntimeError(
                            "MuxPD isolated routing requires unwrapped communicators"
                        )
                    previous = vars(native).get(name, _MISSING)
                    wrapper = self._wrap(original, target)
                    setattr(native, name, wrapper)
                    self._installed.append((native, name, previous, wrapper))
        except BaseException:
            self.restore()
            raise
        return self

    def restore(self) -> None:
        """Restore only our wrappers; leave external replacements intact."""
        self._check_thread()
        changed = []
        for native, name, previous, wrapper in reversed(self._installed):
            if getattr(native, name, None) is not wrapper:
                changed.append(name)
                continue
            if previous is _MISSING:
                delattr(native, name)
            else:
                setattr(native, name, previous)
        self._installed.clear()
        if changed:
            raise RuntimeError(f"MuxPD router methods changed externally: {changed}")
