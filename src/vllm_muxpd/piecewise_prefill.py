"""Ownership guards for startup-only, whole-short-P PIECEWISE capture.

No device modules are imported here. A native runtime adapter supplies graph
wrappers, a private pool, capture scope and parameter snapshots. This is not a
resumable FX executor: one serving scope must return through every P subgraph.
"""

from collections import Counter
from contextlib import contextmanager
from threading import get_ident
from typing import Any

# Diagnostic limits, not model execution limits. A truncated manifest is
# explicitly incomplete and cannot establish whole-model ownership on hardware.
_MAX_DIAGNOSTIC_WRAPPERS = 256
_MAX_DIAGNOSTIC_ENTRIES = 16


def graph_pool_identity(pool):
    """Describe a pool without tensor reads or arbitrary object repr calls.

    The pinned TorchNPU API returns a two-integer pool handle. Its values, not
    the Python tuple identity, distinguish native pools within one worker.
    Other types retain an explicitly weaker process-local object identity.
    """
    if pool is None:
        return None
    if (
        isinstance(pool, tuple)
        and len(pool) == 2
        and all(type(value) is int for value in pool)
    ):
        return {"kind": "native_handle_pair", "values": list(pool)}
    return {
        "kind": "python_object_identity_only",
        "type": type(pool).__module__ + "." + type(pool).__qualname__,
        "object_id": hex(id(pool)),
    }


def graph_wrapper_snapshot(wrapper):
    if wrapper is None:
        return None
    mode = wrapper.runtime_mode
    entries = wrapper.concrete_aclgraph_entries
    ordered = sorted(entries.items(), key=lambda item: item[0].num_tokens)
    return {
        "wrapper_id": hex(id(wrapper)),
        "runtime_mode": getattr(mode, "name", str(mode)),
        "pool": graph_pool_identity(wrapper.graph_pool),
        "entry_count": len(entries),
        "entries_complete": len(entries) <= _MAX_DIAGNOSTIC_ENTRIES,
        "entries": [
            {
                "num_tokens": desc.num_tokens,
                "num_reqs": desc.num_reqs,
                "uniform": desc.uniform,
                "captured": entry.aclgraph is not None,
            }
            for desc, entry in ordered[:_MAX_DIAGNOSTIC_ENTRIES]
        ],
    }


class ShortPrefillGraphs:
    def __init__(self, runtime: Any, *, assert_lane, limit: int):
        self.runtime = runtime
        self.assert_lane = assert_lane
        self.limit = limit
        self._thread = get_ident()
        self._phase = "new"
        self._wrappers: set[Any] = set()
        self._baseline: set[Any] = set()
        self._pool = None
        self._original_call = None
        self._installed_call = None
        self._current: Counter = Counter()
        self.stats: Counter = Counter()

    @property
    def captured(self):
        return self._phase in ("ready", "serving")

    def _check(self):
        if get_ident() != self._thread:
            raise RuntimeError("Short P graphs require their owner thread")
        if self._phase in ("failed", "closed"):
            raise RuntimeError("Short P graph owner is " + self._phase)
        if self._installed_call is not None and (
            self.runtime.wrapper_cls.__call__ is not self._installed_call
        ):
            raise RuntimeError("Short P graph call guard was replaced")

    @contextmanager
    def profile(self):
        self._check()
        if self._phase != "new":
            raise RuntimeError("Short P compilation preparation is startup-only")
        self.assert_lane()
        self._phase = "profiling"
        self._baseline = set(self.runtime.wrappers())
        try:
            yield
            self._wrappers = {
                wrapper
                for wrapper in self.runtime.wrappers()
                if wrapper not in self._baseline
                and wrapper.runtime_mode == self.runtime.piecewise_mode
            }
            if not self._wrappers:
                raise RuntimeError("Native compilation created no P graph wrappers")
            if any(w.concrete_aclgraph_entries for w in self._wrappers):
                raise RuntimeError("P graphs were captured before pool isolation")
            self._pool = self.runtime.new_pool()
            if any(w.graph_pool == self._pool for w in self._baseline):
                raise RuntimeError("Short P pool aliases a pre-existing graph pool")
            for wrapper in self._wrappers:
                wrapper.graph_pool = self._pool
            self._install_guard()
            self._phase = "prepared"
        except BaseException:
            self._phase = "failed"
            raise

    def _install_guard(self):
        wrapper_cls = self.runtime.wrapper_cls
        if getattr(wrapper_cls, "_muxpd_short_piecewise_owner", None) is not None:
            raise RuntimeError("Another short P graph owner is installed")
        original = wrapper_cls.__call__

        def call(wrapper, *args, **kwargs):
            context = self.runtime.context()
            mode = context.cudagraph_runtime_mode
            if mode != self.runtime.piecewise_mode:
                # Native FULL/NONE bypasses internal P entries. In particular,
                # D's outer FULL capture retains its own capturing flag/pool.
                return original(wrapper, *args, **kwargs)
            self._check()
            self.assert_lane()
            if wrapper not in self._wrappers:
                raise RuntimeError("Unowned PIECEWISE wrapper in a P transaction")
            descriptor = context.batch_descriptor
            if not 0 < descriptor.num_tokens <= self.limit:
                raise RuntimeError("Short P graph descriptor exceeds prepared gears")
            if wrapper.graph_pool != self._pool:
                raise RuntimeError("Short P graph pool changed")
            entry = wrapper.concrete_aclgraph_entries.get(descriptor)
            existed = entry is not None and entry.aclgraph is not None
            if self._phase not in ("capturing", "serving"):
                raise RuntimeError("PIECEWISE call outside a short P graph scope")
            if self._phase == "serving":
                if not existed:
                    raise RuntimeError("Serving-time P graph capture is forbidden")
                if entry.input_addresses != self.runtime.input_addresses(args):
                    raise RuntimeError(
                        "P graph inputs changed their captured addresses"
                    )
            previous = context.capturing
            try:
                result = original(wrapper, *args, **kwargs)
            finally:
                # The pinned wrapper sets True during capture and leaves it
                # behind. Uncaptured attention/GDN must see the original False.
                context.capturing = previous
            entry = wrapper.concrete_aclgraph_entries.get(descriptor)
            if entry is None or entry.aclgraph is None:
                raise RuntimeError("P graph call returned without a captured entry")
            self.stats["replays" if existed else "captures"] += 1
            if self._phase == "serving":
                self._current[wrapper] += 1
            return result

        self._original_call = original
        self._installed_call = call
        wrapper_cls._muxpd_short_piecewise_owner = self
        wrapper_cls.__call__ = call

    @contextmanager
    def capture(self, descriptors):
        self._check()
        if self._phase != "prepared":
            raise RuntimeError("P graph capture is single-use after compilation")
        self.assert_lane()
        descriptors = tuple(descriptors)
        if not descriptors or any(
            d.uniform or d.num_reqs is not None or not 0 < d.num_tokens <= self.limit
            for d in descriptors
        ):
            raise RuntimeError("Invalid independent P capture descriptors")
        before = self.runtime.graph_params_snapshot()
        self._phase = "capturing"
        try:
            yield
            for wrapper in self._wrappers:
                entries = wrapper.concrete_aclgraph_entries
                if set(entries) != set(descriptors) or any(
                    entry.aclgraph is None for entry in entries.values()
                ):
                    raise RuntimeError("P startup did not capture every bounded gear")
            if self.runtime.graph_params_snapshot() != before:
                raise RuntimeError("P capture modified D's global GraphParams")
            self._phase = "ready"
        except BaseException:
            self._phase = "failed"
            raise

    @contextmanager
    def serving(self):
        self._check()
        if self._phase != "ready":
            raise RuntimeError("Short P graphs are not ready or already active")
        self.assert_lane()
        self._phase = "serving"
        self._current.clear()
        try:
            yield
            if set(self._current) != self._wrappers or any(
                count != 1 for count in self._current.values()
            ):
                raise RuntimeError("Whole P did not replay every owned subgraph once")
            self.stats["whole_forwards"] += 1
            self._phase = "ready"
        except BaseException:
            self._phase = "failed"
            raise

    def snapshot(self):
        gears = sorted(
            {
                desc.num_tokens
                for w in self._wrappers
                for desc in w.concrete_aclgraph_entries
            }
        )
        return {
            "phase": self._phase,
            "wrappers": len(self._wrappers),
            "limit": self.limit,
            "captured_gears": gears,
            "private_pool": self._pool is not None,
            "pool": graph_pool_identity(self._pool),
            "wrapper_metadata_complete": (
                len(self._wrappers) <= _MAX_DIAGNOSTIC_WRAPPERS
            ),
            "wrapper_metadata": [
                graph_wrapper_snapshot(wrapper)
                for wrapper in sorted(self._wrappers, key=id)[:_MAX_DIAGNOSTIC_WRAPPERS]
            ],
            **self.stats,
        }

    def pool_matches(self, pool):
        if self._pool is None or pool is None:
            return None
        return self._pool == pool

    def close(self):
        """Caller must first join P/D/update streams; no native globals cleared."""
        if self._phase == "closed":
            return
        if get_ident() != self._thread:
            raise RuntimeError("Short P graphs require their owner thread")
        if self._phase in ("serving", "capturing", "profiling"):
            raise RuntimeError("Cannot close an active P graph scope")
        wrapper_cls = self.runtime.wrapper_cls
        if self._installed_call is not None:
            if wrapper_cls.__call__ is self._installed_call:
                wrapper_cls.__call__ = self._original_call
            if getattr(wrapper_cls, "_muxpd_short_piecewise_owner", None) is self:
                delattr(wrapper_cls, "_muxpd_short_piecewise_owner")
        for wrapper in self._wrappers:
            wrapper.concrete_aclgraph_entries.clear()
        self._wrappers.clear()
        self._baseline.clear()
        self._current.clear()
        self._installed_call = self._original_call = None
        self._pool = None
        self._phase = "closed"
