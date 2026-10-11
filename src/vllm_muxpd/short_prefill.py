"""Bounded startup preparation for the experimental whole-short-P FX path.

This module imports no device runtime. It does not compile anything: the native
startup profile must already have produced the dynamic AOT callable. Native
opaque operators retain their own first-use kernel initialization behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def warmup_sizes(compilation: Any, limit: int) -> tuple[int, ...]:
    """Exercise every native dispatch entry that a short P may select.

    PiecewiseBackend prefers exact compile_sizes over interval entries. Warm a
    representative outside exact sizes for each reachable interval as well as
    every exact entry. Calling just the largest shape misses those lazy Python
    code materializations in npugraph_ex._CompiledFxGraph.
    """
    ranges = sorted(compilation.get_compile_ranges(), key=lambda item: item.start)
    exact = set(compilation.compile_sizes or ())
    if any(type(size) is not int or size < 1 for size in exact):
        raise ValueError("Short P requires resolved positive compile_sizes")
    sizes = {size for size in exact if size <= limit}
    covered = 0
    for interval in ranges:
        if interval.start != covered + 1 or interval.end < interval.start:
            raise ValueError("Short P compile ranges must cover contiguous token sizes")
        covered = interval.end
        candidate = interval.start
        end = min(limit, interval.end)
        while candidate in exact and candidate <= end:
            candidate += 1
        if candidate <= end:
            sizes.add(candidate)
        if covered >= limit:
            break
    if covered < limit:
        raise ValueError("Short P threshold exceeds prepared dynamic compile ranges")
    # Bound startup work even if an external compilation config enumerates an
    # excessive number of specializations. No serving-time warmup is allowed.
    if not sizes or len(sizes) > 64:
        raise ValueError("Short P supports at most 64 startup FX dispatch entries")
    return tuple(sorted(sizes))


class PreparedShortPrefill:
    """Observe the existing AOT callable; never fall back to Dynamo compilation."""

    def __init__(self, model: Any, active: Callable[[], bool]) -> None:
        dims = getattr(model, "_dynamic_arg_dims", {})
        if dims.get("input_ids") != 0 or dims.get("positions") != -1:
            raise RuntimeError("Short P requires Qwen's dynamic token/position inputs")
        original = getattr(model, "aot_compiled_fn", None)
        if not callable(original) or getattr(model, "do_not_compile", False):
            raise RuntimeError(
                "Short P requires an already prepared dynamic AOT callable"
            )
        self.model = model
        self.original = original
        self.active = active
        self.successful_calls = 0

        def observed(*args: Any, **kwargs: Any) -> Any:
            is_prefill = self.active()
            result = original(*args, **kwargs)
            if is_prefill:
                self.successful_calls += 1
            return result

        self.observed = observed

    def check_installed(self) -> None:
        if getattr(self.model, "aot_compiled_fn", None) is not self.observed:
            raise RuntimeError("Prepared short-P AOT callable changed after startup")

    def require_return(self, before: int) -> None:
        self.check_installed()
        if self.successful_calls != before + 1:
            raise RuntimeError(
                "Short P did not return through its prepared AOT callable"
            )
