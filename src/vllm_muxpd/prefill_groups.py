"""Experimental P-only groups: startup AOT, serving direct fixed handles.

No torch import at module scope. CPU contract fixtures inject a runtime; real
startup compilation remains an explicit device action performed by the worker.
"""

from __future__ import annotations

import copy
import os
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace
from typing import Any

_MISSING = object()


def clone_group_config(config: Any, group: int, cache_root: str) -> Any:
    """Copy compiler options, never clone modules in static_forward_context."""
    result = copy.copy(config)
    compilation = copy.copy(config.compilation_config)
    for name, value in vars(compilation).items():
        setattr(
            compilation,
            name,
            dict(value) if name == "static_forward_context" else copy.deepcopy(value),
        )
    compilation.traced_files = set()
    compilation.cache_dir = str(Path(cache_root) / f"prefill-group-{group:02d}")
    compilation.local_cache_dir = ""
    compilation.cudagraph_mode = type(compilation.cudagraph_mode).NONE
    compilation.cudagraph_capture_sizes = []
    compilation.max_cudagraph_capture_size = 0
    compilation.compile_sizes = []
    compilation.compile_ranges_endpoints = [
        config.scheduler_config.max_num_batched_tokens
    ]
    compilation.cudagraph_copy_inputs = False
    compilation.use_inductor_graph_partition = False
    result.compilation_config = compilation
    return result


def _runtime():
    import torch
    from vllm import envs
    from vllm.compilation.decorators import maybe_use_cudagraph_partition_wrapper
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import get_forward_context

    from vllm_muxpd.prefill_group_models import GROUP_CLASSES

    if not envs.VLLM_USE_AOT_COMPILE:
        raise ValueError("Prefill groups require native AOT compilation")
    return SimpleNamespace(
        classes=GROUP_CLASSES,
        context=get_forward_context,
        config_context=set_current_vllm_config,
        partition_context=maybe_use_cudagraph_partition_wrapper,
        cache_root=str(Path(envs.VLLM_CACHE_ROOT) / "muxpd-prefill-groups-v1"),
        tensor_type=torch.Tensor,
    )


class _PrefillModelProxy:
    def __init__(self, executor, model):
        self.executor, self._original_model = executor, model

    def __call__(self, **kwargs):
        return self.executor.forward(**kwargs)

    def __getattr__(self, name):
        return getattr(self._original_model, name)


class PrefillGroupExecutor:
    """Own compiled group handles while sharing only the actual model layers.

    The worker provides lane context and group-return checkpoint callbacks.
    Original NPUModelRunner._model_forward still owns its complete call tail.
    A proxy is installed on the P runner only while that call is in progress;
    a suspended coroutine retains the proxy until forward returns or unwinds.
    """

    def __init__(
        self,
        owner,
        backbone,
        *,
        checkpoint,
        lane_check,
        startup_scope,
        is_startup,
        runtime=None,
    ):
        self.owner, self.backbone = owner, backbone
        self.runtime = runtime or _runtime()
        self.checkpoint, self.lane_check = checkpoint, lane_check
        self.startup_scope = startup_scope
        self.is_startup = is_startup
        self.thread = get_ident()
        self.ready = self.failed = self.closed = self.preparing = False
        self.active = False
        self.groups, self.handles, self.patches = [], (), []
        self.calls = [0] * 10
        self.max_tokens = owner.vllm_config.scheduler_config.max_num_batched_tokens
        if (
            len(backbone.layers) != 40
            or backbone.start_layer != 0
            or backbone.end_layer != 40
        ):
            raise ValueError("Prefill groups require the complete 40-layer PP1 model")
        if getattr(backbone, "aux_hidden_state_layers", ()):
            raise ValueError("Prefill groups do not support auxiliary hidden states")
        self.layers = tuple(backbone.layers)
        if len(self.runtime.classes) != 10:
            raise ValueError("Ten distinct group entry points are required")
        codes = [cls.forward.__code__ for cls in self.runtime.classes]
        names = [cls.forward.__qualname__ for cls in self.runtime.classes]
        if len(set(codes)) != 10 or len(set(names)) != 10:
            raise ValueError("Prefill group code identities are not isolated")
        for index, cls in enumerate(self.runtime.classes):
            config = clone_group_config(
                owner.vllm_config, index, self.runtime.cache_root
            )
            with self.runtime.config_context(config):
                self.groups.append(
                    cls(
                        layers=self.layers[index * 4 : index * 4 + 4],
                        vllm_config=config,
                    )
                )

    def _check(self):
        if get_ident() != self.thread or self.closed or self.failed:
            raise RuntimeError("Prefill group executor is unavailable or wrong-thread")

    def _patch(self, name, replacement):
        previous = vars(self.owner).get(name, _MISSING)
        setattr(self.owner, name, replacement)
        self.patches.append((name, previous, replacement))

    def install(self):
        self._check()
        if self.patches:
            raise RuntimeError("Prefill groups already installed")
        model_forward, profile = self.owner._model_forward, self.owner.profile_run

        @wraps(model_forward)
        def grouped_forward(*args, **kwargs):
            self._check()
            if self.active:
                raise RuntimeError("Nested P group forward")
            original = self.owner.model
            proxy = _PrefillModelProxy(self, original)
            self.owner.model = proxy
            self.active = True
            try:
                # Native compile_or_warm_up can run extra owner dummy forwards
                # outside profile_run. Scope just that P call, not D capture:
                # D capture must retain caller lane routing to native groups.
                scope = (
                    self.startup_scope()
                    if not self.preparing and self.is_startup()
                    else nullcontext()
                )
                with scope:
                    return model_forward(*args, **kwargs)
            except Exception:
                self.failed = True
                raise
            finally:
                # Greenlet cancellation also unwinds the proxy without
                # poisoning future cohorts; the worker joins before its ACK.
                self.active = False
                if self.owner.model is not proxy:
                    self.failed = True
                    raise RuntimeError("P model pointer changed during group forward")
                self.owner.model = original

        @wraps(profile)
        def profile_groups(*args, **kwargs):
            self._check()
            if self.ready or self.preparing:
                raise RuntimeError("P group startup profiling cannot repeat")
            old_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", _MISSING)
            try:
                self.preparing = True
                with self.startup_scope():
                    result = profile(*args, **kwargs)
                self.handles = tuple(
                    getattr(group, "aot_compiled_fn", None) for group in self.groups
                )
                if not all(callable(fn) for fn in self.handles) or not all(self.calls):
                    raise RuntimeError("Startup did not execute all ten AOT groups")
                self.ready = True
                return result
            except BaseException:
                self.failed = True
                raise
            finally:
                self.preparing = False
                if old_cache is _MISSING:
                    os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
                else:
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = old_cache

        self._patch("_model_forward", grouped_forward)
        self._patch("profile_run", profile_groups)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        self._check()
        self.lane_check()
        ctx = self.runtime.context()
        if (
            getattr(ctx.cudagraph_runtime_mode, "name", None) != "NONE"
            or ctx.capturing
            or not ctx.skip_compiled
        ):
            raise RuntimeError(
                "Group P requires NONE, noncapturing, skip_compiled context"
            )
        if (
            not (self.preparing or self.ready)
            or intermediate_tensors is not None
            or kwargs
        ):
            raise RuntimeError("Unsupported or unprepared P group forward")
        if positions is None or not 1 <= positions.shape[-1] <= self.max_tokens:
            raise ValueError("P token shape is outside the prepared dynamic range")
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else self.backbone.embed_input_ids(input_ids)
        )
        residual = None
        for index, group in enumerate(self.groups):
            self.lane_check()
            with self.runtime.config_context(group.vllm_config):
                if self.preparing:
                    previous = ctx.skip_compiled
                    try:
                        ctx.skip_compiled = False
                        hidden, residual = group(positions, hidden, residual)
                    finally:
                        ctx.skip_compiled = previous
                    handle = getattr(group, "aot_compiled_fn", None)
                    if not callable(handle):
                        raise RuntimeError(
                            "Native group startup did not create AOT handle"
                        )
                else:
                    handle = self.handles[index]
                    if group.aot_compiled_fn is not handle:
                        raise RuntimeError("P AOT handle changed after startup")
                    with self.runtime.partition_context(group.vllm_config):
                        hidden, residual = handle(group, positions, hidden, residual)
            self.calls[index] += 1
            if not self.preparing:
                self.checkpoint(index, final=index == 9)
        hidden, _ = self.backbone.norm(hidden, residual)
        return hidden

    def snapshot(self):
        return {
            "ready": self.ready,
            "failed": self.failed,
            "closed": self.closed,
            "group_size": 4,
            "groups": len(self.groups),
            "calls": list(self.calls),
            "entry_qualnames": [
                type(group).forward.__qualname__ for group in self.groups
            ],
            "cache_directories": [
                group.compilation_config.cache_dir for group in self.groups
            ],
            "evidence": "successful AOT group returns; not device completion",
        }

    def close(self):
        if self.closed:
            return
        if get_ident() != self.thread or self.active or self.preparing:
            raise RuntimeError("Cannot close active P groups")
        while self.patches:
            name, original, installed = self.patches[-1]
            if vars(self.owner).get(name, _MISSING) is not installed:
                raise RuntimeError("Foreign P group instance method replacement")
            if original is _MISSING:
                delattr(self.owner, name)
            else:
                setattr(self.owner, name, original)
            self.patches.pop()
        self.handles = ()
        self.groups.clear()
        self.layers = ()
        self.backbone = self.owner = None
        self.closed = True
