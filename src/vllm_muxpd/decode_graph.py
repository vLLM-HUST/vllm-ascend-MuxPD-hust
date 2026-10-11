"""Opt-in decode-only ACL graph helpers; importing this module touches no NPU.

The worker must validate its existing model/backend whitelist as well as
``validate_decode_graph_config``. Install P guards before KV initialization,
attach the shared model after loading, and redirect native owner capture only
during startup. This module intentionally does not enable the worker path.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from threading import get_ident
from types import SimpleNamespace
from typing import Any

from vllm_muxpd.piecewise_prefill import (
    ShortPrefillGraphs,
    graph_wrapper_snapshot,
)

_MISSING = object()
_CAPTURE_OWNER: object | None = None


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _mode_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def validate_decode_graph_config(config: Any, policy: Any) -> None:
    """Validate graph-specific restrictions without importing torch or vLLM."""
    errors = []
    if _value(policy, "decode_graph") is not True:
        errors.append("decode_graph must be explicitly true")
    if _value(policy, "execution") != "decoupled":
        errors.append("execution must be decoupled")
    stream_mode = _value(policy, "stream_mode", "fenced")
    if stream_mode not in ("fenced", "overlap"):
        errors.append("stream_mode must be fenced or overlap")
    if stream_mode == "overlap" and (
        _value(policy, "overlap_communication", "shared") != "isolated"
        or _value(policy, "compile_full_prefill", False)
    ):
        errors.append(
            "experimental graph overlap requires isolated communication "
            "and raw prefill (compile_full_prefill=False)"
        )
    if type(_value(policy, "compile_full_prefill", False)) is not bool:
        errors.append("compile_full_prefill must be a boolean")
    short_limit = _value(policy, "piecewise_short_prefill_tokens", 0)
    if type(short_limit) is not int or short_limit < 0:
        errors.append("piecewise_short_prefill_tokens must be a nonnegative integer")
        short_limit = 0  # Continue collecting errors without invalid arithmetic.
    if short_limit and not (
        stream_mode == "overlap"
        and _value(policy, "overlap_communication") == "isolated"
        and not _value(policy, "compile_full_prefill", False)
    ):
        errors.append("short P graphs require isolated overlap without full P FX")
    model = config.model_config
    compilation = config.compilation_config
    scheduler = config.scheduler_config
    if model.enforce_eager:
        errors.append("enforce_eager must be false for the decode graph opt-in")
    expected_mode = "FULL_AND_PIECEWISE" if short_limit else "FULL_DECODE_ONLY"
    if _mode_name(compilation.cudagraph_mode) != expected_mode:
        errors.append("cudagraph_mode must be " + expected_mode)
    if _mode_name(compilation.mode) != "VLLM_COMPILE":
        errors.append("compilation mode must be VLLM_COMPILE")
    sizes = compilation.cudagraph_capture_sizes
    if (
        not isinstance(sizes, (list, tuple))
        or not sizes
        or any(
            type(n) is not int or not 1 <= n <= (short_limit or scheduler.max_num_seqs)
            for n in sizes
        )
        or list(sizes) != sorted(set(sizes))
    ):
        errors.append(
            "capture sizes must be sorted unique integers within max_num_seqs"
        )
    elif compilation.max_cudagraph_capture_size != max(sizes):
        errors.append("max_cudagraph_capture_size must match the largest capture size")
    if short_limit:
        if short_limit > scheduler.max_num_batched_tokens:
            errors.append("short P threshold exceeds complete token admission")
        if (
            not sizes
            or max(sizes) != short_limit
            or scheduler.max_num_seqs not in sizes
            or len(sizes) > 16
        ):
            errors.append(
                "short P requires <=16 gears ending at its threshold and "
                "including max_num_seqs for independent D capture"
            )
        if getattr(compilation, "cudagraph_copy_inputs", False):
            errors.append("shared compiler input-copy buffers are forbidden")
        if getattr(compilation, "use_inductor_graph_partition", False):
            errors.append("short P requires native explicit FX attention splitting")
    if getattr(config, "use_v2_model_runner", False):
        errors.append("only the V1 model runner is supported")
    if scheduler.async_scheduling is not False:
        errors.append("async_scheduling must be false")
    for name in ("speculative_config", "lora_config", "weight_transfer_config"):
        if getattr(config, name, None) is not None:
            errors.append(f"{name} is unsupported by decode graphs")
    offload = getattr(config, "offload_config", None)
    if _value(_value(offload, "uva"), "cpu_offload_gb", 0) or _value(
        _value(offload, "prefetch"), "offload_group_size", 0
    ):
        errors.append("model offloading is unsupported by decode graphs")
    if errors:
        raise ValueError("MuxPD decode graph: " + "; ".join(errors))


@dataclass(frozen=True)
class GraphLaneState:
    forward_context: Any
    stream: Any
    workspace_manager: Any


class GraphFencedExecutionBackend:
    """Full-slice fences plus lane-local workspace ownership.

    Give P the already-profiled native manager and D a new manager. Capture and
    replay must use that same D manager; native capture locks the current one.
    Dependencies are injectable so lifecycle tests need no device runtime.
    """

    def __init__(
        self,
        *,
        runtime: Any = None,
        forward_context_module: Any = None,
        workspace_module: Any = None,
        ascend_utils_module: Any = None,
    ) -> None:
        if runtime is None:
            from .overlap import TorchNPURuntime

            runtime = TorchNPURuntime()
        if forward_context_module is None:
            import vllm.forward_context as forward_context_module
        if workspace_module is None:
            import vllm.v1.worker.workspace as workspace_module
        if ascend_utils_module is None:
            import vllm_ascend.utils as ascend_utils_module

        self.runtime = runtime
        self._forward = forward_context_module
        self._workspace = workspace_module
        self._ascend = ascend_utils_module
        self._thread = get_ident()
        self._tail: Any = None
        self._tail_stream: Any = None

    def _check_thread(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("MuxPD decode graph requires one host thread")

    def capture(self) -> GraphLaneState:
        self._check_thread()
        return GraphLaneState(
            self._forward._forward_context,
            self.runtime.current_stream(),
            self._workspace._manager,
        )

    def restore(self, state: GraphLaneState) -> None:
        self._check_thread()
        self._forward._forward_context = state.forward_context
        self.runtime.set_stream(state.stream)
        self._ascend._CURRENT_STREAM = state.stream
        self._workspace._manager = state.workspace_manager
        if self._tail is not None and state.stream != self._tail_stream:
            self.runtime.wait_event(state.stream, self._tail)

    def fence(self) -> None:
        self._check_thread()
        stream = self.runtime.current_stream()
        self._tail = self.runtime.record_event(stream)
        # Same-stream FIFO already orders that tail. Keep every newest event so
        # the next cross-stream restore still waits for all submitted work.
        self._tail_stream = stream


@contextmanager
def fixed_decode_capture_context(
    *,
    npu: Any,
    npu_runner_module: Any,
    gpu_runner_module: Any,
    workspace_module: Any,
    ascend_utils_module: Any,
    stream: Any,
    workspace: Any,
):
    """Bind both native capture layers to D and restore all startup patches.

    Native NPUModelRunner patches the GPU module's factory and leaves its own
    factory there on exit. Separately torch.npu.graph uses a class-level default
    capture stream, not the current stream. Both need explicit restoration.
    """
    global _CAPTURE_OWNER
    if _CAPTURE_OWNER is not None:
        raise RuntimeError("MuxPD cannot nest or concurrently run graph capture")
    original_npu_factory = npu_runner_module.graph_capture
    original_gpu_factory = gpu_runner_module.graph_capture
    original_graph_stream = npu.graph.default_capture_stream
    original_workspace = workspace_module._manager
    original_ascend_stream = ascend_utils_module._CURRENT_STREAM
    _CAPTURE_OWNER = object()

    @contextmanager
    def graph_capture(device):
        if device != stream.device:
            raise ValueError("Decode graph capture device differs from its stream")
        current = npu.current_stream()
        if current != stream:
            stream.wait_stream(current)
        with npu.stream(stream):
            yield SimpleNamespace(stream=stream)

    try:
        npu_runner_module.graph_capture = graph_capture
        gpu_runner_module.graph_capture = graph_capture
        npu.graph.default_capture_stream = stream
        workspace_module._manager = workspace
        ascend_utils_module._CURRENT_STREAM = stream
        current = npu.current_stream()
        if current != stream:
            stream.wait_stream(current)
        with npu.stream(stream):
            yield
    finally:
        npu_runner_module.graph_capture = original_npu_factory
        gpu_runner_module.graph_capture = original_gpu_factory
        npu.graph.default_capture_stream = original_graph_stream
        workspace_module._manager = original_workspace
        ascend_utils_module._CURRENT_STREAM = original_ascend_stream
        _CAPTURE_OWNER = None


class _NativeGraphRuntime:
    """Native imports happen only when a worker explicitly constructs helpers."""

    def __init__(self) -> None:
        import torch
        import vllm.forward_context as forward
        import vllm.v1.worker.workspace as workspace
        import vllm_ascend.compilation.acl_graph as graphs
        import vllm_ascend.utils as ascend
        import vllm_ascend.worker.model_runner_v1 as runner
        from vllm.config import CUDAGraphMode

        self.npu = torch.npu
        self._tensor_type = torch.Tensor
        self.forward = forward
        self.workspace = workspace
        self.ascend = ascend
        self.graphs = graphs
        self.runner = runner
        self.none_mode = CUDAGraphMode.NONE
        self.full_mode = CUDAGraphMode.FULL
        self.piecewise_mode = CUDAGraphMode.PIECEWISE
        self.full_decode_mode = CUDAGraphMode.FULL_DECODE_ONLY
        self.wrapper_cls = graphs.ACLGraphWrapper

    def context(self):
        return self.forward.get_forward_context()

    def wrappers(self):
        return tuple(self.graphs._acl_graph_wrappers)

    def new_pool(self):
        return self.npu.graph_pool_handle()

    def input_addresses(self, args):
        return [x.data_ptr() for x in args if isinstance(x, self._tensor_type)]

    def graph_params_snapshot(self):
        params = self.graph_params()
        if params is None:
            return None
        return (
            id(params),
            tuple(
                (
                    name,
                    tuple(
                        (key, tuple(id(x) for x in values))
                        for key, values in sorted(getattr(params, name).items())
                    ),
                )
                for name in ("events", "handles", "attn_params")
            ),
            tuple(
                (key, None if value is None else value.data_ptr())
                for key, value in sorted(params.workspaces.items())
            ),
        )

    def validate_piecewise_backend(self, owner):
        compilation = owner.ascend_config.ascend_compilation_config
        if compilation.enable_npugraph_ex or compilation.enable_static_kernel:
            raise ValueError("P PIECEWISE requires the native fusion-pass backend")
        if not owner.compilation_config.is_attention_compiled_piecewise():
            raise ValueError("P PIECEWISE must split every native attention/GDN op")

    def guard_capture_disabled(self):
        from vllm.compilation.monitor import set_cudagraph_capturing_enabled

        set_cudagraph_capturing_enabled(False)

    def graph_params(self) -> Any:
        return self.graphs.get_graph_params()

    def unwrap(self, model: Any) -> Any:
        if isinstance(model, self.graphs.ACLGraphWrapper):
            return model.unwrap()
        return model

    def wrap(self, model: Any, decode_runner: Any) -> Any:
        return self.graphs.ACLGraphWrapper(
            model,
            decode_runner.vllm_config,
            runtime_mode=self.full_mode,
            use_eagle=False,
            enable_enpu=False,
        )

    def new_stream(self, device: Any) -> Any:
        return self.npu.Stream(device=device)

    def capture_context(self, decode_runner: Any, stream: Any, workspace: Any):
        import sys

        parent = self.runner._get_gpu_model_runner_module_name(decode_runner)
        return fixed_decode_capture_context(
            npu=self.npu,
            npu_runner_module=self.runner,
            gpu_runner_module=sys.modules[parent],
            workspace_module=self.workspace,
            ascend_utils_module=self.ascend,
            stream=stream,
            workspace=workspace,
        )


class DecodeGraphController:
    """Coordinate one startup-only graph owner without changing worker classes.

    ``runtime`` is a small injectable adapter: none_mode/full_mode, forward,
    graph_params(), unwrap(), wrap(), new_stream(), capture_context(). Native
    defaults are lazy. Existing worker validation remains mandatory.
    """

    def __init__(
        self,
        owner: Any,
        decode_runner: Any,
        decode_stream: Any,
        decode_workspace: Any,
        *,
        runtime: Any = None,
        policy: Any = None,
        prefill_startup_scope: Any = None,
        assert_prefill_lane: Any = None,
    ) -> None:
        config = owner.vllm_config
        if policy is None:
            policy = config.scheduler_config.batch_admission_policy_config or {}
        validate_decode_graph_config(config, policy)
        if owner is decode_runner or decode_workspace is None:
            raise ValueError("Decode graph requires a separate runner and workspace")
        for runner in (owner, decode_runner):
            if getattr(runner, "enable_enpu", False):
                raise ValueError("Decode graph does not support ENPU")
            ascend = getattr(runner, "ascend_config", None)
            if _value(_value(ascend, "xlite_graph_config"), "enabled", False):
                raise ValueError("Decode graph does not support xlite")
        self.owner = owner
        self.decode = decode_runner
        self.stream = decode_stream
        self.workspace = decode_workspace
        self.runtime = runtime or _NativeGraphRuntime()
        self._compile_full_prefill = _value(policy, "compile_full_prefill", False)
        self.piecewise_short_prefill_tokens = _value(
            policy, "piecewise_short_prefill_tokens", 0
        )
        self._prefill_startup_scope = prefill_startup_scope
        self._piecewise = None
        self._prefill_capture_active = False
        self._prefill_capture_model = owner.capture_model
        if self.piecewise_short_prefill_tokens:
            if prefill_startup_scope is None or assert_prefill_lane is None:
                raise ValueError("P graphs require explicit P startup/lane ownership")
            self.runtime.validate_piecewise_backend(owner)
            self._piecewise = ShortPrefillGraphs(
                self.runtime,
                assert_lane=assert_prefill_lane,
                limit=self.piecewise_short_prefill_tokens,
            )
        self._compile_prefill_active = False
        self._prefill_profiled = False
        self._thread = get_ident()
        self._installed = False
        self._attached = False
        self._redirecting = False
        self._capture_started = False
        self._captured = False
        self._failed = False
        self._closed = False
        self._patches: list[tuple[Any, str, Any, Any]] = []
        self._decode_graph_model: Any = None
        self._statistics = {
            "decode_replay_calls": 0,
            "decode_capture_forward_calls": 0,
            "decode_eager_fallback_calls": 0,
            "decode_uncaptured_full_calls": 0,
            "decode_forward_failures": 0,
            "prefill_forced_eager_calls": 0,
            "prefill_compiled_calls": 0,
            "prefill_compile_profile_runs": 0,
            "prefill_compile_profile_failures": 0,
            "capture_attempts": 0,
            "capture_successes": 0,
            "capture_failures": 0,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return independent audit data, including startup warmup forwards.

        Decode forward counters describe successful calls; failed forwards have
        a separate counter. ``decode_uncaptured_full_calls`` is a subset of
        fallback calls: FULL alone does not prove that a captured graph ran.
        Prefill counts attempts after its eager guard has been applied. To
        measure serving calls, subtract the startup snapshot from the final one.
        The snapshot remains available after a failed startup for diagnosis.
        """
        entries = getattr(
            getattr(self.decode, "model", None), "concrete_aclgraph_entries", {}
        )
        return {
            "enabled": True,
            "attached": self._attached,
            "capture_started": self._capture_started,
            "captured": self._captured,
            "failed": self._failed,
            "closed": self._closed,
            "compile_full_prefill": self.compile_full_prefill,
            "piecewise_short_prefill_tokens": self.piecewise_short_prefill_tokens,
            "prefill_piecewise": (
                self._piecewise.snapshot() if self._piecewise else None
            ),
            "decode_graph_metadata": (
                graph_wrapper_snapshot(self._decode_graph_model)
                if self._piecewise
                else None
            ),
            "prefill_decode_pool_equal": (
                self._piecewise.pool_matches(
                    getattr(self._decode_graph_model, "graph_pool", None)
                )
                if self._piecewise
                else None
            ),
            "prefill_compile_profiled": self._prefill_profiled,
            "includes_startup_warmups": True,
            "captured_graph_entries": sum(
                getattr(entry, "aclgraph", None) is not None
                for entry in entries.values()
            ),
            **self._statistics,
        }

    @property
    def compile_full_prefill(self) -> bool:
        return self._compile_full_prefill

    @property
    def _compile_prefill_enabled(self):
        return bool(self.compile_full_prefill or self._piecewise)

    @contextmanager
    def _prefill_compile_context(self, *, startup: bool):
        self._check()
        if not self._compile_prefill_enabled or not self._attached:
            raise RuntimeError("Full prefill compilation is not enabled and attached")
        if self._compile_prefill_active:
            raise RuntimeError("Full prefill compilation contexts cannot be nested")
        if startup:
            if self._capture_started:
                raise RuntimeError("Prefill compilation profiling must precede capture")
        elif not self._prefill_profiled or not self._captured:
            raise RuntimeError(
                "Full prefill compilation requires startup profile and capture"
            )
        self._compile_prefill_active = True
        try:
            yield
        finally:
            self._compile_prefill_active = False

    @contextmanager
    def full_prefill_compile_context(self):
        """Allow FX for one fresh, complete, non-yielding prefill forward.

        The worker validates the full layer quota and successful native return.
        Legacy FX keeps NONE; short PIECEWISE uses only its private prepared
        entries. Neither path may capture or replay the decode ACL graph.
        """
        try:
            with self._prefill_compile_context(startup=False):
                scope = self._piecewise.serving() if self._piecewise else nullcontext()
                with scope:
                    yield
        except BaseException:
            if self._piecewise:
                self._failed = True
            raise

    def _check(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("MuxPD decode graph requires one host thread")
        if self._closed:
            raise RuntimeError("Decode graph controller is closed")
        if self._failed:
            raise RuntimeError("Decode graph capture failed; restart the worker")

    def _patch_instance(self, target: Any, name: str, replacement: Any) -> None:
        original = vars(target).get(name, _MISSING)
        setattr(target, name, replacement)
        self._patches.append((target, name, original, replacement))

    def close(self) -> None:
        """Release this lane after its streams have been synchronized.

        The worker joins P/D/update streams before calling this method. Only
        D's graph wrapper and instance patches belong to this controller: the
        native owner must subsequently release shared model weights and KV.
        Never call the decode runner's native shutdown on the shared cache.
        Available even after partial initialization or failed graph capture.
        """
        if get_ident() != self._thread:
            raise RuntimeError("MuxPD decode graph requires one host thread")
        if self._closed:
            return
        if self._redirecting:
            raise RuntimeError("Cannot close during owner capture redirection")
        if self._compile_prefill_active:
            raise RuntimeError("Cannot close during full prefill compilation")
        self._closed = True
        try:
            if self._piecewise is not None:
                self._piecewise.close()
            wrapper = self._decode_graph_model
            if wrapper is not None:
                # Drop graph objects before the native owner frees captured KV.
                wrapper.concrete_aclgraph_entries.clear()
                if self.decode.model is wrapper:
                    self.decode.model = None
        finally:
            try:
                for target, name, original, installed in reversed(self._patches):
                    # Preserve a later external override; do not clobber it.
                    if vars(target).get(name, _MISSING) is not installed:
                        continue
                    if original is _MISSING:
                        delattr(target, name)
                    else:
                        setattr(target, name, original)
            finally:
                self._patches.clear()
                self._decode_graph_model = None
                self._prefill_capture_model = None
                self._prefill_startup_scope = None
                self._attached = False
                self.owner = None
                self.decode = None
                self.stream = None
                self.workspace = None
                self.runtime = None

    def install_prefill_guards(self) -> None:
        """Install before owner KV init; never mutate shared compilation config."""
        self._check()
        if self._installed or hasattr(self.owner, "_muxpd_decode_graph_controller"):
            raise RuntimeError("Decode graph prefill guards are already installed")
        if self.runtime.graph_params() is not None:
            raise RuntimeError("GraphParams already has an owner")
        dispatch = self.owner._determine_batch_execution_and_padding
        dispatch_signature = signature(dispatch)
        model_forward = self.owner._model_forward
        profile_run = None
        if self._compile_prefill_enabled:
            profile_run = getattr(self.owner, "profile_run", None)
            if not callable(profile_run):
                raise RuntimeError(
                    "Full prefill compilation requires native profile_run"
                )

        @wraps(dispatch)
        def eager_dispatch(*args, **kwargs):
            self._check()
            bound = dispatch_signature.bind(*args, **kwargs)
            if (
                self._piecewise
                and self._compile_prefill_active
                and (self._prefill_capture_active or self._piecewise.captured)
            ):
                # Preserve native NONE warmup requests; actual short P and
                # capture calls may select only P's filtered PIECEWISE keys.
                bound.arguments["force_uniform_decode"] = False
            else:
                bound.arguments["force_eager"] = True
            return dispatch(*bound.args, **bound.kwargs)

        @wraps(model_forward)
        def eager_forward(*args, **kwargs):
            self._check()
            ctx = self.runtime.forward._forward_context
            if ctx is None:
                raise RuntimeError("Prefill forward requires a native forward context")
            values = {
                "cudagraph_runtime_mode": (
                    ctx.cudagraph_runtime_mode
                    if self._piecewise
                    and self._compile_prefill_active
                    and (self._prefill_capture_active or self._piecewise.captured)
                    else self.runtime.none_mode
                ),
                "skip_compiled": not self._compile_prefill_active,
                "capturing": False,
            }
            previous = {name: getattr(ctx, name, _MISSING) for name in values}
            try:
                for name, value in values.items():
                    setattr(ctx, name, value)
                self._statistics["prefill_forced_eager_calls"] += 1
                if self._compile_prefill_active:
                    self._statistics["prefill_compiled_calls"] += 1
                return model_forward(*args, **kwargs)
            finally:
                for name, value in previous.items():
                    if value is _MISSING:
                        delattr(ctx, name)
                    else:
                        setattr(ctx, name, value)

        self._patch_instance(
            self.owner, "_determine_batch_execution_and_padding", eager_dispatch
        )
        self._patch_instance(self.owner, "_model_forward", eager_forward)
        self._patch_instance(self.owner, "use_aclgraph", False)
        self._patch_instance(self.owner, "_muxpd_decode_graph_controller", self)
        if profile_run is not None:

            @wraps(profile_run)
            def compiled_profile_run(*args, **kwargs):
                # Native startup profiles max_num_tokens before KV sizing and
                # D capture. Decoder hooks are inactive in this worker phase.
                self._check()
                before = self._statistics["prefill_compiled_calls"]
                try:
                    startup = (
                        self._prefill_startup_scope()
                        if self._piecewise
                        else nullcontext()
                    )
                    with startup:
                        with self._prefill_compile_context(startup=True):
                            prepare = (
                                self._piecewise.profile()
                                if self._piecewise
                                else nullcontext()
                            )
                            with prepare:
                                result = profile_run(*args, **kwargs)
                    if self._statistics["prefill_compiled_calls"] == before:
                        raise RuntimeError(
                            "Native profile_run did not compile a prefill forward"
                        )
                except BaseException:
                    self._statistics["prefill_compile_profile_failures"] += 1
                    self._failed = True
                    raise
                self._prefill_profiled = True
                self._statistics["prefill_compile_profile_runs"] += 1
                return result

            self._patch_instance(self.owner, "profile_run", compiled_profile_run)
        if self._piecewise:
            self._isolate_dispatcher(self.owner, prefill=True)
            self._isolate_dispatcher(self.decode, prefill=False)
        self._installed = True

    def _isolate_dispatcher(self, runner, *, prefill):
        dispatcher = runner.cudagraph_dispatcher
        initialize = dispatcher.initialize_cudagraph_keys

        @wraps(initialize)
        def isolated_keys(*args, **kwargs):
            result = initialize(*args, **kwargs)
            keep = self.runtime.piecewise_mode if prefill else self.runtime.full_mode
            maximum = (
                self.piecewise_short_prefill_tokens
                if prefill
                else runner.vllm_config.scheduler_config.max_num_seqs
            )
            for mode, keys in dispatcher.cudagraph_keys.items():
                dispatcher.cudagraph_keys[mode] = {
                    desc for desc in keys if mode == keep and desc.num_tokens <= maximum
                }
            dispatcher.cudagraph_mode = (
                self.runtime.piecewise_mode
                if prefill
                else self.runtime.full_decode_mode
            )
            return result

        self._patch_instance(dispatcher, "initialize_cudagraph_keys", isolated_keys)

    def attach_shared_model(self) -> None:
        self._check()
        if not self._installed or self._attached or self._capture_started:
            raise RuntimeError("Attach decode graph model once after installing guards")
        if hasattr(self.decode, "_muxpd_decode_graph_controller"):
            raise RuntimeError("Decode graph forward observer is already installed")
        raw = self.runtime.unwrap(self.owner.model)
        wrapper = self.runtime.wrap(raw, self.decode)
        if wrapper is raw or wrapper is self.owner.model:
            raise RuntimeError("Decode must have its own ACLGraphWrapper")
        update_stream = self.runtime.new_stream(self.stream.device)
        self._decode_graph_model = wrapper
        self.owner.model = raw
        self.decode.model = wrapper
        self._patch_instance(self.decode, "update_stream", update_stream)
        self.decode.model_memory_usage = 0
        for name in ("_has_sinks", "is_multimodal_pruning_enabled"):
            setattr(self.decode, name, getattr(self.owner, name))
        self._install_decode_observer()
        self._attached = True

    def _install_decode_observer(self) -> None:
        model_forward = self.decode._model_forward

        def has_captured_entry(descriptor: Any) -> bool:
            entries = self.decode.model.concrete_aclgraph_entries
            entry = entries.get(descriptor)
            return entry is not None and getattr(entry, "aclgraph", None) is not None

        @wraps(model_forward)
        def observed_forward(*args, **kwargs):
            self._check()
            ctx = self.runtime.forward._forward_context
            full = (
                getattr(ctx, "cudagraph_runtime_mode", None) == self.runtime.full_mode
            )
            capturing = bool(getattr(ctx, "capturing", False))
            descriptor = getattr(ctx, "batch_descriptor", None)
            captured_before = full and has_captured_entry(descriptor)
            try:
                result = model_forward(*args, **kwargs)
            except BaseException:
                self._statistics["decode_forward_failures"] += 1
                raise
            # ACLGraphWrapper sets capturing inside its first FULL call, after
            # this observer starts. A new concrete entry also proves capture if
            # a native implementation restores that flag before returning.
            if (
                capturing
                or getattr(ctx, "capturing", False)
                or (full and not captured_before and has_captured_entry(descriptor))
            ):
                key = "decode_capture_forward_calls"
            elif full and captured_before:
                key = "decode_replay_calls"
            else:
                key = "decode_eager_fallback_calls"
                if full:
                    self._statistics["decode_uncaptured_full_calls"] += 1
            self._statistics[key] += 1
            return result

        self._patch_instance(self.decode, "_model_forward", observed_forward)
        self._patch_instance(self.decode, "_muxpd_decode_graph_controller", self)

    def _validate_capture(self) -> None:
        if not self._attached:
            raise RuntimeError("Decode graph model has not been attached")
        if self._piecewise:
            validate_decode_graph_config(
                self.owner.vllm_config,
                self.owner.vllm_config.scheduler_config.batch_admission_policy_config,
            )
            self.runtime.validate_piecewise_backend(self.owner)
        if self._compile_prefill_enabled and not self._prefill_profiled:
            raise RuntimeError(
                "Full prefill compilation requires native startup profiling"
            )
        if self.runtime.graph_params() is None:
            raise RuntimeError("Decode must initialize GraphParams before capture")
        if self.owner.input_batch is self.decode.input_batch:
            raise RuntimeError("Decode graph input batch aliases prefill")
        for name in ("input_ids", "positions", "seq_lens", "query_start_loc"):
            p = getattr(self.owner, name)
            d = getattr(self.decode, name)
            p, d = getattr(p, "gpu", p), getattr(d, "gpu", d)
            if p is d or (
                hasattr(p, "data_ptr")
                and hasattr(d, "data_ptr")
                and p.data_ptr() == d.data_ptr()
            ):
                raise RuntimeError(f"Decode graph {name} aliases prefill")
        descriptors = [
            (mode, tuple(descs))
            for mode, descs in self.decode.cudagraph_dispatcher.get_capture_descs()
        ]
        if not descriptors or not any(descs for _, descs in descriptors):
            raise RuntimeError("Decode graph has no capture descriptors")
        for mode, descs in descriptors:
            if mode != self.runtime.full_mode or any(not d.uniform for d in descs):
                raise RuntimeError("Decode graph capture must be FULL and uniform")

    def capture_decode(self) -> int:
        """A capture attempt is single-use; partial failure poisons the worker."""
        self._check()
        if self._capture_started:
            raise RuntimeError("Decode graph capture has already been attempted")
        self._validate_capture()
        self._capture_started = True
        self._statistics["capture_attempts"] += 1
        try:
            prefill_memory = 0
            if self._piecewise:
                prefill_memory = self._capture_short_prefill()
            with self.runtime.capture_context(self.decode, self.stream, self.workspace):
                memory = self.decode.capture_model()
            if type(memory) is not int or memory < 0:
                raise RuntimeError("Native capture returned invalid graph memory usage")
            entries = self.decode.model.concrete_aclgraph_entries
            if not entries or any(entry.aclgraph is None for entry in entries.values()):
                raise RuntimeError("Decode graph capture produced incomplete graphs")
        except BaseException:
            self._statistics["capture_failures"] += 1
            self._failed = True
            raise
        finally:
            if self._piecewise:
                self.runtime.guard_capture_disabled()
        self._captured = True
        self._statistics["capture_successes"] += 1
        return memory + prefill_memory

    def _capture_short_prefill(self):
        capture_descs = self.owner.cudagraph_dispatcher.get_capture_descs()
        if (
            len(capture_descs) != 1
            or capture_descs[0][0] != self.runtime.piecewise_mode
        ):
            raise RuntimeError("P capture must have only PIECEWISE descriptors")
        self._prefill_capture_active = True
        self._compile_prefill_active = True
        try:
            with self._prefill_startup_scope():
                p_stream = self.runtime.npu.current_stream()
                p_workspace = self.runtime.workspace._manager
                with self.runtime.capture_context(self.owner, p_stream, p_workspace):
                    with self._piecewise.capture(capture_descs[0][1]):
                        memory = self._prefill_capture_model()
            if type(memory) is not int or memory < 0:
                raise RuntimeError("Native P capture returned invalid memory usage")
            return memory
        finally:
            # Native capture's global flag is not protected by finally on error.
            self.runtime.guard_capture_disabled()
            self._prefill_capture_active = False
            self._compile_prefill_active = False

    @contextmanager
    def redirect_owner_capture(self):
        """Wrap native worker warmup without replacing its memory-owning runner."""
        self._check()
        if self._redirecting or not self._attached or self._capture_started:
            raise RuntimeError("Invalid or repeated owner capture redirection")
        original = vars(self.owner).get("capture_model", _MISSING)
        original_bound = self.owner.capture_model
        replacement = self.capture_decode
        self._redirecting = True
        self.owner.capture_model = replacement
        try:
            yield
        finally:
            changed = self.owner.capture_model is not replacement
            if not changed:
                if original is _MISSING:
                    delattr(self.owner, "capture_model")
                else:
                    self.owner.capture_model = original
            self._redirecting = False
            if changed:
                self._failed = True
                raise RuntimeError("Owner capture method changed during redirection")
            # Keep the bound method alive throughout the scope, including errors.
            del original_bound
