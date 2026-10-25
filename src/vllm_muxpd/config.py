"""Explicit, fail-closed configuration for the synchronous MuxPD adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class MuxPDConfig:
    """Times are milliseconds; estimates are initial values, not SLO guarantees."""

    ttft_slo_ms: float = 1000.0
    tpot_slo_ms: float = 100.0
    max_prefill_wait_ms: float = 100.0
    max_decode_steps: int = 8
    max_prefill_requests: int = 8
    max_prefill_batch_tokens: int = 0
    prefill_fixed_ms: float = 2.0
    prefill_token_ms: float = 0.05
    decode_fixed_ms: float = 10.0
    decode_context_token_ms: float = 0.001
    estimate_safety_factor: float = 1.2
    ewma_alpha: float = 0.2
    trace_path: str | None = None
    execution: str = "serial"
    layers_per_slice: int = 1
    stream_mode: str = "fenced"
    adaptive_layers: bool = False
    prefill_first: bool = False
    overlap_communication: str = "shared"
    decode_graph: bool = False
    compile_full_prefill: bool = False
    compile_prefill_groups: bool = False
    prefill_layer_token_budget: int = 0
    decode_budget_guard: bool = False

    def __post_init__(self) -> None:
        for name in (
            "ttft_slo_ms",
            "tpot_slo_ms",
            "max_prefill_wait_ms",
            "prefill_fixed_ms",
            "prefill_token_ms",
            "decode_fixed_ms",
            "decode_context_token_ms",
            "estimate_safety_factor",
            "ewma_alpha",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"MuxPD {name} must be a finite number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"MuxPD {name} must be finite and positive")
        if self.ewma_alpha > 1 or self.estimate_safety_factor < 1:
            raise ValueError(
                "MuxPD requires ewma_alpha <= 1 and estimate_safety_factor >= 1"
            )
        for name in (
            "max_decode_steps",
            "max_prefill_requests",
            "max_prefill_batch_tokens",
            "layers_per_slice",
            "prefill_layer_token_budget",
        ):
            value = getattr(self, name)
            minimum = (
                0
                if name in ("max_prefill_batch_tokens", "prefill_layer_token_budget")
                else 1
            )
            if type(value) is not int or value < minimum:
                raise ValueError(f"MuxPD {name} must be an integer >= {minimum}")
        if self.execution not in ("serial", "layered", "decoupled"):
            raise ValueError(
                "MuxPD execution must be 'serial', 'layered', or 'decoupled'"
            )
        if self.prefill_layer_token_budget and self.execution != "decoupled":
            raise ValueError(
                "MuxPD prefill_layer_token_budget requires execution='decoupled'"
            )
        if self.stream_mode not in ("fenced", "overlap"):
            raise ValueError("MuxPD stream_mode must be 'fenced' or 'overlap'")
        if self.execution == "serial" and self.stream_mode != "fenced":
            raise ValueError("MuxPD serial execution requires stream_mode='fenced'")
        if type(self.adaptive_layers) is not bool:
            raise ValueError("MuxPD adaptive_layers must be a boolean")
        if self.adaptive_layers and self.execution != "decoupled":
            raise ValueError("MuxPD adaptive_layers requires execution='decoupled'")
        if type(self.decode_budget_guard) is not bool:
            raise ValueError("MuxPD decode_budget_guard must be a boolean")
        if self.decode_budget_guard and not (
            self.execution == "decoupled"
            and self.adaptive_layers
            and self.stream_mode == "fenced"
        ):
            raise ValueError(
                "MuxPD decode_budget_guard requires decoupled adaptive_layers "
                "with stream_mode='fenced'"
            )
        if type(self.prefill_first) is not bool:
            raise ValueError("MuxPD prefill_first must be a boolean")
        if self.prefill_first and not (
            self.execution == "decoupled"
            and self.adaptive_layers
            and self.stream_mode == "fenced"
        ):
            raise ValueError(
                "MuxPD prefill_first requires decoupled adaptive_layers "
                "with stream_mode='fenced'"
            )
        if self.overlap_communication not in ("shared", "isolated"):
            raise ValueError(
                "MuxPD overlap_communication must be 'shared' or 'isolated'"
            )
        if self.overlap_communication == "isolated" and not (
            self.execution == "decoupled" and self.stream_mode == "overlap"
        ):
            raise ValueError(
                "MuxPD isolated overlap_communication requires "
                "execution='decoupled' with stream_mode='overlap'"
            )
        if type(self.decode_graph) is not bool:
            raise ValueError("MuxPD decode_graph must be a boolean")
        if self.decode_graph and not (
            self.execution == "decoupled"
            and (
                self.stream_mode == "fenced" or self.overlap_communication == "isolated"
            )
        ):
            raise ValueError(
                "MuxPD decode_graph requires decoupled fenced execution or "
                "experimental overlap with isolated communication"
            )
        if type(self.compile_full_prefill) is not bool:
            raise ValueError("MuxPD compile_full_prefill must be a boolean")
        if self.compile_full_prefill and not (
            self.decode_graph
            and self.execution == "decoupled"
            and self.stream_mode == "fenced"
        ):
            raise ValueError(
                "MuxPD compile_full_prefill requires decode_graph with "
                "decoupled fenced execution"
            )
        if type(self.compile_prefill_groups) is not bool:
            raise ValueError("MuxPD compile_prefill_groups must be a boolean")
        if self.compile_prefill_groups and not (
            self.execution == "decoupled"
            and self.decode_graph
            and self.stream_mode == "overlap"
            and self.overlap_communication == "isolated"
            and self.layers_per_slice == 4
            and self.prefill_layer_token_budget == 0
            and not self.adaptive_layers
            and not self.prefill_first
            and not self.decode_budget_guard
            and not self.compile_full_prefill
        ):
            raise ValueError(
                "MuxPD compile_prefill_groups requires isolated graph overlap, "
                "fixed q4, work budget zero and no other prefill experiment"
            )
        if self.trace_path is not None:
            if not isinstance(self.trace_path, str) or not self.trace_path.strip():
                raise ValueError("MuxPD trace_path must be a nonempty string or null")
            try:
                self.trace_path.format(pid=0)
            except (KeyError, ValueError, IndexError) as exc:
                raise ValueError(
                    "MuxPD trace_path supports only the {pid} placeholder"
                ) from exc

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "MuxPDConfig":
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("MuxPD policy configuration must be a JSON object")
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown MuxPD configuration keys: {sorted(unknown)}")
        return cls(**raw)


def validate_engine_config(config: Any) -> None:
    """Reject modes that can invalidate phase separation or full prefills."""
    scheduler = config.scheduler_config
    model = config.model_config
    cache = config.cache_config
    parallel = config.parallel_config
    errors: list[str] = []
    if scheduler.async_scheduling is not False:
        errors.append("async_scheduling must be explicitly false")
    if scheduler.enable_chunked_prefill:
        errors.append("enable_chunked_prefill must be false")
    if scheduler.long_prefill_token_threshold != 0:
        errors.append("long_prefill_token_threshold must be 0")
    for name in ("max_num_batched_tokens", "max_num_scheduled_tokens"):
        value = getattr(scheduler, name, None)
        if value is not None and value < model.max_model_len:
            errors.append(f"{name} must be >= max_model_len for complete prefills")
    if cache.enable_prefix_caching:
        errors.append("enable_prefix_caching must be false in this validated adapter")
    if getattr(cache, "mamba_cache_mode", "none") != "none":
        errors.append("mamba_cache_mode must be 'none'; 'align' splits prefills")
    for name in (
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            errors.append(f"{name} must be 1 (tensor parallelism is supported)")
    for name in (
        "speculative_config",
        "kv_transfer_config",
        "ec_transfer_config",
    ):
        if getattr(config, name, None) is not None:
            errors.append(f"{name} is not supported")
    if getattr(getattr(config, "aux_output_config", None), "enabled", False):
        errors.append("auxiliary output connectors are not supported")
    if getattr(model, "return_sampling_mask", False):
        errors.append("return_sampling_mask is not supported")
    if getattr(model, "is_encoder_decoder", False) or getattr(
        model, "is_diffusion", False
    ):
        errors.append("only decoder-only autoregressive generation is supported")
    if getattr(model, "runner_type", "generate") != "generate":
        errors.append("model runner_type must be 'generate'")
    if getattr(config, "max_concurrent_batches", 1) != 1:
        errors.append("max_concurrent_batches must be 1")
    policy = getattr(scheduler, "batch_admission_policy", None)
    if policy is not None:
        name = (
            policy
            if isinstance(policy, str)
            else f"{policy.__module__}.{policy.__qualname__}"
        )
        if name != "vllm_muxpd.policy.MuxPDPolicy":
            errors.append(
                "batch_admission_policy must be vllm_muxpd.policy.MuxPDPolicy"
            )
    if errors:
        raise ValueError("MuxPD configuration is incompatible: " + "; ".join(errors))
