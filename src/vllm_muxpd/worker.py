"""Opt-in Ascend worker with independent, cooperatively scheduled P/D forwards.

The first execution mode fences complete decoder slices. It validates separate
batches and shared weights/cache without claiming simultaneous NPU kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import vllm.forward_context as forward_context
import vllm_ascend.utils as ascend_utils
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.worker import NPUModelRunner, NPUWorker

from vllm_muxpd.config import validate_engine_config
from vllm_muxpd.lanes import LayerInterleaver

logger = init_logger(__name__)


@dataclass(frozen=True)
class _NPUState:
    forward_context: Any
    stream: Any


class _NPUExecutionBackend:
    def __init__(self) -> None:
        self._tail: Any = None

    def capture(self) -> _NPUState:
        return _NPUState(forward_context._forward_context, torch.npu.current_stream())

    def restore(self, state: _NPUState) -> None:
        forward_context._forward_context = state.forward_context
        torch.npu.set_stream(state.stream)
        ascend_utils._CURRENT_STREAM = state.stream
        if self._tail is not None:
            state.stream.wait_event(self._tail)

    def fence(self) -> None:
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        self._tail = event


class _SharedCacheRunner(NPUModelRunner):
    """Independent buffers/builders, with no second model/cache allocation."""

    def __init__(self, config: Any, device: Any, owner: NPUModelRunner):
        super().__init__(config, device)
        self._owner = owner
        self.requests = owner.requests
        self.num_prompt_logprobs = owner.num_prompt_logprobs
        self.mamba_state_idx = owner.mamba_state_idx

    def attach_model(self) -> None:
        self.model = self._owner.model
        self.model_memory_usage = 0
        self._has_sinks = self._owner._has_sinks
        self.is_multimodal_pruning_enabled = self._owner.is_multimodal_pruning_enabled

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        self.shared_kv_cache_layers = self._owner.shared_kv_cache_layers
        self.runner_only_attn_layers = self._owner.runner_only_attn_layers.copy()
        super().initialize_kv_cache(kv_cache_config)

    def initialize_kv_cache_tensors(self, kv_cache_config: Any) -> dict:
        del kv_cache_config
        if not self._owner.kv_caches:
            raise RuntimeError("MuxPD cache owner must be initialized first")
        self.kv_caches = self._owner.kv_caches
        layers = self.compilation_config.static_forward_context
        return {
            name: layers[name].kv_cache
            for group in self.kv_cache_config.kv_cache_groups
            for name in group.layer_names
        }


def _select_requests(
    output: SchedulerOutput, req_ids: set[str], *, carry_side_effects: bool
) -> SchedulerOutput:
    """Slice every per-request field together; preserve native batch contracts."""
    cached = output.scheduled_cached_reqs
    indices = [i for i, req_id in enumerate(cached.req_ids) if req_id in req_ids]
    selected_cached = replace(
        cached,
        req_ids=[cached.req_ids[i] for i in indices],
        resumed_req_ids=cached.resumed_req_ids & req_ids,
        new_token_ids=(
            [cached.new_token_ids[i] for i in indices] if cached.new_token_ids else []
        ),
        all_token_ids={
            key: value for key, value in cached.all_token_ids.items() if key in req_ids
        },
        new_block_ids=[cached.new_block_ids[i] for i in indices],
        num_computed_tokens=[cached.num_computed_tokens[i] for i in indices],
        num_output_tokens=[cached.num_output_tokens[i] for i in indices],
    )
    scheduled = {
        key: value
        for key, value in output.num_scheduled_tokens.items()
        if key in req_ids
    }
    return replace(
        output,
        scheduled_new_reqs=[
            req for req in output.scheduled_new_reqs if req.req_id in req_ids
        ],
        scheduled_cached_reqs=selected_cached,
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        preempted_req_ids=(output.preempted_req_ids or set()) & req_ids,
        free_encoder_mm_hashes=(
            output.free_encoder_mm_hashes if carry_side_effects else []
        ),
        new_block_ids_to_zero=(
            output.new_block_ids_to_zero if carry_side_effects else None
        ),
    )


def _merge_outputs(
    outputs: list[ModelRunnerOutput], order: list[str]
) -> ModelRunnerOutput:
    sampled = {}
    prompt_logprobs = {}
    num_nans = {}
    for output in outputs:
        if output.logprobs is not None or output.pooler_output:
            raise RuntimeError("MuxPD lane output has unsupported sampling metadata")
        for req_id, tokens in zip(output.req_ids, output.sampled_token_ids):
            if req_id in sampled:
                raise RuntimeError(f"Request scheduled in both MuxPD lanes: {req_id}")
            sampled[req_id] = tokens
        prompt_logprobs.update(output.prompt_logprobs_dict)
        num_nans.update(output.num_nans_in_logits or {})
    if set(sampled) != set(order):
        raise RuntimeError("MuxPD lane outputs do not match scheduled request IDs")
    return ModelRunnerOutput(
        req_ids=order,
        req_id_to_index={req_id: i for i, req_id in enumerate(order)},
        sampled_token_ids=[sampled[req_id] for req_id in order],
        prompt_logprobs_dict=prompt_logprobs,
        num_nans_in_logits=num_nans or None,
    )


def _validate_worker_config(config: Any) -> None:
    validate_engine_config(config)
    errors = []
    if not config.model_config.enforce_eager:
        errors.append("enforce_eager must be true")
    if config.use_v2_model_runner:
        errors.append("only the V1 NPU runner is supported")
    if config.model_config.dtype != torch.bfloat16:
        errors.append("only BF16 weights are supported")
    if not config.model_config.hf_text_config.model_type.startswith("qwen3_5"):
        errors.append("the cooperative adapter currently supports Qwen3.5 only")
    if not getattr(config.model_config, "is_multimodal_model", False):
        pass
    elif not config.model_config.multimodal_config.language_model_only:
        errors.append("multimodal models require language_model_only")
    for name in ("lora_config", "quant_config", "weight_transfer_config"):
        if getattr(config, name, None) is not None:
            errors.append(f"{name} is unsupported by cooperative execution")
    if config.model_config.enable_sleep_mode:
        errors.append("sleep mode is unsupported by cooperative execution")
    if config.model_config.enable_return_routed_experts:
        errors.append("routed-expert capture is unsupported")
    if getattr(config.parallel_config, "enable_dbo", False):
        errors.append("native DBO must be disabled")
    if getattr(config.parallel_config, "use_ubatching", False):
        errors.append("native ubatching must be disabled")
    if errors:
        raise ValueError(
            "MuxPD worker configuration is incompatible: " + "; ".join(errors)
        )


class MuxPDNPUWorker(NPUWorker):
    """Explicit ``--worker-cls vllm_muxpd.worker.MuxPDNPUWorker`` adapter."""

    execution_mode = "layered"

    def __init__(self, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        _validate_worker_config(vllm_config)
        raw = vllm_config.scheduler_config.batch_admission_policy_config or {}
        self._layers_per_slice = raw.get("layers_per_slice", 1)
        if type(self._layers_per_slice) is not int or self._layers_per_slice < 1:
            raise ValueError("MuxPD layers_per_slice must be a positive integer")
        execution = raw.get("execution", self.execution_mode)
        if execution != self.execution_mode:
            raise ValueError(
                f"MuxPD cooperative worker requires execution={self.execution_mode}"
            )
        self._interleaver: LayerInterleaver | None = None
        self._pending_lanes: list[str] = []
        self._pending_order: list[str] = []
        self._hook_handles: list[Any] = []
        self.muxpd_stats = {"prefill_batches": 0, "decode_batches": 0, "slices": 0}
        super().__init__(vllm_config, *args, **kwargs)

    def init_device(self) -> None:
        super().init_device()
        if ascend_utils.get_ascend_device_type() != ascend_utils.AscendDeviceType.A2:
            raise ValueError("MuxPD cooperative runner currently supports Ascend A2")
        ascend = get_ascend_config()
        if ascend.eplb_config.dynamic_eplb or ascend.eplb_config.expert_map_path:
            raise ValueError("MuxPD cooperative execution requires EPLB disabled")
        if ascend.multistream_overlap_shared_expert:
            raise ValueError(
                "MuxPD requires shared-expert multistream overlap disabled"
            )
        if ascend_utils.enable_sp() or ascend_utils.enable_sp_by_pass():
            raise ValueError("MuxPD cooperative execution requires FlashComm disabled")
        if self.vllm_config.parallel_config.tensor_parallel_size >= 16:
            raise ValueError("MuxPD A2 adapter does not support MC2 communication")
        self.decode_runner = _SharedCacheRunner(
            self.vllm_config, self.device, self.model_runner
        )
        self._streams = {
            "prefill": torch.npu.Stream(device=self.device),
            "decode": torch.npu.Stream(device=self.device),
        }
        self._execution_backend = _NPUExecutionBackend()

    def load_model(self) -> None:
        super().load_model()
        self.decode_runner.attach_model()
        model = self.model_runner.get_model()
        if hasattr(model, "language_model"):
            model = model.language_model
        layers = model.model.layers
        if not layers:
            raise ValueError("MuxPD could not find the Qwen3.5 decoder layers")

        def after_layer(module: Any, args: Any, output: Any) -> None:
            del module, args, output
            if self._interleaver is not None:
                self.muxpd_stats["slices"] += 1
                self._interleaver.checkpoint()

        self._hook_handles = [
            layer.register_forward_hook(after_layer) for layer in layers
        ]
        logger.info(
            "MuxPD cooperative worker enabled: shared weights/KV, independent P/D "
            "inputs, %d layers per slice, full layer event fences (no kernel overlap)",
            self._layers_per_slice,
        )

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        super().initialize_from_config(kv_cache_config)
        self.decode_runner.initialize_kv_cache(kv_cache_config)

    def _retire_finished(self, finished: set[str]) -> None:
        for runner in (self.model_runner, self.decode_runner):
            for req_id in finished:
                runner.input_batch.remove_request(req_id)
                runner.requests.pop(req_id, None)
                runner.num_prompt_logprobs.pop(req_id, None)
                runner.mamba_state_idx.pop(req_id, None)
                runner._hybrid_accepted_by_req.pop(req_id, None)

    def _invalidate_batch_rows(self, req_ids: set[str]) -> None:
        # Preemption may happen while one lane is idle. A shared request registry
        # does not refresh that lane's cached block-table and token-buffer rows.
        for runner in (self.model_runner, self.decode_runner):
            for req_id in req_ids:
                runner.input_batch.remove_request(req_id)
                runner._hybrid_accepted_by_req.pop(req_id, None)

    def _validate_requests(self, output: SchedulerOutput) -> None:
        if output.scheduled_encoder_inputs or output.scheduled_spec_decode_tokens:
            raise ValueError("MuxPD cooperative worker accepts text-only target tokens")
        for request in output.scheduled_new_reqs:
            params = request.sampling_params
            if request.mm_features or request.prompt_embeds is not None:
                raise ValueError("MuxPD cooperative worker requires text token IDs")
            if params is None or (
                params.logprobs is not None
                or params.prompt_logprobs is not None
                or params.structured_outputs is not None
            ):
                raise ValueError(
                    "MuxPD cooperative worker does not support logprobs, "
                    "pooling or structured outputs"
                )

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        if self._pending_lanes:
            raise RuntimeError("MuxPD requires sample_tokens before the next execute")
        self._validate_requests(scheduler_output)
        self._retire_finished(scheduler_output.finished_req_ids)
        self._invalidate_batch_rows(
            set(scheduler_output.preempted_req_ids or ())
            | scheduler_output.scheduled_cached_reqs.resumed_req_ids
            | {req.req_id for req in scheduler_output.scheduled_new_reqs}
        )
        scheduled = set(scheduler_output.num_scheduled_tokens)
        if not scheduled:
            return ModelRunnerOutput(req_ids=[], req_id_to_index={})
        declared_prefills = getattr(scheduler_output, "muxpd_prefill_request_ids", None)
        if declared_prefills is None:
            raise ValueError(
                "MuxPD worker requires scheduler_output.muxpd_prefill_request_ids"
            )
        prefills = set(declared_prefills)
        if not prefills <= scheduled:
            raise ValueError("MuxPD prefill IDs must be scheduled in this iteration")
        phases = {"decode": scheduled - prefills, "prefill": prefills}
        runners = {"prefill": self.model_runner, "decode": self.decode_runner}
        order = [phase for phase, ids in phases.items() if ids]
        interleaver = LayerInterleaver(self._execution_backend)
        self._interleaver = interleaver
        self._execution_backend.fence()
        for index, phase in enumerate(order):
            lane_output = _select_requests(
                scheduler_output, phases[phase], carry_side_effects=index == 0
            )
            runner = runners[phase]
            interleaver.add(
                phase,
                lambda runner=runner, output=lane_output: runner.execute_model(output),
                _NPUState(None, self._streams[phase]),
            )
        try:
            results = interleaver.run(order, self._layers_per_slice)
            if any(value is not None for value in results.values()):
                raise RuntimeError("MuxPD expected deferred native sampler outputs")
            for phase in order:
                self.muxpd_stats[f"{phase}_batches"] += 1
            self._pending_lanes = order
            self._pending_order = list(scheduler_output.num_scheduled_tokens)
        finally:
            self._interleaver = None
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: Any) -> ModelRunnerOutput:
        if grammar_output is not None:
            raise ValueError("MuxPD cooperative worker does not support grammar output")
        outputs = []
        runners = {"prefill": self.model_runner, "decode": self.decode_runner}
        backend = self._execution_backend
        caller_state = backend.capture()
        try:
            for phase in self._pending_lanes:
                backend.restore(_NPUState(None, self._streams[phase]))
                output = runners[phase].sample_tokens(None)
                if not isinstance(output, ModelRunnerOutput):
                    raise RuntimeError(
                        "MuxPD requires synchronous native sampler output"
                    )
                outputs.append(output)
                backend.fence()
            merged = _merge_outputs(outputs, self._pending_order)
            self._pending_lanes = []
            self._pending_order = []
            return merged
        finally:
            backend.restore(caller_state)

    def shutdown(self) -> None:
        if self._interleaver is not None:
            self._interleaver.close()
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        # The primary runner owns the shared model/cache; release it only once.
        super().shutdown()
        if hasattr(self, "decode_runner"):
            self.decode_runner = None
