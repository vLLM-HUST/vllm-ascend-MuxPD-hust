"""Opt-in Ascend worker with independent, cooperatively scheduled P/D forwards."""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass, replace
from typing import Any

import torch
import vllm.forward_context as forward_context
import vllm_ascend.utils as ascend_utils
from vllm.distributed import get_ep_group, get_tp_group
from vllm.distributed.parallel_state import init_model_parallel_group
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.worker import NPUModelRunner, NPUWorker

from vllm_muxpd.communicators import LaneCollectiveRouter
from vllm_muxpd.config import validate_engine_config
from vllm_muxpd.decode_graph import (
    DecodeGraphController,
    GraphFencedExecutionBackend,
    GraphLaneState,
    validate_decode_graph_config,
)
from vllm_muxpd.evidence import emit_runtime_effective
from vllm_muxpd.lanes import LayerInterleaver
from vllm_muxpd.native_compat import (
    runner_contract,
    validate_lane_result,
    validate_lane_schedule,
    validate_output_types,
)
from vllm_muxpd.overlap import CollectiveGates, NPUOverlapBackend

# The pinned vLLM logging configuration installs its INFO handler only on the
# `vllm` namespace; an external plugin module name would silently lose INFO.
logger = init_logger("vllm.muxpd.worker")


def _physical_stream_id(stream: Any) -> int | None:
    """Query CANN's physical ID; Python stream IDs/handles are different IDs."""
    try:
        runtime = ctypes.CDLL("libacl_rt.so")
        get_id = runtime.aclrtStreamGetId
        get_id.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        get_id.restype = ctypes.c_int
        result = ctypes.c_int32()
        status = get_id(ctypes.c_void_p(stream.npu_stream), ctypes.byref(result))
        return result.value if status == 0 else None
    except (AttributeError, OSError):
        return None


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
        self._native_cache_contract = runner_contract(NPUModelRunner)
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

    def initialize_kv_cache(
        self, kv_cache_config: Any, kv_cache_allocation_context: Any = None
    ) -> None:
        profile = runner_contract(NPUModelRunner)
        if profile == "hust-oct10" and kv_cache_allocation_context is not None:
            raise RuntimeError("Legacy MuxPD cannot use KV allocation contexts")
        self.shared_kv_cache_layers = self._owner.shared_kv_cache_layers
        self.runner_only_attn_layers = self._owner.runner_only_attn_layers.copy()
        if profile == "hust-oct25":
            super().initialize_kv_cache(
                kv_cache_config, kv_cache_allocation_context=kv_cache_allocation_context
            )
        else:
            super().initialize_kv_cache(kv_cache_config)

    def initialize_kv_cache_tensors(
        self,
        kv_cache_config: Any,
        kernel_block_sizes: list[int] | None = None,
        kv_cache_allocation_context: Any = None,
    ) -> dict:
        # Native still initializes this runner's independent builders/buffers.
        # Tensor allocation itself belongs exclusively to the owner. Do not
        # enter the allocation context or bind shared model layer caches again.
        profile = runner_contract(NPUModelRunner)
        if profile == "hust-oct10":
            if (
                kernel_block_sizes is not None
                or kv_cache_allocation_context is not None
            ):
                raise RuntimeError("Legacy MuxPD received new KV initialization fields")
        elif kernel_block_sizes != self._owner.kernel_block_sizes:
            raise RuntimeError("MuxPD P/D kernel block layouts do not match")
        if (
            kv_cache_config.kv_cache_groups
            != self._owner.kv_cache_config.kv_cache_groups
        ):
            raise RuntimeError("MuxPD P/D KV cache groups do not match")
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
    validate_lane_schedule(output)
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
    outputs: list[ModelRunnerOutput], order: list[str], *, passthrough: bool = False
) -> ModelRunnerOutput:
    for output in outputs:
        validate_lane_result(output)
    if passthrough and len(outputs) == 1:
        output = outputs[0]
        # Native input-batch row order can differ after request removal or
        # compaction. Reuse the result only when every row already matches the
        # scheduler order; otherwise the regular merge rebuilds the mapping.
        if (
            output.req_ids == order
            and len(output.req_id_to_index) == len(order)
            and all(output.req_id_to_index.get(rid) == i for i, rid in enumerate(order))
            and len(output.sampled_token_ids) == len(order)
            and all(len(tokens) == 1 for tokens in output.sampled_token_ids)
            and output.logprobs is None
            and not output.pooler_output
            and not output.prompt_logprobs_dict
            and output.kv_connector_output is None
            and output.ec_connector_output is None
            and getattr(output, "routed_experts", None) is None
            and output.cudagraph_stats is None
        ):
            return output
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
    policy = config.scheduler_config.batch_admission_policy_config or {}
    decode_graph = policy.get("decode_graph", False)
    errors = []
    if type(decode_graph) is not bool:
        errors.append("decode_graph must be a boolean")
    elif decode_graph:
        validate_decode_graph_config(config, policy)
    elif not config.model_config.enforce_eager:
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
    if getattr(config.model_config, "enable_return_routed_experts", False):
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
        profile = validate_output_types(SchedulerOutput, ModelRunnerOutput)
        if runner_contract(NPUModelRunner) != profile:
            raise RuntimeError("MuxPD requires matching native core/Ascend contracts")
        _validate_worker_config(vllm_config)
        raw = vllm_config.scheduler_config.batch_admission_policy_config or {}
        self._decode_graph_enabled = raw.get("decode_graph", False)
        self._decode_graph_controller: DecodeGraphController | None = None
        self._layers_per_slice = raw.get("layers_per_slice", 1)
        self._stream_mode = raw.get("stream_mode", "fenced")
        self._prefill_slice_timing_enabled = raw.get("decode_budget_guard", False)
        if self._stream_mode not in ("fenced", "overlap"):
            raise ValueError("MuxPD stream_mode must be fenced or overlap")
        self._overlap_communication = raw.get("overlap_communication", "shared")
        if self._overlap_communication not in ("shared", "isolated"):
            raise ValueError("MuxPD overlap_communication must be shared or isolated")
        if self._overlap_communication == "isolated" and not (
            self.execution_mode == "decoupled" and self._stream_mode == "overlap"
        ):
            raise ValueError("MuxPD isolated communication requires decoupled overlap")
        self._collective_gates: CollectiveGates | None = None
        self._collective_router: LaneCollectiveRouter | None = None
        self._prefill_groups: list[Any] = []
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
        self._decoder_layer_count = 0
        self.muxpd_stats = {"prefill_batches": 0, "decode_batches": 0, "slices": 0}
        super().__init__(vllm_config, *args, **kwargs)

    def init_device(self) -> None:
        super().init_device()
        if ascend_utils.get_ascend_device_type() != ascend_utils.AscendDeviceType.A2:
            raise ValueError("MuxPD cooperative runner currently supports Ascend A2")
        ascend = get_ascend_config()
        if ascend.msmonitor_use_daemon:
            raise ValueError("MuxPD cooperative execution requires msmonitor disabled")
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
        if self._stream_mode == "overlap" and (
            ascend_utils.mlp_tp_enable() or ascend_utils.oproj_tp_enable()
        ):
            raise ValueError("MuxPD overlap requires fine-grained TP disabled")
        self.decode_runner = _SharedCacheRunner(
            self.vllm_config, self.device, self.model_runner
        )
        self._streams = {
            "prefill": torch.npu.Stream(device=self.device),
            "decode": torch.npu.Stream(device=self.device),
        }
        for lane, stream in {
            "caller": torch.npu.current_stream(),
            **self._streams,
        }.items():
            logger.info(
                "MuxPD stream rank=%d device=%s lane=%s "
                "physical_stream_id=%s torch_stream_id=%s npu_stream_handle=%s",
                self.rank,
                self.device,
                lane,
                _physical_stream_id(stream),
                getattr(stream, "stream_id", None),
                getattr(stream, "npu_stream", None),
            )
        self._configure_execution_backend()

    def _configure_execution_backend(self) -> None:
        """Compose graph ownership with stream ordering before loading weights."""
        graph_workspaces = None
        if self._decode_graph_enabled:
            import vllm.v1.worker.workspace as workspace

            if workspace._manager is None:
                raise RuntimeError("MuxPD requires the native prefill workspace")
            graph_workspaces = {
                "prefill": workspace._manager,
                "decode": workspace.WorkspaceManager(self.device, num_ubatches=1),
            }
        if self._stream_mode == "overlap":
            backend = NPUOverlapBackend()
            self._execution_backend = backend
            self._lane_states = {
                phase: backend.lane_state(
                    phase,
                    stream,
                    workspace_manager=(
                        graph_workspaces[phase] if graph_workspaces else None
                    ),
                )
                for phase, stream in self._streams.items()
            }
        elif graph_workspaces is not None:
            self._execution_backend = GraphFencedExecutionBackend()
            self._lane_states = {
                phase: GraphLaneState(None, stream, graph_workspaces[phase])
                for phase, stream in self._streams.items()
            }
        else:
            self._execution_backend = _NPUExecutionBackend()
            self._lane_states = {
                phase: _NPUState(None, stream)
                for phase, stream in self._streams.items()
            }
        try:
            if graph_workspaces is not None:
                self._decode_graph_controller = DecodeGraphController(
                    self.model_runner,
                    self.decode_runner,
                    self._streams["decode"],
                    graph_workspaces["decode"],
                )
                self._decode_graph_controller.install_prefill_guards()
            if self._stream_mode == "overlap":
                self._configure_overlap_communication(self._execution_backend)
        except BaseException:
            # Communication setup cleans up its own partial groups. No model
            # or graph exists yet; restore any installed owner guards too.
            if self._decode_graph_controller is not None:
                self._decode_graph_controller.close()
                self._decode_graph_controller = None
            raise

    def _configure_overlap_communication(self, backend: NPUOverlapBackend) -> None:
        groups = [("tp", get_tp_group())]
        if self.vllm_config.parallel_config.enable_expert_parallel:
            groups.append(("ep", get_ep_group()))
        groups = [(name, group) for name, group in groups if group.world_size > 1]
        if not groups:
            return
        if self._overlap_communication == "shared":
            self._collective_gates = CollectiveGates(backend.runtime).install(
                group.device_communicator for _, group in groups
            )
            return

        # This returns new coordinators without replacing global _TP/_EP. Every
        # rank creates the same groups in TP, EP order before any forward starts.
        siblings: dict[int, Any] = {}
        pairs = []
        try:
            for name, native in groups:
                sibling = siblings.get(id(native.device_group))
                if sibling is None:
                    sibling = init_model_parallel_group(
                        group_ranks=[list(ranks) for ranks in native.group_ranks],
                        local_rank=native.local_rank,
                        # Ascend's GroupCoordinatorPatch stores this as
                        # `backend`, unlike upstream's coordinator. Query the
                        # actual process group instead of either class layout.
                        backend=torch.distributed.get_backend(native.device_group),
                        use_message_queue_broadcaster=False,
                        group_name=f"{name}:muxpd_prefill",
                    )
                    self._prefill_groups.append(sibling)
                    # The Ascend registry keys HCCL options including our unique
                    # group_name. Fail before warmup if a future registry change
                    # aliases a native PG despite the independent coordinator.
                    if any(
                        sibling.device_group is group.device_group
                        for _, group in groups
                    ):
                        raise RuntimeError(
                            "MuxPD prefill process group aliases a native group"
                        )
                    siblings[id(native.device_group)] = sibling
                pairs.append((native.device_communicator, sibling.device_communicator))
                logger.info(
                    "MuxPD isolated group rank=%d kind=%s native=%s prefill=%s "
                    "ranks=%s",
                    self.rank,
                    name,
                    native.unique_name,
                    sibling.unique_name,
                    native.ranks,
                )
            self._warmup_prefill_groups()
            self._collective_router = LaneCollectiveRouter(backend)
            self._collective_router.install(pairs)
        except BaseException:
            try:
                self._release_overlap_communication()
            except BaseException:
                logger.exception(
                    "MuxPD failed to clean up communication initialization"
                )
            raise

    @torch.inference_mode()
    def _warmup_prefill_groups(self) -> None:
        # Native init_snapshot is captured before init_device returns. Materialize
        # sibling HCCL allocations now so determine_available_memory accounts for
        # them; do not pay lazy communicator initialization on the first request.
        with torch.npu.stream(self._streams["prefill"]):
            probe = torch.ones(1, device=self.device, dtype=torch.float32)
            for group in self._prefill_groups:
                group.device_communicator.all_reduce(probe)
            self._streams["prefill"].synchronize()

    def _release_overlap_communication(self) -> None:
        router = getattr(self, "_collective_router", None)
        gates = getattr(self, "_collective_gates", None)
        groups = getattr(self, "_prefill_groups", [])
        if router is None and gates is None and not groups:
            return
        # If synchronization fails, keep ownership: destroying a group with
        # pending device work is unsafe. A later shutdown may retry cleanup.
        for stream in self._streams.values():
            stream.synchronize()
        for attribute, installed in (
            ("_collective_router", router),
            ("_collective_gates", gates),
        ):
            if installed is not None:
                installed.restore()
                setattr(self, attribute, None)
        errors = []
        for group in list(reversed(groups)):
            try:
                group.destroy()
            except BaseException as error:
                errors.append(error)
            else:
                groups.remove(group)
        if errors:
            raise RuntimeError("MuxPD communication cleanup failed") from errors[0]

    def load_model(self) -> None:
        super().load_model()
        if self._decode_graph_controller is not None:
            self._decode_graph_controller.attach_shared_model()
        else:
            self.decode_runner.attach_model()
        model = self.model_runner.get_model()
        if hasattr(model, "language_model"):
            model = model.language_model
        layers = model.model.layers
        if not layers:
            raise ValueError("MuxPD could not find the Qwen3.5 decoder layers")
        self._decoder_layer_count = len(layers)
        final_layer = layers[-1]

        def after_layer(module: Any, args: Any, output: Any) -> None:
            del args, output
            if self._interleaver is not None:
                self.muxpd_stats["slices"] += 1
                self._interleaver.checkpoint(final=module is final_layer)

        self._hook_handles = [
            layer.register_forward_hook(after_layer) for layer in layers
        ]
        logger.info(
            "MuxPD cooperative worker enabled: shared weights/KV, independent P/D "
            "inputs, %d layers per slice, stream_mode=%s, overlap_communication=%s",
            self._layers_per_slice,
            self._stream_mode,
            self._overlap_communication,
        )

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        super().initialize_from_config(kv_cache_config)
        if self._decode_graph_controller is None:
            self.decode_runner.initialize_kv_cache(kv_cache_config)
            return
        backend = self._execution_backend
        caller = backend.capture()
        # Owner allocations precede D metadata construction. Keep any future
        # builder workspace allocations with the same manager used by capture.
        if self._stream_mode == "overlap":
            backend.begin_batch(preparation_lane=None)
        else:
            backend.fence()
        try:
            backend.restore(self._lane_states["decode"])
            self.decode_runner.initialize_kv_cache(kv_cache_config)
        finally:
            backend.fence()
            backend.restore(caller)
            if self._stream_mode == "overlap":
                # Startup caller work must see D's initialized metadata. This
                # joins only startup work, not P/D serving iterations.
                backend.join(["decode"])

    def compile_or_warm_up_model(self) -> Any:
        controller = self._decode_graph_controller
        if controller is None:
            return super().compile_or_warm_up_model()
        if (
            self._interleaver is not None
            or self._pending_lanes
            or getattr(self, "_prefill_task", None) is not None
            or getattr(self, "_awaiting_sample", False)
        ):
            raise RuntimeError("MuxPD graph capture is restricted to worker startup")
        # Keep the primary runner as the memory owner and preserve all native
        # ATB warmup, CPU binding, memory accounting and seed-reset behavior.
        with controller.redirect_owner_capture():
            result = super().compile_or_warm_up_model()
        logger.info("MuxPD decode graph rank=%d %s", self.rank, controller.snapshot())
        return result

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
        validate_lane_schedule(output)
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
                or getattr(params, "prompt_logprob_token_ids", None) is not None
            ):
                raise ValueError(
                    "MuxPD cooperative worker does not support logprobs, "
                    "pooling or structured outputs"
                )

    def _step_profiler(self) -> None:
        if getattr(self, "profiler", None) is not None:
            self.profiler.step()

    def _begin_execution(self, output: SchedulerOutput) -> SchedulerOutput:
        if self._stream_mode == "overlap":
            # Publish shared initialization from the caller before either lane
            # starts. Repeating all-block zeroing in a shard would race the
            # other lane's writes. Pending cohorts retain pinned disjoint KV.
            if output.new_block_ids_to_zero:
                self.model_runner._zero_block_ids(output.new_block_ids_to_zero)
            self._execution_backend.begin_batch(preparation_lane=None)
            return replace(output, new_block_ids_to_zero=None)
        self._execution_backend.fence()
        return output

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        if self._pending_lanes:
            raise RuntimeError("MuxPD requires sample_tokens before the next execute")
        self._step_profiler()
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
        execution_output = self._begin_execution(scheduler_output)
        for index, phase in enumerate(order):
            lane_output = _select_requests(
                execution_output, phases[phase], carry_side_effects=index == 0
            )
            runner = runners[phase]
            interleaver.add(
                phase,
                lambda runner=runner, output=lane_output: runner.execute_model(output),
                self._lane_states[phase],
            )
        try:
            # One active phase needs no intermediate layer fences or host yields.
            budget = self._layers_per_slice if len(order) > 1 else sys.maxsize
            results = interleaver.run(order, budget)
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
        sampled_lanes = self._pending_lanes.copy()
        try:
            for phase in self._pending_lanes:
                backend.restore(self._lane_states[phase])
                output = runners[phase].sample_tokens(None)
                if not isinstance(output, ModelRunnerOutput):
                    raise RuntimeError(
                        "MuxPD requires synchronous native sampler output"
                    )
                outputs.append(output)
                backend.fence()
            merged = _merge_outputs(
                outputs,
                self._pending_order,
                passthrough=getattr(self, "_decode_only_fast_path", False),
            )
            self._pending_lanes = []
            self._pending_order = []
        finally:
            backend.restore(caller_state)
            if self._stream_mode == "overlap":
                # Decode ACK must not wait for a still-suspended prefill. A
                # completed prefill is joined only when it was sampled here.
                backend.join(sampled_lanes)
        if self.execution_mode == "layered":
            self._record_effective_output(merged)
        return merged

    def _record_effective_output(self, output: ModelRunnerOutput) -> None:
        tokens = sum(len(row) for row in output.sampled_token_ids)
        requests = sum(bool(row) for row in output.sampled_token_ids)
        if tokens:
            emit_runtime_effective(
                self,
                role="worker",
                request_count=requests,
                output_token_count=tokens,
            )

    def shutdown(self) -> None:
        if getattr(self, "_muxpd_shutdown_complete", False) or getattr(
            self, "_muxpd_shutdown_in_progress", False
        ):
            return
        self._muxpd_shutdown_in_progress = True
        try:
            # Fail at the first unsafe boundary. A finally cleanup here could
            # free shared KV after a failed join and mask the original error.
            if self._interleaver is not None:
                self._interleaver.close()
                self._interleaver = None
            while self._hook_handles:
                self._hook_handles[-1].remove()
                self._hook_handles.pop()
            controller = getattr(self, "_decode_graph_controller", None)
            if controller is not None:
                for stream in self._streams.values():
                    stream.synchronize()
                update_stream = getattr(self.decode_runner, "update_stream", None)
                if update_stream is not None:
                    update_stream.synchronize()
                logger.info(
                    "MuxPD decode graph final rank=%d %s",
                    self.rank,
                    controller.snapshot(),
                )
                controller.close()
                self._decode_graph_controller = None
            self._release_overlap_communication()
            # Native cleanup may itself fail halfway. Keep runner/stream
            # references and the incomplete flag so a later call can retry.
            super().shutdown()
            self.decode_runner = None
            self._lane_states = {}
            self._streams = {}
            self._muxpd_shutdown_complete = True
        finally:
            self._muxpd_shutdown_in_progress = False
