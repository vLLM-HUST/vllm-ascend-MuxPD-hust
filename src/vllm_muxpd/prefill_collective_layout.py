"""Default-off experimental P TP collective prefix layout.

The tensor transformation follows the frozen v1 diagnostic. Active ownership
is independent of bounded diagnostic history; no lifetime cohort limit exists.
Importing this module does not import torch, vLLM, or an accelerator runtime.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import operator
import os
from collections import deque
from pathlib import Path
from threading import get_ident

ATTRIBUTE = "_prefill_collective_layout"
POLICY_KEY = "experimental_prefill_collective_layout"
PROFILE = "experimental-qwen35-moe-tp4ep4-prefix-v1"
MODES = frozenset(("prefix",))
_MISSING = object()
HISTORY_SIZE = 16
MAX_CALLS = 4096
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024


def _unwrap(argument):
    try:
        return object.__getattribute__(argument, "worker")
    except AttributeError:
        return argument


def _ints(values):
    if str(getattr(values, "device", "cpu")) != "cpu":
        raise RuntimeError("Layout metadata must already be on CPU")
    return [operator.index(value) for value in values]


def _digest(values):
    return hashlib.sha256(
        json.dumps(_ints(values), separators=(",", ":")).encode()
    ).hexdigest()


def _sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class TorchRuntime:
    def __init__(self):
        import torch
        from greenlet import getcurrent
        from vllm.forward_context import get_forward_context

        self.torch = torch
        self.context = get_forward_context
        self.coroutine = getcurrent

    def transform(self, tensor, prefix, suffix, original):
        scratch = tensor.new_zeros((prefix + tensor.shape[0], tensor.shape[1]))
        result = None
        try:
            scratch[prefix : prefix + tensor.shape[0]].copy_(tensor)
            result = original(scratch)
            if (
                result.shape != scratch.shape
                or result.dtype != scratch.dtype
                or result.device != scratch.device
            ):
                raise RuntimeError(
                    "Existing collective changed payload shape/dtype/device"
                )
            tensor.copy_(result[prefix : prefix + tensor.shape[0]])
            return tensor
        except BaseException:
            # Do not release buffers referenced by possibly submitted device work.
            # Successful calls retain normal same-stream allocator semantics.
            self.failed_buffers = (tensor, scratch, result)
            raise


def _route(worker, group, *, serving=False):
    communicator = group.device_communicator
    router = worker._collective_router
    function = communicator.all_reduce
    if (
        getattr(function, "_muxpd_collective_router", None) is not router
        or router._backend is not worker._execution_backend
        or router._failed
        or communicator.device_group is not group.device_group
        or not any(
            owner is communicator and name == "all_reduce" and installed is function
            for owner, name, _, installed in router._installed
        )
    ):
        raise RuntimeError("Existing isolated P route is not owned/intact")
    closure = inspect.getclosurevars(function).nonlocals
    target = getattr(closure.get("prefill"), "__self__", None)
    siblings = [s for s in worker._prefill_groups if s.device_communicator is target]
    if (
        len(siblings) != 1
        or getattr(closure.get("original"), "__self__", None) is not communicator
    ):
        raise RuntimeError("Router does not identify one existing P sibling")
    sibling = siblings[0]
    if (
        sibling.device_group is group.device_group
        or sibling.ranks != group.ranks
        or sibling.device_communicator.device_group is not sibling.device_group
    ):
        raise RuntimeError("P group must be independent with identical rank membership")
    if serving and router._backend.active_lane != "prefill":
        raise RuntimeError("Layout collective is outside the actual P lane")
    return {
        "process_id": os.getpid(),
        "coordinator_name": group.unique_name,
        "coordinator_python_id": id(group),
        "device_group_python_id": id(group.device_group),
        "selected_coordinator_name": sibling.unique_name,
        "selected_device_group_python_id": id(sibling.device_group),
        "ranks": list(group.ranks),
        "rank_in_group": int(group.rank_in_group),
        "router_calls": dict(router.calls),
        "active_lane": router._backend.active_lane,
    }


class Layout:
    def __init__(self, worker, mode, group, runtime, hidden_size):
        if mode not in MODES:
            raise ValueError("mode must be prefix")
        self.worker, self.mode, self.group, self.runtime = worker, mode, group, runtime
        self.hidden_size, self.rank = hidden_size, int(worker.rank)
        self.local_rank = int(getattr(worker, "local_rank", worker.rank))
        self.expected_device = f"npu:{self.local_rank}"
        self.patches = []
        self.cohorts = deque(maxlen=HISTORY_SIZE)
        self.stats = {
            "admitted": 0,
            "closed": 0,
            "admission_failed": 0,
            "treated": 0,
            "transformed_calls": 0,
            "zero_d_calls": 0,
        }
        self.last_cohort_id = 0
        self.thread = get_ident()
        self.startup_route = None
        self.first_execute_verified = False
        self.primary_failure = None
        self.install_complete = False
        self.active = None
        self.context = None
        self.in_collective = False
        self.errors = deque(maxlen=8)
        self.restored_count = 0
        self.uninstalled = False

    def _patch(self, owner, name, factory):
        slot = vars(owner).get(name, _MISSING)
        original = getattr(owner, name)
        wrapper = factory(original)
        self.patches.append((owner, name, slot, original, wrapper))
        setattr(owner, name, wrapper)

    def _fail(self, exc):
        error = f"{type(exc).__name__}: {exc}"
        if self.primary_failure is None:
            self.primary_failure = error
        if not self.errors or self.errors[-1] != error:
            self.errors.append(error)
        if self.active is not None:
            self.active["error"] = self.primary_failure

    def _check(self):
        if get_ident() != self.thread:
            raise RuntimeError("Layout requires its owning host thread")
        if self.primary_failure is not None:
            raise RuntimeError(f"Layout is poisoned: {self.primary_failure}")
        if self.uninstalled or not self.install_complete:
            raise RuntimeError("Layout is not installed")
        if any(vars(o).get(n, _MISSING) is not w for o, n, _, _, w in self.patches):
            raise RuntimeError("Layout method ownership changed externally")

    def install(self):
        if self.patches or self.install_complete or self.uninstalled:
            raise RuntimeError("Layout installation already attempted")
        self.startup_route = _route(self.worker, self.group)
        # First version supports the original methods only, without outer observers.
        for owner, name in (
            (self.worker, "execute_model"),
            (self.worker, "_clear_prefill"),
            (self.group, "_all_reduce_out_place"),
        ):
            method = getattr(owner, name)
            if getattr(method, "__self__", None) is not owner or getattr(
                method, "__func__", None
            ) is not getattr(type(owner), name):
                raise RuntimeError(f"Layout refuses foreign method slot: {name}")
        try:
            self._patch(self.worker, "execute_model", self._execute_wrapper)
            self._patch(self.worker, "_clear_prefill", self._clear_wrapper)
            self._patch(self.group, "_all_reduce_out_place", self._collective_wrapper)
            self.install_complete = True
        except BaseException as exc:
            self._fail(exc)
            # Restore only slots this failed install actually owns. Retain the
            # controller if any restoration fails so shutdown can retry safely.
            for owner, name, slot, _, wrapper in reversed(self.patches):
                if vars(owner).get(name, _MISSING) is not wrapper:
                    continue
                try:
                    if slot is _MISSING:
                        delattr(owner, name)
                    else:
                        setattr(owner, name, slot)
                    self.restored_count += 1
                except BaseException as restore_error:
                    self._fail(restore_error)
            raise

    def _archive(self, status):
        record = self.active
        record["status"] = status
        self.stats[status] += 1
        self.cohorts.append(record)
        self.active = self.context = None

    def _admit(self, output):
        new = set(output.muxpd_prefill_request_ids)
        if not new:
            return
        if self.active is not None or self.worker._prefill_task is not None:
            raise RuntimeError("A layout cohort cannot replace an owned P cohort")
        counts = dict(output.num_scheduled_tokens)
        if (
            not new <= counts.keys()
            or any(type(n) is not int or n <= 0 for n in counts.values())
            or sum(counts.values()) != output.total_num_scheduled_tokens
            or type(output.muxpd_prefill_cohort_id) is not int
            or output.muxpd_prefill_cohort_id <= 0
            or set(output.muxpd_pending_prefill_ids) != new
            or output.muxpd_cancel_prefill_request_ids
            or output.scheduled_spec_decode_tokens
        ):
            raise RuntimeError("Malformed new-P admission layout")
        p_counts = {rid: n for rid, n in counts.items() if rid in new}
        d_counts = {rid: n for rid, n in counts.items() if rid not in new}
        if any(n != 1 for n in d_counts.values()):
            raise RuntimeError("Layout diagnostic requires ordinary one-token decodes")
        if output.muxpd_prefill_cohort_id <= self.last_cohort_id:
            raise RuntimeError("Layout cohort identity must increase monotonically")
        cached = output.scheduled_cached_reqs
        requests = output.scheduled_new_reqs
        if (
            new
            & (
                set(cached.req_ids)
                | set(cached.resumed_req_ids)
                | set(output.preempted_req_ids or ())
            )
            or {request.req_id for request in requests} != new
            or len(requests) != len(new)
            or any(
                request.num_computed_tokens != 0
                or len(request.prompt_token_ids) != p_counts[request.req_id]
                for request in requests
            )
        ):
            raise RuntimeError("Layout requires complete fresh scheduled_new_reqs")
        record = {
            "cohort_id": output.muxpd_prefill_cohort_id,
            "mode": self.mode,
            "admission_p_counts": p_counts,
            "admission_d_counts": d_counts,
            "p_rows": sum(p_counts.values()),
            "d_rows": sum(d_counts.values()),
            "status": "admitted",
            "batch": None,
            "calls": [],
            "zero_d_bypass_calls": 0,
            "clear_count": 0,
            "error": None,
            "native_equivalent_order_claimed": False,
        }
        self.stats["admitted"] += 1
        self.last_cohort_id = record["cohort_id"]
        self.active = record

    def _execute_wrapper(self, original):
        signature = inspect.signature(original)

        def wrapper(*args, **kwargs):
            admitted = None
            try:
                self._check()
                if not self.first_execute_verified:
                    route = _route(self.worker, self.group)
                    if route != self.startup_route or not self._idle():
                        raise RuntimeError(
                            "Unexpected model/collective work after layout installation"
                        )
                    self.first_execute_verified = True
                output = signature.bind(*args, **kwargs).arguments["scheduler_output"]
                if getattr(output, "muxpd_execution", None) != "decoupled":
                    raise RuntimeError(
                        "Layout requires the original decoupled scheduler"
                    )
                previous = self.active
                self._admit(output)
                if self.active is not previous:
                    admitted = self.active
                return original(*args, **kwargs)
            except BaseException as exc:
                self._fail(exc)
                # Only an admission that never obtained a worker cohort can
                # roll back immediately. Suspended/in-flight P retains ownership.
                if (
                    admitted is not None
                    and self.worker._prefill_task is None
                    and self.worker._prefill_cohort_id is None
                ):
                    self._archive("admission_failed")
                raise

        return wrapper

    def _clear_wrapper(self, original):
        def wrapper(*args, **kwargs):
            try:
                if get_ident() != self.thread:
                    raise RuntimeError("Layout clear requires owning host thread")
                task = self.worker._prefill_task
                if self.in_collective or (
                    task is not None and not task.tasks["prefill"].done
                ):
                    raise RuntimeError(
                        "Cannot clear layout while P owns unfinished work"
                    )
                result = original(*args, **kwargs)
                if (
                    self.worker._prefill_task is not None
                    or self.worker._prefill_cohort_id is not None
                ):
                    raise RuntimeError(
                        "Original P clear did not release cohort ownership"
                    )
                if self.active is not None:
                    self.active["clear_count"] += 1
                    self._archive("closed")
                return result
            except BaseException as exc:
                self._fail(exc)
                raise

        return wrapper

    def _verify_owner_and_batch(self):
        worker, record = self.worker, self.active
        if record is None or worker._prefill_cohort_id != record["cohort_id"]:
            raise RuntimeError("Actual P lane has no matching frozen layout cohort")
        interleaver = worker._interleaver
        if (
            interleaver is None
            or interleaver is not worker._prefill_task
            or interleaver.active is None
            or interleaver.active.name != "prefill"
            or interleaver.active.coroutine is not self.runtime.coroutine()
        ):
            raise RuntimeError("Layout is outside the owning P coroutine")
        context = self.runtime.context()
        mode = getattr(getattr(context, "cudagraph_runtime_mode", None), "name", None)
        if (
            context is None
            or getattr(context, "capturing", False)
            or getattr(context, "in_profile_run", False)
            or mode != "NONE"
            or not getattr(context, "skip_compiled", False)
            or worker._prefill_compiled
        ):
            raise RuntimeError("Layout requires raw non-capturing P execution")
        if self.context is not None and context is not self.context:
            raise RuntimeError("P forward context changed inside frozen layout cohort")
        self.context = context
        if record["batch"] is not None:
            return
        runner = worker.model_runner
        batch = runner.input_batch
        n = int(batch.num_reqs)
        ids = list(batch.req_ids[:n])
        qsl = _ints(runner.query_start_loc.cpu[: n + 1])
        computed = _ints(batch.num_computed_tokens_cpu[:n])
        if (
            len(set(ids)) != n
            or set(ids) != set(record["admission_p_counts"])
            or len(qsl) != n + 1
            or qsl[0] != 0
            or qsl[-1] != record["p_rows"]
            or any(computed)
        ):
            raise RuntimeError("Layout requires the complete untouched P token axis")
        rows = []
        for i, rid in enumerate(ids):
            request = runner.requests[rid]
            tokens = _ints(request.prompt_token_ids)
            begin, end = qsl[i : i + 2]
            if (
                request.output_token_ids
                or end - begin != len(tokens)
                or end - begin != record["admission_p_counts"][rid]
                or _ints(runner.input_ids.cpu[begin:end]) != tokens
                or _ints(runner._positions_np_buf[begin:end])
                != list(range(len(tokens)))
            ):
                raise RuntimeError(
                    "Layout rejects partial/recomputed/mismatched P rows"
                )
            rows.append(
                {
                    "request_id": rid,
                    "prompt_sha256": _digest(tokens),
                    "prompt_tokens": len(tokens),
                    "computed_tokens": computed[i],
                    "token_start": begin,
                    "token_end": end,
                    "output_tokens": 0,
                }
            )
        record["batch"] = {
            "req_ids": ids,
            "query_start_loc": qsl,
            "rows": rows,
            "order_matches_admission_p_map": ids == list(record["admission_p_counts"]),
            "packed_tokens_and_positions_verified": True,
        }
        record["status"] = "executing"

    def _collective_wrapper(self, original):
        def transform(input_):
            self._check()
            self._verify_owner_and_batch()
            record = self.active
            if record["d_rows"] == 0:
                record["zero_d_bypass_calls"] += 1
                self.stats["zero_d_calls"] += 1
                if self.in_collective:
                    raise RuntimeError("Reentrant P layout collective")
                self.in_collective = True
                try:
                    _route(self.worker, self.group, serving=True)
                    return original(input_)
                finally:
                    self.in_collective = False
            shape = [int(value) for value in input_.shape]
            if (
                shape != [record["p_rows"], self.hidden_size]
                or str(input_.dtype) != "torch.bfloat16"
                or str(input_.device) != self.expected_device
                or not input_.is_contiguous()
            ):
                raise RuntimeError(
                    "Layout accepts only contiguous BF16 P token-axis TP reductions"
                )
            prefix = record["d_rows"]
            suffix = 0
            payload = [shape[0] + prefix + suffix, shape[1]]
            if (
                payload[0] * payload[1] * 2 > MAX_PAYLOAD_BYTES
                or len(record["calls"]) >= MAX_CALLS
            ):
                raise RuntimeError("Layout payload/per-cohort call bound exceeded")
            if self.in_collective:
                raise RuntimeError("Reentrant P layout collective")
            before = _route(self.worker, self.group, serving=True)
            call = {
                "ordinal": len(record["calls"]) + 1,
                "mode": self.mode,
                "input_shape": shape,
                "collective_payload_shape": payload,
                "prefix_rows": prefix,
                "suffix_rows": suffix,
                "dummy_rows": prefix + suffix,
                "copyback_offset": prefix,
                "dtype": str(input_.dtype),
                "device": str(input_.device),
                "existing_collective_calls": 0,
                "return_original_input": False,
                "route_before": before,
                "status": "started",
                "error": None,
            }
            if not record["calls"]:
                self.stats["treated"] += 1
            record["calls"].append(call)

            def once(scratch):
                if call["existing_collective_calls"] or list(scratch.shape) != payload:
                    raise RuntimeError(
                        "Layout collective count or actual payload differs from plan"
                    )
                call["existing_collective_calls"] += 1
                return original(scratch)

            self.in_collective = True
            try:
                result = self.runtime.transform(input_, prefix, suffix, once)
                if result is not input_ or call["existing_collective_calls"] != 1:
                    raise RuntimeError(
                        "Layout must reduce once and return original input object"
                    )
                after = _route(self.worker, self.group, serving=True)
                delta = {
                    lane: after["router_calls"][lane] - before["router_calls"][lane]
                    for lane in ("caller", "decode", "prefill")
                }
                if delta != {"caller": 0, "decode": 0, "prefill": 1}:
                    raise RuntimeError(
                        "Layout did not call exactly one existing P collective"
                    )
                call.update(
                    return_original_input=True,
                    router_call_delta=delta,
                    route_after=after,
                    status="complete",
                )
                self.stats["transformed_calls"] += 1
                return result
            except BaseException as exc:
                call["status"] = "failed"
                call["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.in_collective = False

        def wrapper(input_):
            lane = self.worker._execution_backend.active_lane
            if lane in (None, "decode"):
                return original(input_)
            try:
                if lane != "prefill":
                    raise RuntimeError(f"Unknown collective lane: {lane}")
                return transform(input_)
            except BaseException as exc:
                self._fail(exc)
                raise

        return wrapper

    def snapshot(self):
        return json.loads(
            json.dumps(
                {
                    "schema": "muxpd-prefill-collective-prefix-v1",
                    "profile": PROFILE,
                    "rank": self.rank,
                    "mode": self.mode,
                    "module_sha256": _sha(),
                    "installed_hook_count": len(self.patches),
                    "all_hooks_intact": all(
                        vars(o).get(n, _MISSING) is w for o, n, _, _, w in self.patches
                    ),
                    "hidden_size": self.hidden_size,
                    "expected_device": self.expected_device,
                    "local_rank": self.local_rank,
                    "active_cohort": None
                    if self.active is None
                    else self.active["cohort_id"],
                    "in_collective": self.in_collective,
                    "errors": list(self.errors),
                    "primary_failure": self.primary_failure,
                    "startup_route": self.startup_route,
                    "startup_boundary": (
                        "native warmup returned; capture redirect exited"
                    ),
                    "first_execute_verified": self.first_execute_verified,
                    "stats": dict(self.stats),
                    "active": self.active,
                    "recent_cohorts": list(self.cohorts),
                    "history_is_complete": self.stats["admitted"] <= HISTORY_SIZE,
                    "treated_cohorts": self.stats["treated"],
                    "uninstalled": self.uninstalled,
                    "limits": {
                        "history_size": HISTORY_SIZE,
                        "max_calls_per_cohort": MAX_CALLS,
                        "max_payload_bytes": MAX_PAYLOAD_BYTES,
                    },
                    "scope": (
                        "Experimental communication layout; no native batch "
                        "equivalence or quality/performance qualification"
                    ),
                }
            )
        )

    def _idle(self):
        worker = self.worker
        if (
            self.active is not None
            or self.in_collective
            or worker._prefill_task is not None
            or worker._prefill_cohort_id is not None
            or getattr(worker, "_awaiting_sample", False)
            or getattr(worker, "_pending_lanes", ())
            or getattr(worker, "_interleaver", None) is not None
            or worker._execution_backend.active_lane is not None
        ):
            return False
        for runner in (worker.model_runner, worker.decode_runner):
            batch = runner.input_batch
            if (
                runner.requests
                or batch.num_reqs
                or batch.req_id_to_index
                or any(rid is not None for rid in batch.req_ids)
                or getattr(runner, "execute_model_state", None) is not None
            ):
                return False
        return True

    def _restored(self):
        return all(
            (name not in vars(owner) and getattr(owner, name) == original)
            if slot is _MISSING
            else vars(owner).get(name, _MISSING) is slot
            for owner, name, slot, original, _ in self.patches
        )

    def uninstall(self, *, after_drain=False):
        if get_ident() != self.thread:
            raise RuntimeError("Layout uninstall requires owning host thread")
        worker = self.worker
        if (
            self.active is not None
            or self.in_collective
            or worker._prefill_task is not None
            or worker._prefill_cohort_id is not None
            or worker._interleaver is not None
            or worker._execution_backend.active_lane is not None
            or (not after_drain and not self._idle())
        ):
            raise RuntimeError("Layout uninstall requires aborted/drained idle worker")
        # Preflight every slot before changing any; accept already-restored slots
        # on retry, but never overwrite a foreign wrapper.
        pending = []
        for owner, name, slot, original, wrapper in self.patches:
            current = vars(owner).get(name, _MISSING)
            if current is wrapper:
                pending.append((owner, name, slot))
            elif not (
                current is slot
                and (slot is not _MISSING or getattr(owner, name) == original)
            ):
                raise RuntimeError(f"Layout slot changed externally: {name}")
        for owner, name, slot in reversed(pending):
            if slot is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, slot)
            self.restored_count += 1
        self.uninstalled = True
        if after_drain:
            self.runtime.failed_buffers = ()
        return {
            "schema": "muxpd-prefill-collective-prefix-uninstall-v1",
            "rank": self.rank,
            "mode": self.mode,
            "module_sha256": _sha(),
            "installed_hook_count": len(self.patches),
            "uninstalled": self.uninstalled,
            "in_collective": self.in_collective,
            "active_cohort": None if self.active is None else self.active["cohort_id"],
            "restored_count": self.restored_count,
            "all_originals_restored": self._restored(),
            "primary_failure": self.primary_failure,
            "stats": dict(self.stats),
            "retained_history_size": len(self.cohorts),
            "history_is_complete": self.stats["admitted"] <= HISTORY_SIZE,
            "collection_success_not_inferred_from_cleanup": True,
        }


def validate_prefix_engine_config(config):
    """Check the supported architectural profile without accelerator imports."""
    scheduler, model, parallel = (
        config.scheduler_config,
        config.model_config,
        config.parallel_config,
    )
    raw = scheduler.batch_admission_policy_config or {}
    if raw.get(POLICY_KEY, "off") == "off":
        return
    errors = []
    for value, expected, label in (
        (
            scheduler.scheduler_cls,
            "vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler",
            "scheduler",
        ),
        (
            parallel.worker_cls,
            "vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker",
            "worker",
        ),
    ):
        actual = (
            value
            if isinstance(value, str)
            else (f"{value.__module__}.{value.__qualname__}")
        )
        if actual != expected:
            errors.append(f"requires exact synchronous {label} class")
    if parallel.tensor_parallel_size != 4 or not parallel.enable_expert_parallel:
        errors.append("requires TP4/EP4")
    for name in (
        "use_sequence_parallel_moe",
        "enable_dbo",
        "use_ubatching",
        "enable_eplb",
        "enable_elastic_ep",
    ):
        if getattr(parallel, name, False):
            errors.append(f"{name} must be disabled")
    if str(model.dtype) != "torch.bfloat16":
        errors.append("requires BF16")
    text_config = model.hf_text_config
    # Architectural family only. This is not a weight identity or numerical
    # equivalence assertion; each checkpoint still needs an independent bridge.
    profile = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    if any(
        getattr(text_config, name, None) != value for name, value in profile.items()
    ):
        errors.append("requires the Qwen3.5-35B-A3B architectural profile")
    compilation = config.compilation_config
    if (
        list(compilation.cudagraph_capture_sizes or ()) != [1, 2, 4, 8]
        or compilation.max_cudagraph_capture_size != 8
    ):
        errors.append("requires decode graph capture sizes [1,2,4,8]")
    from .decode_graph import validate_decode_graph_config

    validate_decode_graph_config(config, raw)
    for name in ("lora_config", "quant_config", "weight_transfer_config"):
        if getattr(config, name, None) is not None:
            errors.append(f"{name} is unsupported")
    if getattr(model, "quantization", None) is not None:
        errors.append("quantization is unsupported")
    if getattr(config, "use_v2_model_runner", False):
        errors.append("requires V1 model runner")
    if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "0") != "0":
        errors.append("elastic EP startup is unsupported")
    if errors:
        raise ValueError("MuxPD prefix layout: " + "; ".join(errors))


def install_on_worker(worker):
    """Called only after native warmup and capture redirection have returned."""
    from .config import validate_engine_config
    from .determinism import validate_profile_environment

    validate_engine_config(worker.vllm_config)
    policy = worker.vllm_config.scheduler_config.batch_admission_policy_config or {}
    if policy.get(POLICY_KEY, "off") != "prefix":
        raise RuntimeError("Layout installation requires explicit prefix selection")
    validate_profile_environment(policy)
    if getattr(worker, ATTRIBUTE, None) is not None:
        raise RuntimeError("Layout installation already attempted")
    # Class identity rejects atomic/async and subclasses, including wrappers
    # with the same execution_mode string. Import only after CPU validation.
    from .decoupled_worker import DecoupledMuxPDNPUWorker
    from .worker import NPUModelRunner, _SharedCacheRunner

    if (
        type(worker) is not DecoupledMuxPDNPUWorker
        or type(worker.model_runner) is not NPUModelRunner
        or type(worker.decode_runner) is not _SharedCacheRunner
    ):
        raise RuntimeError("Layout requires the exact synchronous worker")
    if (
        worker.execution_mode != "decoupled"
        or worker._stream_mode != "overlap"
        or worker._overlap_communication != "isolated"
        or not worker._decode_graph_enabled
        or worker._prefill_groups_enabled
        or worker._layers_per_slice != 4
        or getattr(worker, "_muxpd_layer_capture_v3", None) is not None
    ):
        raise RuntimeError("Layout requires unobserved raw isolated q4 execution")
    graph = worker._decode_graph_controller.snapshot()
    if (
        not graph["captured"]
        or graph["failed"]
        or graph["closed"]
        or graph["compile_full_prefill"]
    ):
        raise RuntimeError("Install layout only after successful raw-P graph startup")
    from vllm.distributed import get_ep_group, get_tp_group

    group, ep = get_tp_group(), get_ep_group()
    if (
        group.world_size != 4
        or ep.world_size != 4
        or group.ranks[group.rank_in_group] != worker.rank
        or list(ep.ranks) != list(group.ranks)
        or not group.use_custom_op_call
    ):
        raise RuntimeError("Layout requires original opaque TP4 and matching EP4")
    layout = Layout(
        worker,
        "prefix",
        group,
        TorchRuntime(),
        worker.vllm_config.model_config.hf_text_config.hidden_size,
    )
    if not layout._idle():
        raise RuntimeError("Layout installation requires idle runners and lanes")
    setattr(worker, ATTRIBUTE, layout)
    try:
        layout.install()
    except BaseException as exc:
        layout._fail(exc)
        raise
    result = layout.snapshot()
    result["decode_graph"] = graph
    return result


def shutdown_on_worker(worker):
    """Terminal drain before graph/router/group cleanup; failures keep ownership.

    The decoupled worker has already aborted its P coroutine. Outstanding
    sampling state is discarded only by terminal worker destruction, never
    acknowledged as successful work or passed back for scheduler KV reuse.
    """
    layout = getattr(worker, ATTRIBUTE, None)
    if layout is None:
        return None
    for stream in worker._streams.values():
        stream.synchronize()
    update_stream = getattr(worker.decode_runner, "update_stream", None)
    if update_stream is not None:
        update_stream.synchronize()
    return layout.uninstall(after_drain=True)


def snapshot_on_worker(argument):
    layout = getattr(_unwrap(argument), ATTRIBUTE, None)
    return None if layout is None else layout.snapshot()
