"""Pinned native sampler bodies with CPU tensors/fakes; never allocate an NPU.

Importing the pinned runtime is intentionally gated. Run only in an explicitly
approved CPU-test window, not during clean serving measurements.
"""

import importlib
import os
from contextlib import nullcontext
from types import MethodType
from types import SimpleNamespace as NS

import pytest
from test_decode_sampling import Runner, Runtime, Tensor

from vllm_muxpd.decode_sampling import DecodeSamplePipeline


@pytest.mark.integration
def test_real_native_sampler_retains_bookkeeping_and_hybrid_feedback(monkeypatch):
    if os.environ.get("VLLM_VERSION") != "0.25.1":
        pytest.skip("source runtime/env.sh for pinned HUST sampler contracts")
    # Use the native worker's import order; directly importing its runner first
    # enters Ascend's DeviceOperator / fused-MoE circular-import path.
    importlib.import_module("vllm_ascend.worker.worker")
    module = importlib.import_module("vllm_ascend.worker.model_runner_v1")
    import numpy as np

    events = []
    runner = Runner(events)
    native = module.NPUModelRunner
    runner.sample_tokens = MethodType(native.sample_tokens, runner)
    runner._sample = MethodType(native._sample, runner)
    runner._bookkeeping_sync = MethodType(native._bookkeeping_sync, runner)
    runner.need_accepted_tokens = True
    runner.sampling_done_event = NS(
        record=lambda: events.append("native_sampling_done_event_record")
    )
    runner.supports_mm_inputs = False
    runner.dynamic_eplb = False
    runner.ascend_config = NS(scheduler_config=NS(
        profiling_chunk_config=NS(need_timing=False)
    ))
    runner.max_model_len = 16
    runner.discard_request_indices = NS(np=np.array([], dtype=np.int32))
    runner._get_prompt_logprobs_dict = lambda *args: {}
    runner._finalize_dump_data = lambda: events.append("native_finalize")
    runner._update_states_after_model_execute = (
        lambda *args: events.append("native_hybrid_feedback")
    )
    batch = runner.input_batch
    batch.update_async_output_token_ids = lambda: None
    batch.generators = {}
    batch.num_tokens_no_spec = np.array([3], dtype=np.int32)
    batch.num_tokens = np.array([3], dtype=np.int32)
    batch.token_ids_cpu = np.zeros((1, 16), dtype=np.int32)
    batch.is_token_ids = np.zeros((1, 16), dtype=bool)
    batch.sampling_metadata.top_k = None
    batch.sampling_metadata.output_token_ids = [runner.requests["d"].output_token_ids]
    sampled = NS(sampled_token_ids=Tensor([[7]]), logprobs_tensors=None)

    def sampler(**kwargs):
        assert kwargs["sampling_metadata"] is batch.sampling_metadata
        events.append("native_sample_device_submit")
        return sampled

    runner.sampler = sampler
    state = module.ExecuteModelState(
        NS(num_scheduled_tokens={"d": 1}, total_num_scheduled_tokens=1),
        Tensor([[0, 1]]), None, None, Tensor([[0, 1]]), None,
        None, None, None, None, None, None,
    )
    runner.execute_model_state = state
    feedback_stream = NS(wait_event=lambda event: events.append("native_feedback_wait"))
    monkeypatch.setattr(module, "get_pp_group", lambda: NS(world_size=1))
    monkeypatch.setattr(module, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(module, "global_stream", lambda: feedback_stream)
    monkeypatch.setattr(module.torch.npu, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        module, "record_function_or_nullcontext", lambda *a: nullcontext()
    )
    pipe = DecodeSamplePipeline(
        runner, Runtime(events), lambda: None, builtin_processor_types=()
    )
    assert pipe.enqueue()
    assert runner.execute_model_state is state
    assert runner.requests["d"].output_token_ids == [11]
    assert batch.num_tokens.tolist() == [3]
    assert "native_sampling_done_event_record" not in events
    events.append("prefill_slice")
    output = runner.sample_tokens(None)
    assert output.req_ids == ["d"] and output.sampled_token_ids == [[7]]
    assert runner.requests["d"].output_token_ids == [11, 7]
    assert batch.num_tokens.tolist() == [4]
    assert batch.token_ids_cpu[0, 3] == 7
    assert events == [
        "native_sample_device_submit", "copy_enqueued", "copy_event_recorded",
        "prefill_slice", "native_sampling_done_event_record", "copy_event_wait",
        "native_finalize", "native_feedback_wait", "native_hybrid_feedback",
    ]
    assert pipe.stats["enqueued"] == pipe.stats["retired"] == 1
    pipe.close()
