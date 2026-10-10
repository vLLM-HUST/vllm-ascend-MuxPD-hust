"""Exercise the actual native vLLM allocator and lifecycle without model weights."""

import json
from pathlib import Path

import pytest

pytest.importorskip("vllm")
import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_muxpd.scheduler import MuxPDScheduler


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-llama-config")
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 256,
                "vocab_size": 128,
            }
        )
    )
    return path


@pytest.fixture
def make_scheduler(tiny_model, tmp_path):
    schedulers = []

    def make(*, num_blocks=64, max_num_seqs=8, options=None):
        model = ModelConfig(
            model=str(tiny_model),
            max_model_len=256,
            dtype="float16",
            skip_tokenizer_init=True,
        )
        config = VllmConfig(
            model_config=model,
            scheduler_config=SchedulerConfig(
                max_model_len=256,
                max_num_seqs=max_num_seqs,
                is_encoder_decoder=False,
                max_num_batched_tokens=256,
                enable_chunked_prefill=False,
                async_scheduling=False,
                watermark=0.0,
                scheduler_cls="vllm_muxpd.scheduler.MuxPDScheduler",
                batch_admission_policy="vllm_muxpd.policy.MuxPDPolicy",
                batch_admission_policy_config={
                    "trace_path": str(tmp_path / "trace.jsonl"),
                    "tpot_slo_ms": 0.001,
                    "max_decode_steps": 1,
                    **(options or {}),
                },
            ),
            cache_config=CacheConfig(
                block_size=16,
                enable_prefix_caching=False,
                mamba_cache_mode="none",
            ),
            parallel_config=ParallelConfig(),
        )
        config.cache_config.num_gpu_blocks = num_blocks
        kv_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=16,
                        num_kv_heads=1,
                        head_size=16,
                        dtype=torch.float16,
                    ),
                )
            ],
        )
        init_none_hash(sha256)
        register_all_kvcache_specs(config)
        scheduler = MuxPDScheduler(
            vllm_config=config,
            kv_cache_config=kv_config,
            block_size=16,
            structured_output_manager=StructuredOutputManager(config),
        )
        schedulers.append(scheduler)
        return scheduler

    yield make
    for scheduler in schedulers:
        scheduler.shutdown()


def request(rid, tokens=16, max_tokens=8):
    sampling = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    sampling.update_from_generation_config({}, 127)
    return Request(rid, [1] * tokens, sampling, None)


def complete(scheduler, output, token=2):
    ids = list(output.num_scheduled_tokens)
    return scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[[token] for _ in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_schedule_both_entry_paths_separate_prefill_and_decode(make_scheduler):
    scheduler = make_scheduler()
    scheduler.add_request(request("a", 20))
    initial = scheduler.schedule()
    assert initial.num_scheduled_tokens == {"a": 20}
    complete(scheduler, initial)
    scheduler.add_request(request("b", 21))
    batch_id, decode = scheduler.schedule_batch(frozenset())
    assert batch_id and decode.num_scheduled_tokens == {"a": 1}
    complete(scheduler, decode)
    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"b": 21}
    complete(scheduler, prefill)
    assert scheduler.export_muxpd_stats()["prefill_batches"] == 2
    trace = [
        json.loads(line)
        for line in Path(scheduler.muxpd_policy.config.trace_path)
        .read_text()
        .splitlines()
    ]
    for step in (row for row in trace if row["event"] == "schedule"):
        assert step["scheduled_tokens"] == step["expected_tokens"]
        assert len(set(step["request_phases"].values())) <= 1


def test_never_chunks_multiple_full_prefills(make_scheduler):
    scheduler = make_scheduler()
    scheduler.add_request(request("a", 180))
    scheduler.add_request(request("b", 100))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"a": 180}
    assert scheduler.requests["b"].num_computed_tokens == 0
    complete(scheduler, output)


def test_preempted_request_recomputes_prompt_and_outputs(make_scheduler):
    scheduler = make_scheduler()
    req = request("a", 20)
    scheduler.add_request(req)
    output = scheduler.schedule()
    complete(scheduler, output)
    scheduler.running.remove(req)
    scheduler._preempt_request(req, 0.0)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"a": 21}
    assert scheduler.last_muxpd_step["phase"] == "prefill"
    resumed_ids = output.scheduled_cached_reqs.resumed_req_ids
    new_ids = {item.req_id for item in output.scheduled_new_reqs}
    # MRv1 uses cached/resumed data; MRv2 sends a fresh request registration.
    assert "a" in resumed_ids | new_ids
    complete(scheduler, output)


def test_native_capacity_preemption_and_recompute(make_scheduler):
    # 1 null + 2 usable blocks. Both prompts fill KV; their next decode needs
    # another block, so the native allocator must preempt one request.
    scheduler = make_scheduler(num_blocks=3)
    scheduler.add_request(request("a", 16))
    scheduler.add_request(request("b", 16))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"a": 16, "b": 16}
    complete(scheduler, output)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"a": 1}
    assert scheduler.requests["b"].status == RequestStatus.PREEMPTED
    assert scheduler.requests["b"].num_output_tokens == 1
    complete(scheduler, output)
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"b": 17}
    complete(scheduler, output)


def test_prefill_kv_stall_progresses_decode_and_preserves_notifications(make_scheduler):
    scheduler = make_scheduler(num_blocks=3)
    scheduler.add_request(request("a", 8))
    output = scheduler.schedule()
    complete(scheduler, output)
    scheduler.add_request(request("b", 20))
    scheduler.muxpd_policy.decode_steps = 1
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"a": 1}
    assert scheduler.export_muxpd_stats()["prefill_capacity_deferrals"] == 1
    complete(scheduler, output)


def test_finish_and_cancel_release_native_state(make_scheduler):
    scheduler = make_scheduler()
    scheduler.add_request(request("done", max_tokens=1))
    scheduler.add_request(request("cancelled"))
    scheduler.finish_requests("cancelled", RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    assert "cancelled" in output.finished_req_ids
    complete(scheduler, output)
    assert not scheduler.requests
    maintenance = scheduler.schedule()
    assert not maintenance.num_scheduled_tokens
    assert "done" in maintenance.finished_req_ids


def test_full_sequence_slots_do_not_stall_decode(make_scheduler):
    scheduler = make_scheduler(max_num_seqs=1)
    scheduler.add_request(request("a"))
    first = scheduler.schedule()
    complete(scheduler, first)
    scheduler.add_request(request("b"))
    scheduler.muxpd_policy.decode_steps = 100
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"a": 1}
    complete(scheduler, output)


def test_impossible_full_prefill_fails_instead_of_spinning(make_scheduler):
    scheduler = make_scheduler(num_blocks=2)
    scheduler.add_request(request("too-large", 32))
    with pytest.raises(RuntimeError, match="complete prefill"):
        scheduler.schedule()


def test_inflight_and_duplicate_schedule_fail_closed(make_scheduler):
    scheduler = make_scheduler()
    with pytest.raises(RuntimeError, match="synchronous"):
        scheduler.schedule_batch(frozenset({"other"}))
    scheduler.add_request(request("a"))
    output = scheduler.schedule()
    with pytest.raises(RuntimeError, match="update_from_output"):
        scheduler.schedule()
    complete(scheduler, output)


def test_native_partial_allocation_is_rejected(make_scheduler, monkeypatch):
    scheduler = make_scheduler()
    scheduler.add_request(request("a", 20))
    # Simulate an upstream regression that starts splitting full prefills.
    scheduler.scheduler_config.long_prefill_token_threshold = 8
    with pytest.raises(RuntimeError, match="refuses partial"):
        scheduler.schedule()


def test_stale_empty_output_cannot_retire_pending_batch(make_scheduler):
    scheduler = make_scheduler()
    stale_empty_output = scheduler.schedule()
    scheduler.add_request(request("a"))
    pending_output = scheduler.schedule()
    with pytest.raises(RuntimeError, match="unknown or reordered"):
        complete(scheduler, stale_empty_output)
    assert scheduler._muxpd_pending[0] is pending_output
    complete(scheduler, pending_output)
    assert scheduler._muxpd_pending is None
