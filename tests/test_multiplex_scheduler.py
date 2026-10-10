"""Check layered admission against the real host allocator without a model load."""

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("vllm")

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
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_muxpd.multiplex_scheduler import LayeredMuxPDScheduler


@pytest.fixture
def make_layered_scheduler(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 4,
        "max_position_embeddings": 256, "vocab_size": 128,
    }))
    schedulers = []

    def make(*, num_blocks=64, max_num_seqs=8, options=None):
        config = VllmConfig(
            model_config=ModelConfig(
                model=str(model_path), max_model_len=256, dtype="float16",
                skip_tokenizer_init=True, enforce_eager=True,
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=256, max_num_seqs=max_num_seqs,
                is_encoder_decoder=False, max_num_batched_tokens=256,
                enable_chunked_prefill=False, async_scheduling=False,
                watermark=0.0,
                scheduler_cls=(
                    "vllm_muxpd.multiplex_scheduler.LayeredMuxPDScheduler"
                ),
                batch_admission_policy="vllm_muxpd.policy.MuxPDPolicy",
                batch_admission_policy_config={
                    "execution": "layered", "max_decode_steps": 1,
                    "tpot_slo_ms": 0.001,
                    "trace_path": str(tmp_path / "trace.jsonl"),
                    **(options or {}),
                },
            ),
            cache_config=CacheConfig(
                block_size=16, enable_prefix_caching=False,
                mamba_cache_mode="none",
            ),
            parallel_config=ParallelConfig(
                worker_cls="vllm_muxpd.worker.MuxPDNPUWorker",
            ),
        )
        config.cache_config.num_gpu_blocks = num_blocks
        kv_config = KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=16, dtype=torch.float16,
            ))],
        )
        init_none_hash(sha256)
        register_all_kvcache_specs(config)
        scheduler = LayeredMuxPDScheduler(
            vllm_config=config, kv_cache_config=kv_config, block_size=16,
            structured_output_manager=StructuredOutputManager(config),
        )
        schedulers.append(scheduler)
        return scheduler

    yield make
    for scheduler in schedulers:
        scheduler.shutdown()


def request(rid, tokens=8, max_tokens=8):
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, 127)
    return Request(rid, [1] * tokens, params, None)


def complete(scheduler, output):
    ids = list(output.num_scheduled_tokens)
    return scheduler.update_from_output(output, ModelRunnerOutput(
        req_ids=ids, req_id_to_index={rid: index for index, rid in enumerate(ids)},
        sampled_token_ids=[[2] for _ in ids], logprobs=None,
        prompt_logprobs_dict={}, pooler_output=[],
    ))


def test_separate_lanes_and_one_token_prompt_classification(make_layered_scheduler):
    scheduler = make_layered_scheduler()
    scheduler.add_request(request("decode", 16))
    first = scheduler.schedule()
    complete(scheduler, first)
    scheduler.add_request(request("one-token-prefill", 1))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"decode": 1, "one-token-prefill": 1}
    assert output.muxpd_prefill_request_ids == {"one-token-prefill"}
    assert output.muxpd_phase == "multiplex"
    complete(scheduler, output)
    assert scheduler.export_muxpd_stats()["completed_transactions"] == 2


def test_complete_prefill_hard_budget_defers_other_prompt(make_layered_scheduler):
    scheduler = make_layered_scheduler()
    scheduler.add_request(request("decode"))
    complete(scheduler, scheduler.schedule())
    scheduler.add_request(request("prefill", 180))
    scheduler.add_request(request("later", 100))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"decode": 1, "prefill": 180}
    assert scheduler.requests["later"].num_computed_tokens == 0
    complete(scheduler, output)


def test_full_slots_keep_decode_progressing(make_layered_scheduler):
    scheduler = make_layered_scheduler(max_num_seqs=1)
    scheduler.add_request(request("decode"))
    complete(scheduler, scheduler.schedule())
    scheduler.add_request(request("waiting"))
    scheduler.muxpd_policy.decode_steps = 100
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"decode": 1}
    complete(scheduler, output)


def test_long_prefill_gets_its_own_complete_transaction(make_layered_scheduler):
    scheduler = make_layered_scheduler()
    scheduler.add_request(request("a"))
    scheduler.add_request(request("b"))
    complete(scheduler, scheduler.schedule())
    scheduler.add_request(request("long", 255))
    scheduler.muxpd_policy.decode_steps = 1
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"long": 255}
    assert output.muxpd_phase == "prefill"
    complete(scheduler, output)


def test_p_only_kv_stall_falls_back_to_decode(make_layered_scheduler):
    # 17 usable blocks: a and b hold one each; long needs 16. Its 255 tokens
    # cannot fit in the 256-token budget together with the two decode tokens.
    scheduler = make_layered_scheduler(num_blocks=18)
    scheduler.add_request(request("a"))
    scheduler.add_request(request("b"))
    complete(scheduler, scheduler.schedule())
    scheduler.add_request(request("long", 255))
    scheduler.add_request(request("cancelled"))
    scheduler.finish_requests("cancelled", RequestStatus.FINISHED_ABORTED)
    scheduler.muxpd_policy.decode_steps = 1
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"a": 1, "b": 1}
    assert output.finished_req_ids == {"cancelled"}
    assert scheduler.export_muxpd_stats()["prefill_capacity_deferrals"] == 1
    complete(scheduler, output)


def test_impossible_complete_prefill_fails_without_spinning(make_layered_scheduler):
    scheduler = make_layered_scheduler(num_blocks=2)
    scheduler.add_request(request("impossible", 32))
    with pytest.raises(RuntimeError, match="complete prefill"):
        scheduler.schedule()


def test_preemption_recomputes_context_and_retains_phase(make_layered_scheduler):
    scheduler = make_layered_scheduler(num_blocks=3)
    scheduler.add_request(request("a", 16))
    scheduler.add_request(request("b", 16))
    complete(scheduler, scheduler.schedule())
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"a": 1}
    assert output.preempted_req_ids == {"b"}
    complete(scheduler, output)
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"b": 17}
    assert output.muxpd_prefill_request_ids == {"b"}
    # MRv2 re-registers resumed requests while MRv1 uses cached request data.
    resumed = output.scheduled_cached_reqs.resumed_req_ids
    registered = {row.req_id for row in output.scheduled_new_reqs}
    assert "b" in resumed | registered
    complete(scheduler, output)
    assert scheduler.export_muxpd_stats()["preemptions"] == 1


def test_lifecycle_only_step_has_no_compute_phase(make_layered_scheduler):
    scheduler = make_layered_scheduler()
    scheduler.add_request(request("cancelled"))
    scheduler.finish_requests("cancelled", RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    assert not output.num_scheduled_tokens
    assert output.finished_req_ids == {"cancelled"}
    assert output.muxpd_phase is None
    assert scheduler.last_muxpd_step["phase"] is None
    assert scheduler._layered_pending is None


def test_partial_native_recomputation_fails_closed(make_layered_scheduler):
    scheduler = make_layered_scheduler()
    req = request("resumed", 20)
    scheduler.add_request(req)
    complete(scheduler, scheduler.schedule())
    scheduler.running.remove(req)
    scheduler._preempt_request(req, 0.0)
    scheduler.scheduler_config.long_prefill_token_threshold = 8
    with pytest.raises(RuntimeError, match="partial"):
        scheduler.schedule()


def test_empty_update_cannot_retire_other_pending_transaction(make_layered_scheduler):
    scheduler = make_layered_scheduler()
    empty_output = scheduler.schedule()
    scheduler.add_request(request("a"))
    pending = scheduler.schedule()
    with pytest.raises(RuntimeError, match="Unknown/reordered"):
        complete(scheduler, empty_output)
    assert scheduler._layered_pending[0] is pending
    complete(scheduler, pending)


def test_trace_exposes_real_phase_and_full_token_cost(make_layered_scheduler):
    scheduler = make_layered_scheduler()
    scheduler.add_request(request("a"))
    complete(scheduler, scheduler.schedule())
    scheduler.add_request(request("b", 17))
    complete(scheduler, scheduler.schedule())
    rows = [json.loads(line) for line in Path(
        scheduler.muxpd_policy.config.trace_path,
    ).read_text().splitlines()]
    scheduled = [row for row in rows if row["event"] == "schedule"]
    assert scheduled[-1]["phase"] == "multiplex"
    assert scheduled[-1]["request_phases"] == {"a": "decode", "b": "prefill"}
    for row in scheduled:
        assert row["scheduled_tokens"] == row["expected_tokens"]
    assert len([row for row in rows if row["event"] == "complete"]) == 2


def test_pause_new_defers_waiting_and_pause_all_has_no_work(make_layered_scheduler):
    scheduler = make_layered_scheduler()
    scheduler.add_request(request("running"))
    complete(scheduler, scheduler.schedule())
    scheduler.add_request(request("waiting"))
    scheduler._pause_state = PauseState.PAUSED_NEW
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"running": 1}
    complete(scheduler, output)
    scheduler._pause_state = PauseState.PAUSED_ALL
    output = scheduler.schedule()
    assert not output.num_scheduled_tokens
    assert output.muxpd_phase is None
