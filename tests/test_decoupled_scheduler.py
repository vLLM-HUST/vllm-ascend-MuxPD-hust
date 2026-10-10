"""Exercise cross-iteration prefill ownership with the native vLLM allocator."""

import json
import time
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

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
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_muxpd.decoupled_scheduler import DecoupledMuxPDScheduler


@pytest.fixture
def make_scheduler(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 256,
                "vocab_size": 128,
            }
        )
    )
    schedulers = []

    def make(
        *, num_blocks=64, max_num_seqs=8, layers_per_slice=1, adaptive_layers=False
    ):
        config = VllmConfig(
            model_config=ModelConfig(
                model=str(model_path),
                max_model_len=256,
                dtype="float16",
                skip_tokenizer_init=True,
                enforce_eager=True,
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=256,
                max_num_seqs=max_num_seqs,
                is_encoder_decoder=False,
                max_num_batched_tokens=256,
                enable_chunked_prefill=False,
                async_scheduling=False,
                watermark=0.0,
                scheduler_cls=(
                    "vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler"
                ),
                batch_admission_policy="vllm_muxpd.policy.MuxPDPolicy",
                batch_admission_policy_config={
                    "execution": "decoupled",
                    "layers_per_slice": layers_per_slice,
                    "adaptive_layers": adaptive_layers,
                    "max_decode_steps": 1,
                    "tpot_slo_ms": 0.001,
                    "trace_path": str(tmp_path / "trace.jsonl"),
                },
            ),
            cache_config=CacheConfig(
                block_size=16,
                enable_prefix_caching=False,
                mamba_cache_mode="none",
            ),
            parallel_config=ParallelConfig(
                worker_cls="vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker",
            ),
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
        scheduler = DecoupledMuxPDScheduler(
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


def request(rid, tokens=8, max_tokens=32, client=0):
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, 127)
    return Request(rid, [1] * tokens, params, None, client_index=client)


def receipt(output, *, completed=(), cancelled=()):
    ids = [
        rid
        for rid in output.num_scheduled_tokens
        if rid not in output.muxpd_prefill_request_ids
    ] + sorted(completed)
    result = ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: index for index, rid in enumerate(ids)},
        sampled_token_ids=[[index + 2] for index in range(len(ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    result.muxpd_prefill_cohort_id = output.muxpd_prefill_cohort_id
    result.muxpd_completed_prefill_ids = frozenset(completed)
    result.muxpd_cancelled_prefill_ids = frozenset(cancelled)
    total = output.muxpd_prefill_layers_total
    progress = output.muxpd_prefill_layers_completed_before
    pending = set(output.muxpd_pending_prefill_ids)
    if (
        pending
        and not output.muxpd_pause_prefill
        and (set(output.muxpd_cancel_prefill_request_ids) != pending)
    ):
        progress = (
            total
            if output.muxpd_finish_prefill
            else min(total, progress + output.muxpd_layers_per_slice)
        )
    result.muxpd_prefill_layers_completed = progress
    result.muxpd_prefill_layers_total = total
    return result


def retire(scheduler, output, **kwargs):
    return scheduler.update_from_output(output, receipt(output, **kwargs))


def start_decode(scheduler, tokens=8, max_tokens=32, client=0):
    req = request("d", tokens, max_tokens, client)
    scheduler.add_request(req)
    output = scheduler.schedule()
    assert output.muxpd_finish_prefill
    retire(scheduler, output, completed={"d"})
    return req


def output_ids(outputs):
    return {row.request_id for packet in outputs.values() for row in packet.outputs}


def test_decode_retires_each_iteration_before_prefill_finishes(make_scheduler):
    scheduler = make_scheduler()
    decode = start_decode(scheduler)
    prefill = request("p", 64)
    scheduler.add_request(prefill)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1, "p": 64}
    cohort_id = output.muxpd_prefill_cohort_id
    assert output_ids(retire(scheduler, output)) == {"d"}
    assert prefill.num_computed_tokens == 64 and prefill.num_output_tokens == 0
    assert decode.num_output_tokens == 2
    for expected_decode_tokens in (3, 4):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {"d": 1}
        assert not output.muxpd_prefill_request_ids
        assert output.muxpd_pending_prefill_ids == {"p"}
        assert output.muxpd_prefill_cohort_id == cohort_id
        assert output_ids(retire(scheduler, output)) == {"d"}
        assert decode.num_output_tokens == expected_decode_tokens
        assert prefill.num_computed_tokens == 64 and prefill.num_output_tokens == 0
    output = scheduler.schedule()
    assert output_ids(retire(scheduler, output, completed={"p"})) == {"d", "p"}
    assert prefill.num_output_tokens == 1
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1, "p": 1}
    assert not output.muxpd_pending_prefill_ids
    retire(scheduler, output)


def test_no_decode_zero_token_step_finishes_remaining_prefill(make_scheduler):
    scheduler = make_scheduler()
    start_decode(scheduler, max_tokens=2)
    scheduler.add_request(request("p", 64))
    output = scheduler.schedule()
    retire(scheduler, output)  # d reaches its output limit; p remains pending.
    assert "d" not in scheduler.requests and scheduler.has_requests()
    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens == 0
    assert output.muxpd_finish_prefill and output.muxpd_pending_prefill_ids == {"p"}
    assert output_ids(retire(scheduler, output, completed={"p"})) == {"p"}
    assert scheduler._prefill_cohort is None


def test_pending_prefill_reserves_slots_and_prevents_second_cohort(make_scheduler):
    scheduler = make_scheduler(max_num_seqs=2, layers_per_slice=2)
    start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    retire(scheduler, scheduler.schedule())
    scheduler.add_request(request("later", 20))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1}
    assert scheduler.requests["later"].num_computed_tokens == 0
    assert scheduler.max_num_running_reqs == 2
    assert {r.request_id for r in scheduler.running} == {"d", "p"}
    retire(scheduler, output, completed={"p"})


def test_native_preemption_cannot_evict_pending_prefill(make_scheduler):
    # Three usable blocks: d's first decode fills its first block while p holds
    # two. The next decode needs a second block, so only d may be preempted.
    scheduler = make_scheduler(num_blocks=4)
    start_decode(scheduler, tokens=15)
    scheduler.add_request(request("p", 32))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1, "p": 32}
    retire(scheduler, output)
    output = scheduler.schedule()
    assert output.preempted_req_ids == {"d"}
    assert output.total_num_scheduled_tokens == 0 and output.muxpd_finish_prefill
    assert scheduler.requests["p"].status == RequestStatus.RUNNING
    assert scheduler.requests["p"].num_computed_tokens == 32
    retire(scheduler, output, completed={"p"})


def test_partial_cancellation_pins_blocks_until_delayed_ack(make_scheduler):
    scheduler = make_scheduler()
    start_decode(scheduler)
    scheduler.add_request(request("p1", 32))
    scheduler.add_request(request("p2", 32))
    retire(scheduler, scheduler.schedule())
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    scheduler.finish_requests("p1", RequestStatus.FINISHED_ABORTED)
    assert "p1" not in scheduler.requests
    held = scheduler._held_prefill_blocks["p1"]
    assert len(held) == 2 and all(block.ref_cnt > 0 for block in held)
    assert pool.get_num_free_blocks() == free_before
    output = scheduler.schedule()
    assert output.muxpd_cancel_prefill_request_ids == {"p1"}
    assert output.muxpd_pending_prefill_ids == {"p1", "p2"}
    assert "p1" not in output.finished_req_ids
    retire(scheduler, output)  # worker cannot remove one row from a suspended P.
    assert pool.get_num_free_blocks() == free_before
    retire(scheduler, scheduler.schedule())
    output = scheduler.schedule()
    result = retire(scheduler, output, completed={"p2"}, cancelled={"p1"})
    assert output_ids(result) == {"d", "p2"}
    assert not scheduler._held_prefill_blocks
    assert all(block.ref_cnt == 0 for block in held)
    assert pool.get_num_free_blocks() == free_before + 2


def test_all_cancelled_keeps_engine_alive_until_ack_and_cleanup(make_scheduler):
    scheduler = make_scheduler()
    start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    retire(scheduler, scheduler.schedule())
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    assert not scheduler.requests and scheduler.has_requests()
    output = scheduler.schedule()
    assert not output.num_scheduled_tokens
    assert output.muxpd_cancel_prefill_request_ids == {"p"}
    assert "p" not in output.finished_req_ids
    retire(scheduler, output, cancelled={"p"})
    assert scheduler._prefill_cohort is None
    assert not scheduler._held_prefill_blocks
    output = scheduler.schedule()
    assert "p" in output.finished_req_ids
    retire(scheduler, output)
    assert not scheduler.has_requests()


def test_completion_racing_with_abort_drops_sample_and_unpins(make_scheduler):
    scheduler = make_scheduler(layers_per_slice=4)
    start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    output = scheduler.schedule()
    model_output = receipt(output, completed={"p"})
    # EngineCore processes aborts between worker completion and scheduler update.
    scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    held = scheduler._held_prefill_blocks["p"]
    result = scheduler.update_from_output(output, model_output)
    assert output_ids(result) == {"d"}
    assert all(block.ref_cnt == 0 for block in held)
    assert scheduler._prefill_cohort is None


def test_first_token_limit_frees_completed_prefill_normally(make_scheduler):
    scheduler = make_scheduler(layers_per_slice=4)
    start_decode(scheduler)
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    scheduler.add_request(request("p", 32, max_tokens=1))
    output = scheduler.schedule()
    retire(scheduler, output, completed={"p"})
    assert "p" not in scheduler.requests
    assert not scheduler._held_prefill_blocks
    assert pool.get_num_free_blocks() == free_before


@pytest.mark.parametrize("clear_cache", [False, True])
def test_keep_pause_rejects_before_state_or_cache_changes(make_scheduler, clear_cache):
    scheduler = make_scheduler()
    start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    retire(scheduler, scheduler.schedule())
    reset = Mock()
    engine = SimpleNamespace(scheduler=scheduler, _reset_caches=reset)
    with pytest.raises(RuntimeError, match="Cannot keep-pause"):
        EngineCore.pause_scheduler(engine, mode="keep", clear_cache=clear_cache)
    assert scheduler.pause_state == PauseState.UNPAUSED
    reset.assert_not_called()
    assert scheduler.has_requests()


def test_pause_new_allows_pending_cancellation_ack(make_scheduler):
    scheduler = make_scheduler()
    start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    retire(scheduler, scheduler.schedule())
    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    assert scheduler.has_requests()
    output = scheduler.schedule()
    retire(scheduler, output, cancelled={"p"})
    assert not scheduler._held_prefill_blocks


@pytest.mark.parametrize("cancel", [False, True])
def test_native_engine_step_executes_zero_token_prefill_controls(
    make_scheduler, cancel
):
    scheduler = make_scheduler()
    start_decode(scheduler, max_tokens=2)
    scheduler.add_request(request("p", 32))
    retire(scheduler, scheduler.schedule())
    if cancel:
        scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    dispatched = []

    def execute(output, non_block):
        assert non_block and output.total_num_scheduled_tokens == 0
        dispatched.append(output)
        result = Future()
        result.set_result(None)
        return result

    def sample(grammar):
        assert grammar is None
        return receipt(
            dispatched[-1],
            completed=() if cancel else {"p"},
            cancelled={"p"} if cancel else (),
        )

    engine = SimpleNamespace(
        scheduler=scheduler,
        model_executor=SimpleNamespace(execute_model=execute, sample_tokens=sample),
        _should_throttle_prefills=lambda: False,
        log_error_detail=lambda output: nullcontext(),
        log_iteration_details=lambda output: nullcontext(),
        _process_aborts_queue=lambda: None,
    )
    outputs, counted_tokens = EngineCore.step(engine)
    assert len(dispatched) == 1 and not counted_tokens
    assert output_ids(outputs) == (set() if cancel else {"p"})
    assert scheduler._prefill_cohort is None
    assert not scheduler._held_prefill_blocks


def test_full_cancellation_requires_worker_ack(make_scheduler):
    scheduler = make_scheduler()
    start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    retire(scheduler, scheduler.schedule())
    scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    with pytest.raises(RuntimeError, match="full cohort cancellation"):
        retire(scheduler, output)
    assert scheduler._held_prefill_blocks
    retire(scheduler, output, cancelled={"p"})


def test_reset_and_request_id_reuse_cannot_invalidate_pending_state(make_scheduler):
    scheduler = make_scheduler()
    start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    retire(scheduler, scheduler.schedule())
    with pytest.raises(RuntimeError, match="pending prefill"):
        scheduler.reset_prefix_cache(reset_running_requests=True)
    scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    with pytest.raises(ValueError, match="awaiting prefill retirement"):
        scheduler.add_request(request("p"))
    output = scheduler.schedule()
    retire(scheduler, output, cancelled={"p"})
    scheduler.add_request(request("p"))


@pytest.mark.parametrize("fault", ["cohort", "unknown", "live_cancel", "samples"])
def test_bad_receipts_fail_before_retiring_decode(make_scheduler, fault):
    scheduler = make_scheduler(layers_per_slice=4)
    decode = start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    output = scheduler.schedule()
    bad = receipt(output)
    if fault == "cohort":
        bad.muxpd_prefill_cohort_id += 1
    elif fault == "unknown":
        bad.muxpd_completed_prefill_ids = {"unknown"}
    elif fault == "live_cancel":
        bad.muxpd_cancelled_prefill_ids = {"p"}
    else:
        bad.req_ids.append("p")
        bad.req_id_to_index["p"] = 1
        bad.sampled_token_ids.append([3])
    with pytest.raises(RuntimeError):
        scheduler.update_from_output(output, bad)
    assert decode.num_output_tokens == 1
    assert scheduler.requests["p"].num_output_tokens == 0
    assert scheduler._decoupled_step.output is output
    retire(scheduler, output, completed={"p"})


def test_phase_outputs_merge_per_client(make_scheduler):
    scheduler = make_scheduler(layers_per_slice=4)
    start_decode(scheduler, client=5)
    scheduler.add_request(request("p", 32, client=7))
    output = scheduler.schedule()
    result = retire(scheduler, output, completed={"p"})
    assert output_ids({5: result[5]}) == {"d"}
    assert output_ids({7: result[7]}) == {"p"}


def test_duplicate_step_and_queued_execution_are_rejected(make_scheduler):
    scheduler = make_scheduler()
    start_decode(scheduler)
    output = scheduler.schedule()
    with pytest.raises(RuntimeError, match="synchronous worker step"):
        scheduler.schedule()
    retire(scheduler, output)
    with pytest.raises(RuntimeError, match="Unknown/reordered"):
        retire(scheduler, output)
    with pytest.raises(RuntimeError, match="batch queue"):
        scheduler.schedule_batch(frozenset())


def test_trace_records_cohort_progress_and_ack(make_scheduler):
    scheduler = make_scheduler(layers_per_slice=2)
    start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    retire(scheduler, scheduler.schedule())
    retire(scheduler, scheduler.schedule(), completed={"p"})
    with open(scheduler.muxpd_policy.config.trace_path) as stream:
        rows = [json.loads(line) for line in stream]
    schedules = [row for row in rows if row["event"] == "decoupled_schedule"]
    completions = [row for row in rows if row["event"] == "decoupled_complete"]
    assert schedules[-1]["pending_prefill_ids"] == ["p"]
    assert schedules[-1]["new_prefill_ids"] == []
    assert completions[-2]["completed_prefill_ids"] == []
    assert completions[-1]["completed_prefill_ids"] == ["p"]
    assert completions[-1]["cohort_id"] == completions[-2]["cohort_id"]


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    return clock


def calibrate(scheduler, clock):
    scheduler.add_request(request("d", 64))
    output = scheduler.schedule()
    clock.now += 0.320
    retire(scheduler, output, completed={"d"})
    output = scheduler.schedule()
    clock.now += 0.220
    retire(scheduler, output)


def test_adaptive_requires_both_pure_phase_samples(make_scheduler, clock):
    scheduler = make_scheduler(adaptive_layers=True)
    scheduler.add_request(request("d", 64))
    output = scheduler.schedule()
    clock.now += 0.320
    retire(scheduler, output, completed={"d"})
    scheduler.add_request(request("p", 64))
    output = scheduler.schedule()
    assert output.muxpd_layers_per_slice == 1
    assert not output.muxpd_finish_prefill
    rows = [json.loads(x) for x in open(scheduler.muxpd_policy.config.trace_path)]
    assert rows[-1]["quota"]["reason"] == "uncalibrated"
    assert rows[-1]["quota"]["latency_samples"] == {"prefill": 1, "decode": 0}
    clock.now += 0.280
    retire(scheduler, output)
    assert scheduler._latency_samples == {"prefill": 1, "decode": 0}


def test_adaptive_deadline_quota_uses_measured_cost_and_retains_decode(
    make_scheduler, clock
):
    scheduler = make_scheduler(adaptive_layers=True)
    calibrate(scheduler, clock)
    scheduler.add_request(request("p", 64))
    output = scheduler.schedule()
    # Safety-adjusted P=384ms, D=264ms: the 1s TTFT budget permits two rounds.
    assert output.muxpd_layers_per_slice == 2
    assert output.num_scheduled_tokens == {"d": 1, "p": 64}
    assert not output.muxpd_finish_prefill
    clock.now += 0.410
    assert output_ids(retire(scheduler, output)) == {"d"}
    assert scheduler._prefill_cohort.completed_layers == 2
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1}
    assert output.muxpd_layers_per_slice == 2
    assert output.muxpd_finish_prefill
    clock.now += 0.410
    assert output_ids(retire(scheduler, output, completed={"p"})) == {"d", "p"}
    assert scheduler._latency_samples == {"prefill": 1, "decode": 1}
    rows = [json.loads(x) for x in open(scheduler.muxpd_policy.config.trace_path)]
    quota = [x["quota"] for x in rows if x["event"] == "decoupled_schedule"][-1]
    assert quota["reason"] == "deadline_finish"
    assert quota["decode_slo_feasible_without_prefill"] is False


def test_adaptive_includes_time_waiting_before_cohort_admission(make_scheduler, clock):
    scheduler = make_scheduler(adaptive_layers=True)
    calibrate(scheduler, clock)
    prefill = request("p", 64)
    prefill.arrival_time -= 2.0
    scheduler.add_request(prefill)
    output = scheduler.schedule()
    assert output.muxpd_layers_per_slice == 4 and output.muxpd_finish_prefill
    assert output.num_scheduled_tokens == {"d": 1, "p": 64}
    clock.now += 0.550
    retire(scheduler, output, completed={"p"})


def test_fixed_quota_remains_fixed_after_calibration(make_scheduler, clock):
    scheduler = make_scheduler()
    calibrate(scheduler, clock)
    prefill = request("p", 64)
    prefill.arrival_time -= 2.0
    scheduler.add_request(prefill)
    output = scheduler.schedule()
    assert output.muxpd_layers_per_slice == 1 and not output.muxpd_finish_prefill
    retire(scheduler, output)


def test_abort_racing_with_pure_completion_cannot_train_latency(make_scheduler, clock):
    scheduler = make_scheduler(adaptive_layers=True)
    scheduler.add_request(request("p", 64))
    output = scheduler.schedule()
    scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    clock.now += 0.500
    retire(scheduler, output, completed={"p"})
    assert scheduler._latency_samples == {"prefill": 0, "decode": 0}
    assert not scheduler._prefill_latency


@pytest.mark.parametrize("fault", ["negative", "bool", "overquota", "total", "missing"])
def test_bad_progress_cannot_retire_decode_or_release_cancelled_kv(
    make_scheduler, fault
):
    scheduler = make_scheduler()
    decode = start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    output = scheduler.schedule()
    scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    held = scheduler._held_prefill_blocks["p"]
    bad = receipt(output)
    if fault == "negative":
        bad.muxpd_prefill_layers_completed = -1
    elif fault == "bool":
        bad.muxpd_prefill_layers_completed = True
    elif fault == "overquota":
        bad.muxpd_prefill_layers_completed = 2
    elif fault == "total":
        bad.muxpd_prefill_layers_total = 5
    else:
        del bad.muxpd_prefill_layers_completed
    with pytest.raises(RuntimeError, match="layer progress receipt"):
        scheduler.update_from_output(output, bad)
    assert decode.num_output_tokens == 1
    assert scheduler._prefill_cohort.completed_layers == 0
    assert all(block.ref_cnt > 0 for block in held)
    retire(scheduler, output)
    assert scheduler._prefill_cohort.completed_layers == 1
    retire(scheduler, scheduler.schedule(), cancelled={"p"})
    assert all(block.ref_cnt == 0 for block in held)


def test_early_completion_and_unacknowledged_final_layer_fail_closed(make_scheduler):
    scheduler = make_scheduler(layers_per_slice=2)
    decode = start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    output = scheduler.schedule()
    with pytest.raises(RuntimeError, match="before all decoder layers"):
        retire(scheduler, output, completed={"p"})
    assert decode.num_output_tokens == 1
    retire(scheduler, output)
    output = scheduler.schedule()
    with pytest.raises(RuntimeError, match="whole cohort retirement"):
        retire(scheduler, output)
    assert scheduler._prefill_cohort.completed_layers == 2
    retire(scheduler, output, completed={"p"})


@pytest.mark.parametrize("log_stats", [False, True])
def test_mfu_metrics_rejected_before_native_initialization(make_scheduler, log_stats):
    config = make_scheduler().vllm_config
    config.observability_config.enable_mfu_metrics = True
    # Reject even when metrics would otherwise be dormant. Omit native allocator
    # arguments deliberately: this configuration error must precede allocation.
    with pytest.raises(ValueError, match="enable_mfu_metrics"):
        DecoupledMuxPDScheduler(config, log_stats=log_stats)
