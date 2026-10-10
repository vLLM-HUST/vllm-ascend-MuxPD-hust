"""Exercise cross-iteration prefill ownership with the native vLLM allocator."""

import json
import time
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

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
        *,
        num_blocks=64,
        max_num_seqs=8,
        layers_per_slice=1,
        adaptive_layers=False,
        prefill_first=False,
        compile_full_prefill=False,
        prefill_layer_token_budget=0,
        trace=True,
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
                    "prefill_first": prefill_first,
                    "decode_graph": compile_full_prefill,
                    "compile_full_prefill": compile_full_prefill,
                    "prefill_layer_token_budget": prefill_layer_token_budget,
                    "max_decode_steps": 1,
                    "tpot_slo_ms": 0.001,
                    "trace_path": str(tmp_path / "trace.jsonl") if trace else None,
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


def receipt(output, *, completed=(), cancelled=(), source=None):
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
    result.muxpd_prefill_progress_source = (
        source if source is not None else "layer_hooks" if pending else "none"
    )
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


def test_trace_off_preserves_scheduling_receipts_and_allocator_without_records(
    make_scheduler, monkeypatch, clock
):
    import vllm_muxpd.scheduler as scheduler_module

    def exercise(scheduler):
        clock.now = 1000.0
        observed = []

        def step(*, completed=(), cancelled=()):
            clock.now += 0.01
            output = scheduler.schedule()
            controls = (
                dict(output.num_scheduled_tokens),
                output.muxpd_prefill_request_ids,
                output.muxpd_pending_prefill_ids,
                output.muxpd_cancel_prefill_request_ids,
                output.muxpd_prefill_cohort_id,
                output.muxpd_layers_per_slice,
                output.muxpd_finish_prefill,
                output.muxpd_prefill_layers_completed_before,
                output.muxpd_prefill_layers_total,
            )
            clock.now += 0.01
            result = retire(scheduler, output, completed=completed, cancelled=cancelled)
            observed.append(
                (
                    controls,
                    output_ids(result),
                    {
                        rid: (request.num_computed_tokens, request.num_output_tokens)
                        for rid, request in scheduler.requests.items()
                    },
                    scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                    set(scheduler._held_prefill_blocks),
                )
            )

        scheduler.add_request(request("d"))
        step(completed={"d"})
        scheduler.add_request(request("p", 32))
        step()
        step()
        scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
        step(cancelled={"p"})
        scheduler.add_request(request("survivor", 32))
        for index in range(4):
            step(completed={"survivor"} if index == 3 else ())
        step()
        return observed, scheduler.export_muxpd_stats()

    traced = make_scheduler(trace=True)
    expected = exercise(traced)
    untraced = make_scheduler(trace=False)

    def unexpected_record(*args, **kwargs):
        pytest.fail("trace-off scheduling must not construct detailed trace records")

    monkeypatch.setattr(scheduler_module, "asdict", unexpected_record)
    monkeypatch.setattr(untraced, "_trace_decoupled", unexpected_record)
    assert exercise(untraced) == expected
    assert traced.last_muxpd_step is not None
    assert untraced.last_muxpd_step is None
    assert untraced._muxpd_trace is None


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


def urgent_request(rid, tokens=64):
    prefill = request(rid, tokens)
    prefill.arrival_time -= 2.0
    return prefill


@pytest.mark.parametrize("later_tokens", [64, 255])
def test_prefill_first_defers_native_decode_state_then_repays_before_new_prefill(
    make_scheduler, clock, later_tokens
):
    scheduler = make_scheduler(adaptive_layers=True, prefill_first=True)
    calibrate(scheduler, clock)
    decode = scheduler.requests["d"]
    before_computed = decode.num_computed_tokens
    before_outputs = decode.num_output_tokens
    scheduler.add_request(urgent_request("p"))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"p": 64}
    assert output.muxpd_finish_prefill
    assert decode.num_computed_tokens == before_computed
    assert decode.num_output_tokens == before_outputs
    assert not scheduler._prefill_first_decode_debt  # Commit only a valid receipt.
    clock.now += 0.320
    assert output_ids(retire(scheduler, output, completed={"p"})) == {"p"}
    assert decode.num_computed_tokens == before_computed
    assert decode.num_output_tokens == before_outputs
    assert scheduler._prefill_first_decode_debt == {"d"}
    scheduler.add_request(urgent_request("later", later_tokens))
    output = scheduler.schedule()
    # Debt overrides even a full-budget prompt that the base selector would
    # otherwise put in its own P-only transaction.
    assert output.num_scheduled_tokens == {"d": 1, "p": 1}
    assert not output.muxpd_pending_prefill_ids
    assert scheduler.requests["later"].num_computed_tokens == 0
    clock.now += 0.220
    assert output_ids(retire(scheduler, output)) == {"d", "p"}
    assert decode.num_computed_tokens == before_computed + 1
    assert decode.num_output_tokens == before_outputs + 1
    assert not scheduler._prefill_first_decode_debt
    with open(scheduler.muxpd_policy.config.trace_path) as stream:
        rows = [json.loads(line) for line in stream]
    schedules = [row for row in rows if row["event"] == "decoupled_schedule"]
    assert schedules[-2]["admission"]["reason"] == "deadline_prefill_first"
    assert schedules[-1]["admission"]["reason"] == "repay_decode"


def test_prefill_first_keeps_nonurgent_cross_iteration_prefill(make_scheduler, clock):
    scheduler = make_scheduler(adaptive_layers=True, prefill_first=True)
    calibrate(scheduler, clock)
    scheduler.add_request(request("p", 64))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1, "p": 64}
    assert output.muxpd_layers_per_slice == 2 and not output.muxpd_finish_prefill
    clock.now += 0.410
    retire(scheduler, output)
    scheduler.add_request(urgent_request("later"))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1}
    assert output.muxpd_finish_prefill
    clock.now += 0.410
    retire(scheduler, output, completed={"p"})
    assert not scheduler._prefill_first_decode_debt


def test_prefill_first_uncalibrated_keeps_existing_selection(make_scheduler, clock):
    scheduler = make_scheduler(adaptive_layers=True, prefill_first=True)
    scheduler.add_request(request("d", 64))
    output = scheduler.schedule()
    clock.now += 0.320
    retire(scheduler, output, completed={"d"})
    scheduler.add_request(urgent_request("p"))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1, "p": 64}
    assert output.muxpd_layers_per_slice == 1 and not output.muxpd_finish_prefill
    retire(scheduler, output)
    assert not scheduler._prefill_first_decode_debt


def test_prefill_first_kv_failure_falls_back_without_advancing_prefill(
    make_scheduler, clock
):
    scheduler = make_scheduler(num_blocks=8, adaptive_layers=True, prefill_first=True)
    calibrate(scheduler, clock)
    decode = scheduler.requests["d"]
    before_computed = decode.num_computed_tokens
    prefill = urgent_request("p")
    scheduler.add_request(prefill)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1}
    assert decode.num_computed_tokens == before_computed + 1
    assert prefill.num_computed_tokens == 0
    assert scheduler._prefill_cohort is None
    assert scheduler._prefill_first_admission["reason"] == "prefill_kv_fallback"
    assert scheduler.muxpd_stats["prefill_capacity_deferrals"] == 1
    retire(scheduler, output)
    assert not scheduler._prefill_first_decode_debt


@pytest.mark.parametrize("before_receipt", [False, True])
def test_cancelling_deferred_decode_clears_only_its_debt(
    make_scheduler, clock, before_receipt
):
    scheduler = make_scheduler(adaptive_layers=True, prefill_first=True)
    calibrate(scheduler, clock)
    scheduler.add_request(urgent_request("p"))
    output = scheduler.schedule()
    if before_receipt:
        scheduler.finish_requests("d", RequestStatus.FINISHED_ABORTED)
    retire(scheduler, output, completed={"p"})
    if not before_receipt:
        assert scheduler._prefill_first_decode_debt == {"d"}
        scheduler.finish_requests("d", RequestStatus.FINISHED_ABORTED)
    assert not scheduler._prefill_first_decode_debt
    scheduler.add_request(urgent_request("later"))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"later": 64}
    retire(scheduler, output, completed={"later"})
    assert scheduler._prefill_first_decode_debt == {"p"}


def test_prefill_first_invalid_receipt_cannot_commit_decode_debt(make_scheduler, clock):
    scheduler = make_scheduler(adaptive_layers=True, prefill_first=True)
    calibrate(scheduler, clock)
    scheduler.add_request(urgent_request("p"))
    output = scheduler.schedule()
    invalid = receipt(output, completed={"p"})
    invalid.muxpd_prefill_layers_total += 1
    with pytest.raises(RuntimeError, match="layer progress receipt"):
        scheduler.update_from_output(output, invalid)
    assert not scheduler._prefill_first_decode_debt
    retire(scheduler, output, completed={"p"})
    assert scheduler._prefill_first_decode_debt == {"d"}


def test_cancelling_one_deferred_decode_preserves_other_debt(make_scheduler, clock):
    scheduler = make_scheduler(adaptive_layers=True, prefill_first=True)
    calibrate(scheduler, clock)
    scheduler.add_request(urgent_request("d2", 8))
    retire(scheduler, scheduler.schedule(), completed={"d2"})
    retire(scheduler, scheduler.schedule())
    scheduler.add_request(urgent_request("p"))
    retire(scheduler, scheduler.schedule(), completed={"p"})
    assert scheduler._prefill_first_decode_debt == {"d", "d2"}
    scheduler.finish_requests("d", RequestStatus.FINISHED_ABORTED)
    assert scheduler._prefill_first_decode_debt == {"d2"}
    scheduler.add_request(urgent_request("later"))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d2": 1, "p": 1}
    assert scheduler.requests["later"].num_computed_tokens == 0
    retire(scheduler, output)
    assert not scheduler._prefill_first_decode_debt


def test_decode_repayment_keeps_native_kv_preemption(make_scheduler, clock):
    scheduler = make_scheduler(num_blocks=10, adaptive_layers=True, prefill_first=True)
    calibrate(scheduler, clock)
    # Put D at a block boundary. P can use the remaining four blocks, but the
    # mandatory D turn needs another block and must preempt the newly finished P.
    for _ in range(14):
        retire(scheduler, scheduler.schedule())
    assert scheduler.requests["d"].num_computed_tokens == 79
    scheduler.add_request(urgent_request("p"))
    retire(scheduler, scheduler.schedule(), completed={"p"})
    assert scheduler._prefill_first_decode_debt == {"d"}
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1}
    assert output.preempted_req_ids == {"p"}
    retire(scheduler, output)
    assert scheduler.requests["p"].status == RequestStatus.PREEMPTED
    assert not scheduler._prefill_first_decode_debt


def test_full_budget_single_token_prefills_cannot_starve_decode(make_scheduler, clock):
    scheduler = make_scheduler(adaptive_layers=True, prefill_first=True)
    calibrate(scheduler, clock)
    scheduler.add_request(urgent_request("d2", 8))
    retire(scheduler, scheduler.schedule(), completed={"d2"})
    retire(scheduler, scheduler.schedule())
    for index in range(4):
        scheduler.add_request(request(f"long-{index}", 255, max_tokens=1))
    for index in range(4):
        before = {
            rid: scheduler.requests[rid].num_computed_tokens for rid in ("d", "d2")
        }
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {f"long-{index}": 255}
        assert (
            scheduler._prefill_first_admission["reason"] == "full_budget_prefill_first"
        )
        assert all(
            scheduler.requests[rid].num_computed_tokens == n
            for rid, n in before.items()
        )
        # All queued P requests now exceed max_prefill_wait_ms. Without debt,
        # the base P-only fallback could repeatedly skip D as each slot frees.
        clock.now += 0.320
        retire(scheduler, output, completed={f"long-{index}"})
        assert f"long-{index}" not in scheduler.requests
        assert scheduler._prefill_first_decode_debt == {"d", "d2"}
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {"d": 1, "d2": 1}
        assert scheduler._prefill_first_admission["reason"] == "repay_decode"
        clock.now += 0.220
        retire(scheduler, output)
        assert all(
            scheduler.requests[rid].num_computed_tokens == n + 1
            for rid, n in before.items()
        )
        assert not scheduler._prefill_first_decode_debt


def test_full_budget_prefill_kv_fallback_does_not_create_debt(make_scheduler, clock):
    scheduler = make_scheduler(num_blocks=20, adaptive_layers=True, prefill_first=True)
    calibrate(scheduler, clock)
    scheduler.add_request(urgent_request("d2", 8))
    retire(scheduler, scheduler.schedule(), completed={"d2"})
    retire(scheduler, scheduler.schedule())
    scheduler.add_request(request("long", 255, max_tokens=1))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1, "d2": 1}
    assert scheduler._prefill_first_admission["reason"] == "prefill_kv_fallback"
    assert scheduler.requests["long"].num_computed_tokens == 0
    retire(scheduler, output)
    assert not scheduler._prefill_first_decode_debt


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


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("abort_after_execution", [False, True])
def test_fresh_full_forward_receipt_retires_only_after_return(
    make_scheduler, mixed, abort_after_execution
):
    scheduler = make_scheduler(layers_per_slice=4, compile_full_prefill=True)
    if mixed:
        start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    output = scheduler.schedule()
    result = receipt(output, completed={"p"}, source="full_forward_return")
    if abort_after_execution:
        scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
        held = scheduler._held_prefill_blocks["p"]
    updated = scheduler.update_from_output(output, result)
    assert output_ids(updated) == (
        ({"d"} if mixed else set()) | (set() if abort_after_execution else {"p"})
    )
    assert scheduler._prefill_cohort is None
    if abort_after_execution:
        assert all(block.ref_cnt == 0 for block in held)
    with open(scheduler.muxpd_policy.config.trace_path) as stream:
        rows = [json.loads(line) for line in stream]
    assert rows[-2]["compile_full_prefill"] is True
    assert rows[-1]["progress_source"] == "full_forward_return"
    assert rows[-1]["layers_completed"] == rows[-1]["layers_total"] == 4


@pytest.mark.parametrize("source", ["missing", "unknown", "none", 4, ["layer_hooks"]])
def test_invalid_progress_source_cannot_retire_decode_or_unpin_kv(
    make_scheduler, source
):
    scheduler = make_scheduler(layers_per_slice=4)
    decode = start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    output = scheduler.schedule()
    scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    held = scheduler._held_prefill_blocks["p"]
    invalid = receipt(output, completed={"p"}, source=source)
    if source == "missing":
        del invalid.muxpd_prefill_progress_source
    with pytest.raises(RuntimeError, match="progress source"):
        scheduler.update_from_output(output, invalid)
    assert decode.num_output_tokens == 1
    assert scheduler._prefill_cohort.completed_layers == 0
    assert all(block.ref_cnt > 0 for block in held)
    retire(scheduler, output, completed={"p"})
    assert all(block.ref_cnt == 0 for block in held)


def test_full_forward_progress_requires_explicit_opt_in(make_scheduler):
    scheduler = make_scheduler()
    scheduler.add_request(request("p", 32))
    output = scheduler.schedule()
    with pytest.raises(RuntimeError, match="fresh, complete"):
        retire(scheduler, output, completed={"p"}, source="full_forward_return")
    assert scheduler.requests["p"].num_output_tokens == 0
    retire(scheduler, output, completed={"p"})


def test_partial_and_resumed_prefill_cannot_claim_full_forward_return(make_scheduler):
    scheduler = make_scheduler(layers_per_slice=2, compile_full_prefill=True)
    decode = start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    output = scheduler.schedule()
    with pytest.raises(RuntimeError, match="fresh, complete"):
        retire(scheduler, output, source="full_forward_return")
    assert decode.num_output_tokens == 1
    retire(scheduler, output)
    output = scheduler.schedule()
    assert not output.muxpd_prefill_request_ids
    with pytest.raises(RuntimeError, match="fresh, complete"):
        retire(scheduler, output, completed={"p"}, source="full_forward_return")
    assert scheduler._prefill_cohort.completed_layers == 2
    assert decode.num_output_tokens == 2
    retire(scheduler, output, completed={"p"})


def test_full_forward_source_cannot_ack_paused_or_cancelled_prefill(make_scheduler):
    scheduler = make_scheduler(compile_full_prefill=True)
    start_decode(scheduler)
    scheduler.add_request(request("p", 32))
    retire(scheduler, scheduler.schedule())
    # Public PAUSED_ALL is rejected while P is pending. Exercise the defensive
    # protocol for a native caller that wrote the underlying state directly.
    scheduler._pause_state = PauseState.PAUSED_ALL
    output = scheduler.schedule()
    with pytest.raises(RuntimeError, match="fresh, complete"):
        retire(scheduler, output, source="full_forward_return")
    retire(scheduler, output)
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    held = scheduler._held_prefill_blocks["p"]
    output = scheduler.schedule()
    with pytest.raises(RuntimeError, match="fresh, complete"):
        retire(scheduler, output, cancelled={"p"}, source="full_forward_return")
    assert all(block.ref_cnt > 0 for block in held)
    retire(scheduler, output, cancelled={"p"})
    assert all(block.ref_cnt == 0 for block in held)


@pytest.mark.parametrize("source", ["layer_hooks", "full_forward_return"])
def test_no_cohort_requires_none_progress_source(make_scheduler, source):
    scheduler = make_scheduler(compile_full_prefill=True)
    decode = start_decode(scheduler)
    output = scheduler.schedule()
    with pytest.raises(RuntimeError, match="cohort ownership"):
        retire(scheduler, output, source=source)
    assert decode.num_output_tokens == 1
    retire(scheduler, output)


@pytest.mark.parametrize(
    "tokens,budget,floor,quota",
    [
        (32, 128, 1, 4),
        (64, 128, 1, 2),
        (80, 128, 1, 1),
        (64, 1, 2, 2),
        (64, 0, 1, 1),
        (32, 128, 8, 4),
    ],
)
def test_layer_token_budget_uses_floor_division_and_keeps_complete_prompts(
    make_scheduler, tokens, budget, floor, quota
):
    scheduler = make_scheduler(
        layers_per_slice=floor,
        prefill_layer_token_budget=budget,
    )
    decode = start_decode(scheduler)
    scheduler.add_request(request("p", tokens))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1, "p": tokens}
    assert output.muxpd_layers_per_slice == quota
    assert output.muxpd_prefill_layers_total == 4
    retire(scheduler, output, completed={"p"} if quota == 4 else ())
    assert decode.num_output_tokens == 2


def test_work_quota_keeps_original_cohort_cost_after_partial_cancellation(
    make_scheduler,
):
    scheduler = make_scheduler(prefill_layer_token_budget=64)
    start_decode(scheduler)
    scheduler.add_request(request("p1", 32))
    scheduler.add_request(request("p2", 32))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1, "p1": 32, "p2": 32}
    assert output.muxpd_layers_per_slice == 1
    retire(scheduler, output)
    scheduler.finish_requests("p2", RequestStatus.FINISHED_ABORTED)
    held = scheduler._held_prefill_blocks["p2"]
    for completed in (False, False, True):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {"d": 1}
        assert output.muxpd_layers_per_slice == 1
        retire(
            scheduler,
            output,
            completed={"p1"} if completed else (),
            cancelled={"p2"} if completed else (),
        )
    assert all(block.ref_cnt == 0 for block in held)
    with open(scheduler.muxpd_policy.config.trace_path) as stream:
        rows = [json.loads(line) for line in stream]
    work = [
        r["quota"]["work_quota"]
        for r in rows
        if r["event"] == "decoupled_schedule" and r["cohort_id"] == 2
    ]
    assert len(work) == 4
    assert all(w["cohort_full_prefill_tokens"] == 64 for w in work)
    assert all(w["base_layers_per_slice"] == 1 for w in work)


def test_work_quota_recomputed_prefill_includes_existing_output_tokens(make_scheduler):
    scheduler = make_scheduler(prefill_layer_token_budget=32)
    start_decode(scheduler)
    prefill = request("recomputed", 8)
    prefill.append_output_token_ids([2, 3])
    scheduler.add_request(prefill)
    output = scheduler.schedule()
    assert output.muxpd_prefill_request_ids == {"recomputed"}
    assert output.num_scheduled_tokens == {"d": 1, "recomputed": 10}
    # Using only the original 8 prompt tokens would incorrectly select 4 layers.
    assert output.muxpd_layers_per_slice == 3
    retire(scheduler, output)
    output = scheduler.schedule()
    assert output.muxpd_layers_per_slice == 1 and output.muxpd_finish_prefill
    retire(scheduler, output, completed={"recomputed"})


def test_deadline_can_raise_but_not_reduce_the_work_quota(make_scheduler, clock):
    scheduler = make_scheduler(adaptive_layers=True, prefill_layer_token_budget=192)
    calibrate(scheduler, clock)
    scheduler.add_request(request("p", 64))
    output = scheduler.schedule()
    # Deadline permits two rounds, which alone requested 2 layers. Work budget
    # permits 3, so the base remains 3; no full prompt is split into token chunks.
    assert output.muxpd_layers_per_slice == 3
    assert output.num_scheduled_tokens == {"d": 1, "p": 64}
    retire(scheduler, output)
    output = scheduler.schedule()
    assert output.muxpd_layers_per_slice == 1 and output.muxpd_finish_prefill
    retire(scheduler, output, completed={"p"})


def test_work_quota_prefill_first_works_before_latency_calibration_and_repays_decode(
    make_scheduler,
):
    scheduler = make_scheduler(
        adaptive_layers=True,
        prefill_first=True,
        compile_full_prefill=True,
        prefill_layer_token_budget=128,
    )
    decode = start_decode(scheduler)
    assert scheduler._decode_latency_ms is None
    before = decode.num_computed_tokens
    scheduler.add_request(request("p", 32, max_tokens=1))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"p": 32}
    assert decode.num_computed_tokens == before
    assert output.muxpd_finish_prefill and output.muxpd_layers_per_slice == 4
    assert scheduler._prefill_first_admission["reason"] == "token_budget_prefill_first"
    preview = scheduler._prefill_first_admission["preview"]["work_quota"]
    assert preview["cohort_full_prefill_tokens"] == 32
    assert preview["base_layers_per_slice"] == 4
    retire(scheduler, output, completed={"p"}, source="full_forward_return")
    assert scheduler._prefill_first_decode_debt == {"d"}
    scheduler.add_request(request("next", 32, max_tokens=1))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1}
    assert scheduler.requests["next"].num_computed_tokens == 0
    retire(scheduler, output)
    assert decode.num_computed_tokens == before + 1
    assert not scheduler._prefill_first_decode_debt


def test_work_quota_prefill_first_preserves_kv_failure_decode_fallback(make_scheduler):
    scheduler = make_scheduler(
        num_blocks=8,
        adaptive_layers=True,
        prefill_first=True,
        prefill_layer_token_budget=512,
    )
    decode = start_decode(scheduler)
    before = decode.num_computed_tokens
    scheduler.add_request(request("p", 128))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"d": 1}
    assert scheduler._prefill_first_admission["reason"] == "prefill_kv_fallback"
    assert scheduler.requests["p"].num_computed_tokens == 0
    retire(scheduler, output)
    assert decode.num_computed_tokens == before + 1
    assert not scheduler._prefill_first_decode_debt


def test_work_quota_recomputes_actual_tokens_after_partial_kv_admission(make_scheduler):
    scheduler = make_scheduler(
        num_blocks=6,
        adaptive_layers=True,
        prefill_first=True,
        prefill_layer_token_budget=128,
    )
    start_decode(scheduler)
    scheduler.add_request(request("fits", 32))
    scheduler.add_request(request("no_kv", 64))
    output = scheduler.schedule()
    preview = scheduler._prefill_first_admission["preview"]["work_quota"]
    assert preview["cohort_full_prefill_tokens"] == 96
    assert preview["base_layers_per_slice"] == 1
    # The native KV allocator admitted only the first complete P. Its smaller
    # actual cohort can finish in one call; D was already admitted and remains.
    assert output.num_scheduled_tokens == {"d": 1, "fits": 32}
    assert scheduler.requests["no_kv"].num_computed_tokens == 0
    assert output.muxpd_layers_per_slice == 4 and output.muxpd_finish_prefill
    assert scheduler._prefill_cohort.original_output.total_num_scheduled_tokens == 32
    with open(scheduler.muxpd_policy.config.trace_path) as stream:
        rows = [json.loads(line) for line in stream]
    work = rows[-1]["quota"]["work_quota"]
    assert work["cohort_full_prefill_tokens"] == 32
    assert work["base_layers_per_slice"] == 4
    retire(scheduler, output, completed={"fits"})
    assert not scheduler._prefill_first_decode_debt
