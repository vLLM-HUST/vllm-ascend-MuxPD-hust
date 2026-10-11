"""Exercise live suspended stacks, context restoration and failure cleanup."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vllm_muxpd.lanes import LayerBoundary, LayerInterleaver


class FakeBackend:
    def __init__(self):
        self.state = "caller"
        self.fences = []

    def capture(self):
        return self.state

    def restore(self, state):
        self.state = state

    def fence(self):
        self.fences.append(self.state)


def test_two_live_forwards_keep_local_state_and_restore_context():
    backend = FakeBackend()
    interleaver = LayerInterleaver(backend)
    visits = []

    def forward(lane):
        local_hidden = 0
        for layer in range(3):
            assert backend.state == lane
            local_hidden = local_hidden * 10 + layer + 1
            visits.append((lane, layer, local_hidden))
            interleaver.checkpoint()
        return local_hidden

    interleaver.add("p", lambda: forward("p"), "p")
    interleaver.add("d", lambda: forward("d"), "d")
    assert interleaver.run(("d", "p")) == {"d": 123, "p": 123}
    assert [(lane, layer) for lane, layer, _ in visits] == [
        ("d", 0),
        ("p", 0),
        ("d", 1),
        ("p", 1),
        ("d", 2),
        ("p", 2),
    ]
    assert backend.state == "caller"
    assert len(backend.fences) == 8


def test_step_preserves_prefill_across_caller_operations():
    backend = FakeBackend()
    interleaver = LayerInterleaver(backend)
    layers = []

    def forward():
        for layer in range(5):
            layers.append(layer)
            interleaver.checkpoint()
        return "complete"

    interleaver.add("p", forward, "prefill-context")
    assert interleaver.step("p", 2) == LayerBoundary("p", 2)
    assert layers == [0, 1]
    assert not interleaver.tasks["p"].done
    backend.state = "different-caller-context"
    assert interleaver.step("p", 2) == LayerBoundary("p", 4)
    assert backend.state == "different-caller-context"
    assert interleaver.step("p", 2) is None
    assert layers == list(range(5))
    assert interleaver.tasks["p"].result == "complete"


def test_failure_unwinds_other_suspended_forward_and_restores_caller():
    backend = FakeBackend()
    interleaver = LayerInterleaver(backend)
    closed = []

    def prefill():
        try:
            interleaver.checkpoint()
            interleaver.checkpoint()
        finally:
            closed.append(backend.state)

    def decode():
        raise RuntimeError("device failure")

    interleaver.add("p", prefill, "prefill")
    interleaver.add("d", decode, "decode")
    with pytest.raises(RuntimeError, match="device failure"):
        interleaver.run(("p", "d"))
    assert closed == ["prefill"]
    assert backend.state == "caller"
    assert all(task.done for task in interleaver.tasks.values())


def test_wrong_thread_and_invalid_budget_are_rejected():
    interleaver = LayerInterleaver(FakeBackend())
    interleaver.add("p", lambda: None, "p")
    with pytest.raises(ValueError, match="positive integer"):
        interleaver.step("p", 0)
    with ThreadPoolExecutor(1) as executor:
        result = executor.submit(interleaver.step, "p")
        with pytest.raises(RuntimeError, match="one host thread"):
            result.result()
    interleaver.close()


def test_no_active_task_leaves_normal_model_hooks_inert():
    backend = FakeBackend()
    interleaver = LayerInterleaver(backend)
    interleaver.checkpoint()
    assert not backend.fences


def test_slice_observer_runs_inside_lane_before_its_existing_completion_fence():
    events = []

    class Backend(FakeBackend):
        def fence(self):
            events.append(("fence", self.state))
            super().fence()

    backend = Backend()
    interleaver = LayerInterleaver(backend)

    class Observer:
        def start(self):
            events.append(("start", backend.state))

        def stop(self):
            events.append(("stop", backend.state))

    def forward():
        for layer in range(3):
            events.append(("layer", layer))
            interleaver.checkpoint(final=layer == 2)

    interleaver.add("p", forward, "prefill")
    interleaver.step("p", 2, observer=Observer())
    assert events == [
        ("start", "prefill"), ("layer", 0), ("layer", 1),
        ("stop", "prefill"), ("fence", "prefill"),
    ]
    assert backend.state == "caller"
    events.clear()
    interleaver.step("p", 2, observer=Observer())
    assert events == [
        ("start", "prefill"), ("layer", 2),
        ("stop", "prefill"), ("fence", "prefill"),
    ]
    assert interleaver.tasks["p"].done
    assert backend.state == "caller"


def test_failed_slice_has_no_success_observation_and_restores_caller():
    backend = FakeBackend()
    interleaver = LayerInterleaver(backend)
    observed = []

    def failure():
        raise RuntimeError("forward failed")

    observer = SimpleNamespace(
        start=lambda: observed.append("start"),
        stop=lambda: observed.append("stop"),
    )
    interleaver.add("p", failure, "prefill")
    with pytest.raises(RuntimeError, match="forward failed"):
        interleaver.step("p", observer=observer)
    assert observed == ["start"]
    assert interleaver._slice_observer is None
    assert backend.state == "caller"


def test_failure_before_initial_barrier_does_not_restore_unstarted_lane():
    class InitialBarrierBackend(FakeBackend):
        def restore(self, state):
            if state == "prefill":
                raise RuntimeError("Prefill cannot start before the initial barrier")
            super().restore(state)

    backend = InitialBarrierBackend()
    interleaver = LayerInterleaver(backend)

    def decode():
        raise RuntimeError("initial preparation failed")

    interleaver.add("p", lambda: None, "prefill")
    interleaver.add("d", decode, "decode")
    with pytest.raises(RuntimeError, match="initial preparation failed"):
        interleaver.run(("d", "p"))
    assert all(task.done for task in interleaver.tasks.values())
    assert backend.state == "caller"


@pytest.fixture
def native_worker():
    import importlib
    import os

    if os.environ.get("VLLM_VERSION") != "0.25.1":
        pytest.skip("source runtime/env.sh to test the pinned HUST native contracts")
    return importlib.import_module("vllm_muxpd.worker")


@pytest.mark.integration
def test_scheduler_slice_preserves_aligned_deltas_and_one_time_side_effects(
    native_worker,
):
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    output = SchedulerOutput.make_empty()
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["d", "p"],
        resumed_req_ids={"p"},
        new_token_ids=[],
        all_token_ids={"p": [3, 4]},
        new_block_ids=[([5],), ([7, 8],)],
        num_computed_tokens=[10, 3],
        num_output_tokens=[2, 0],
    )
    output.num_scheduled_tokens = {"d": 1, "p": 7}
    output.total_num_scheduled_tokens = 8
    output.new_block_ids_to_zero = [7, 8]
    output.finished_req_ids = {"old"}
    decode = native_worker._select_requests(output, {"d"}, carry_side_effects=True)
    prefill = native_worker._select_requests(output, {"p"}, carry_side_effects=False)
    assert decode.num_scheduled_tokens == {"d": 1}
    assert decode.total_num_scheduled_tokens == 1
    assert decode.new_block_ids_to_zero == [7, 8]
    assert prefill.new_block_ids_to_zero is None
    assert prefill.scheduled_cached_reqs.req_ids == ["p"]
    assert prefill.scheduled_cached_reqs.num_computed_tokens == [3]
    assert prefill.scheduled_cached_reqs.new_block_ids == [([7, 8],)]
    assert prefill.scheduled_cached_reqs.resumed_req_ids == {"p"}
    assert prefill.scheduled_cached_reqs.all_token_ids == {"p": [3, 4]}
    assert not decode.finished_req_ids
    assert output.finished_req_ids == {"old"}
    assert output.scheduled_cached_reqs.req_ids == ["d", "p"]


@pytest.mark.integration
def test_sampler_outputs_merge_by_request_without_rebatching_tokens(native_worker):
    from vllm.v1.outputs import ModelRunnerOutput

    decode = ModelRunnerOutput(
        req_ids=["d2", "d1"],
        req_id_to_index={"d2": 0, "d1": 1},
        sampled_token_ids=[[22], [11]],
        pooler_output=[],
    )
    prefill = ModelRunnerOutput(
        req_ids=["p"], req_id_to_index={"p": 0}, sampled_token_ids=[[33]]
    )
    merged = native_worker._merge_outputs([decode, prefill], ["d1", "p", "d2"])
    assert merged.req_ids == ["d1", "p", "d2"]
    assert merged.req_id_to_index == {"d1": 0, "p": 1, "d2": 2}
    assert merged.sampled_token_ids == [[11], [33], [22]]
    with pytest.raises(RuntimeError, match="both MuxPD lanes"):
        native_worker._merge_outputs([decode, decode], ["d1", "d2"])
    with pytest.raises(RuntimeError, match="do not match"):
        native_worker._merge_outputs([decode], ["d1"])


@pytest.mark.integration
def test_second_runner_constructs_buffers_but_reuses_weights_and_cache(
    native_worker, monkeypatch
):
    def initialize_buffers(runner, config, device):
        runner.input_batch = object()
        runner.attn_groups = []
        runner.compilation_config = config.compilation_config

    monkeypatch.setattr(native_worker.NPUModelRunner, "__init__", initialize_buffers)
    attention_cache, recurrent_cache = object(), object()
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "attn": SimpleNamespace(kv_cache=attention_cache),
                "gdn": SimpleNamespace(kv_cache=recurrent_cache),
            }
        )
    )
    owner = SimpleNamespace(
        requests={},
        num_prompt_logprobs={},
        mamba_state_idx={},
        model=object(),
        _has_sinks=False,
        is_multimodal_pruning_enabled=False,
        kv_caches=[attention_cache, recurrent_cache],
        input_batch=object(),
    )
    lane = native_worker._SharedCacheRunner(config, "npu", owner)
    lane.attach_model()
    lane.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["attn", "gdn"])]
    )
    caches = lane.initialize_kv_cache_tensors(None)
    assert lane.model is owner.model
    assert lane.input_batch is not owner.input_batch
    assert lane.requests is owner.requests
    assert lane.mamba_state_idx is owner.mamba_state_idx
    assert lane.kv_caches is owner.kv_caches
    assert caches == {"attn": attention_cache, "gdn": recurrent_cache}


@pytest.mark.integration
def test_preemption_invalidates_idle_lane_rows_without_deleting_request_state(
    native_worker,
):
    registry = {"resume": object(), "running": object()}

    class Batch:
        def __init__(self):
            self.rows = {"resume": "old-block-table", "running": "current"}

        def remove_request(self, req_id):
            self.rows.pop(req_id, None)

    def runner():
        return SimpleNamespace(
            input_batch=Batch(),
            requests=registry,
            _hybrid_accepted_by_req={"resume": 1, "running": 1},
        )

    worker = native_worker.MuxPDNPUWorker.__new__(native_worker.MuxPDNPUWorker)
    worker.model_runner = runner()
    worker.decode_runner = runner()
    worker._invalidate_batch_rows({"resume"})
    for lane in (worker.model_runner, worker.decode_runner):
        assert lane.input_batch.rows == {"running": "current"}
        assert lane._hybrid_accepted_by_req == {"running": 1}
        assert lane.requests is registry
    assert set(registry) == {"resume", "running"}
