"""Check observational hooks preserve native call/return and ordering."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_muxpd.numerical_diagnostics import (
    NativeSamplerObserver,
    bounded_constant,
)


class Tensor:
    def __init__(self, values, events):
        self.values = np.asarray(values, dtype=np.float32)
        self.events = events
        self.shape = self.values.shape
        self.dtype = "float32"

    def detach(self):
        return self

    def clone(self):
        self.events.append("clone-after-native-greedy")
        return Tensor(self.values.copy(), self.events)

    def float(self):
        return self

    def cpu(self):
        self.events.append("diagnostic-cpu")
        return self

    def numpy(self):
        return self.values


def fixture_runner(events):
    request = SimpleNamespace(
        prompt_token_ids=[10, 20], output_token_ids=[30], block_ids=([4, 5],)
    )
    batch = SimpleNamespace(
        req_ids=["r"],
        num_tokens_no_spec=[3],
        num_computed_tokens_cpu=[2],
        token_ids_cpu=np.array([[10, 20, 30]]),
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )
    original_result = object()

    def greedy(logits):
        events.append("native-greedy")
        return original_result

    sampler = SimpleNamespace(greedy_sample=greedy)

    def sample(logits, spec):
        events.append("native-sample-enter")
        assert spec is None
        result = sampler.greedy_sample(logits)
        events.append("native-sample-return")
        return result

    native_bookkeeping = (None, [[2]], {}, ["r"], {"r": 0}, [])

    def bookkeeping(*args, **kwargs):
        events.append("native-bookkeeping")
        return native_bookkeeping

    runner = SimpleNamespace(
        input_batch=batch,
        requests={"r": request},
        sampler=sampler,
        _sample=sample,
        _bookkeeping_sync=bookkeeping,
        execute_model=lambda output: original_result,
        use_async_scheduling=False,
        speculative_config=None,
    )
    return runner, original_result, native_bookkeeping


def schedule():
    return SimpleNamespace(
        num_scheduled_tokens={"r": 1},
        total_num_scheduled_tokens=1,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r"], num_computed_tokens=[2], num_output_tokens=[1]
        ),
        finished_req_ids=set(),
        preempted_req_ids=None,
    )


def test_native_decision_and_bookkeeping_are_unchanged(tmp_path):
    events = []
    runner, expected, native = fixture_runner(events)
    original_logits = Tensor([[0.2, 0.4, 0.8, 0.1]], events)
    before = original_logits.values.copy()
    observer = NativeSamplerObserver(runner, tmp_path, "native", 0)
    assert runner.execute_model(schedule()) is expected
    assert runner._sample(original_logits, None) is expected
    assert runner._bookkeeping_sync() is native
    assert events == [
        "native-sample-enter",
        "native-greedy",
        "clone-after-native-greedy",
        "native-sample-return",
        "native-bookkeeping",
        "diagnostic-cpu",
    ]
    np.testing.assert_array_equal(original_logits.values, before)
    row = json.loads((tmp_path / "rank-0-native.jsonl").read_text().splitlines()[-1])
    assert row["native_greedy_calls"] == 1
    sample = row["rows"][0]
    assert sample["output_prefix_token_ids"] == [30]
    assert sample["sampled_token_ids"] == [2]
    assert sample["logits"]["cpu_argmax"] == 2
    np.testing.assert_array_equal(
        np.load(tmp_path / sample["logits"]["file"]), original_logits.values[0]
    )
    assert observer.pending is None


def test_unselected_rank_does_not_clone_or_copy(tmp_path):
    events = []
    runner, expected, native = fixture_runner(events)
    NativeSamplerObserver(runner, tmp_path, "native", 1)
    assert runner._sample(Tensor([[0.1, 0.2, 0.9]], events), None) is expected
    assert runner._bookkeeping_sync() is native
    assert "diagnostic-cpu" not in events
    assert "clone-after-native-greedy" not in events
    assert not list(tmp_path.glob("*.npy"))


def test_padded_graph_descriptor_is_taken_from_native_deferred_state(tmp_path):
    runner, expected, _ = fixture_runner([])
    runner.execute_model_state = (None,) * 11 + (
        SimpleNamespace(
            num_tokens=4,
            num_reqs=4,
            uniform=True,
            has_lora=False,
            num_active_loras=0,
        ),
    )
    NativeSamplerObserver(runner, tmp_path, "native", 0)
    assert runner.execute_model(schedule()) is expected
    row = json.loads((tmp_path / "rank-0-native.jsonl").read_text().splitlines()[-1])
    assert row["event"] == "execute-return"
    assert row["batch_descriptor"] == {
        "num_tokens": 4,
        "num_reqs": 4,
        "uniform": True,
        "has_lora": False,
        "num_active_loras": 0,
    }


def test_reject_async_and_non_greedy_paths(tmp_path):
    runner, _, _ = fixture_runner([])
    runner.use_async_scheduling = True
    with pytest.raises(ValueError, match="synchronous"):
        NativeSamplerObserver(runner, tmp_path, "native", 0)
    runner.use_async_scheduling = False
    NativeSamplerObserver(runner, tmp_path, "native", 0)
    runner.input_batch.sampling_metadata.all_greedy = False
    with pytest.raises(ValueError, match="plain greedy"):
        runner._sample(Tensor([[0.1, 0.2]], []), None)


def test_duplicate_observer_and_output_refused(tmp_path):
    runner, _, _ = fixture_runner([])
    NativeSamplerObserver(runner, tmp_path, "native", 0)
    with pytest.raises(ValueError, match="already"):
        NativeSamplerObserver(runner, tmp_path, "native", 0)
    other, _, _ = fixture_runner([])
    with pytest.raises(FileExistsError):
        NativeSamplerObserver(other, tmp_path, "native", 0)


def test_native_greedy_failure_is_not_replaced(tmp_path):
    events = []
    runner, _, _ = fixture_runner(events)

    def fail(logits):
        events.append("native-failure")
        raise RuntimeError("native error")

    runner.sampler.greedy_sample = fail
    observer = NativeSamplerObserver(runner, tmp_path, "native", 0)
    with pytest.raises(RuntimeError, match="native error"):
        runner._sample(Tensor([[0.1, 0.2]], events), None)
    assert events == ["native-sample-enter", "native-failure"]
    assert observer.pending is None
    assert not list(tmp_path.glob("*.npy"))


def test_compiler_constant_metadata_never_reads_tensor_values():
    tensor = SimpleNamespace(shape=(2, 4), dtype="bfloat16", device="npu:0")
    result = bounded_constant({1: {"BLOCK": 256, "tensor": tensor}})
    assert result["1"]["BLOCK"] == 256
    assert result["1"]["tensor"]["shape"] == [2, 4]
    assert result["1"]["tensor"]["contents_observed"] is False
