"""CPU-only contracts for early submission and unchanged native retirement."""

import sys
from types import SimpleNamespace as NS

import pytest

from vllm_muxpd.config import MuxPDConfig
from vllm_muxpd.decode_sampling import DecodeSamplePipeline, NPUSampleCopyRuntime


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))

    def __getitem__(self, _key):
        return self


class Runtime:
    def __init__(self, events):
        self.events = events
        self.fail_copy = False
        self.fail_sync = False

    def enqueue(self, sampled):
        self.events.extend(["copy_enqueued", "copy_event_recorded"])
        if self.fail_copy:
            raise RuntimeError("copy failed after submission")
        return NS(source=sampled)

    def materialize(self, view):
        self.events.append("copy_event_wait")
        return [list(row) for row in view.source.values]

    def synchronize(self):
        self.events.append("decode_stream_sync")
        if self.fail_sync:
            raise RuntimeError("stream join failed")


class Runner:
    def __init__(self, events):
        self.events = events
        self.use_async_scheduling = False
        self.speculative_config = None
        self.routed_experts_initialized = False
        self.kv_connector_output = None
        self.model_config = NS(logits_processors=[])
        self.num_discarded_requests = 0
        self.sample_calls = 0
        self.native_retire_calls = 0
        self.requests = {
            "d": NS(
                output_token_ids=[11],
                sampling_params=NS(
                    structured_outputs=None, logprobs=None, prompt_logprobs=None
                ),
            )
        }
        self.input_batch = NS(
            req_ids=["d"], req_id_to_index={"d": 0}, sampled_token_ids_cpu=None,
            sampling_metadata=NS(
                all_greedy=True, all_random=False, no_penalties=True,
                max_num_logprobs=None, logprob_token_ids=None,
                bad_words_token_ids={}, allowed_token_ids_mask=None,
                thinking_budget_state_holder=None, logitsprocs=NS(all=[]),
            ),
        )
        self.execute_model_state = NS(
            logits=Tensor([[0, 1]]), spec_decode_metadata=None,
            scheduler_output=NS(num_scheduled_tokens={"d": 1}),
        )

    def _sample(self, logits, spec_decode_metadata):
        assert spec_decode_metadata is None
        self.events.append("native_sample")
        self.sample_calls += 1
        return NS(sampled_token_ids=Tensor([[7]]), logprobs_tensors=None)

    def _to_list(self, sampled):
        self.events.append("native_copy_and_wait")
        return [list(row) for row in sampled.values]

    def sample_tokens(self, grammar_output):
        assert grammar_output is None
        self.native_retire_calls += 1
        state = self.execute_model_state
        self.execute_model_state = None
        output = self._sample(state.logits, state.spec_decode_metadata)
        values = self._to_list(output.sampled_token_ids)
        self.requests["d"].output_token_ids.extend(values[0])
        self.events.append("native_history_and_feedback")
        return values


@pytest.fixture
def setup():
    events = []
    runner = Runner(events)
    runtime = Runtime(events)
    scope = NS(lane="decode")

    def check_scope():
        if scope.lane != "decode":
            raise RuntimeError("not decode scope")

    pipe = DecodeSamplePipeline(
        runner, runtime, check_scope, builtin_processor_types=()
    )
    return NS(events=events, runner=runner, runtime=runtime, pipe=pipe, scope=scope)


def test_option_default_off_and_supported_combination():
    assert not MuxPDConfig().decode_sample_pipeline
    config = MuxPDConfig(
        execution="decoupled", decode_graph=True, stream_mode="overlap",
        overlap_communication="isolated", decode_sample_pipeline=True,
    )
    assert config.decode_sample_pipeline


@pytest.mark.parametrize("override", [
    {"decode_sample_pipeline": 1}, {"decode_graph": False},
    {"stream_mode": "fenced", "overlap_communication": "shared"},
])
def test_option_rejects_unsupported_combinations(override):
    values = dict(
        execution="decoupled", decode_graph=True, stream_mode="overlap",
        overlap_communication="isolated", decode_sample_pipeline=True,
    ) | override
    with pytest.raises(ValueError):
        MuxPDConfig(**values)


def test_early_copy_does_not_retire_native_state_or_tokens(setup):
    s = setup
    state = s.runner.execute_model_state
    assert s.pipe.enqueue()
    assert s.runner.execute_model_state is state
    assert s.runner.requests["d"].output_token_ids == [11]
    assert s.runner.native_retire_calls == 0
    assert s.events == ["native_sample", "copy_enqueued", "copy_event_recorded"]
    s.events.append("prefill_slice")
    assert s.runner.sample_tokens(None) == [[7]]
    assert s.runner.sample_calls == s.runner.native_retire_calls == 1
    assert s.runner.requests["d"].output_token_ids == [11, 7]
    assert s.runner.execute_model_state is None and not s.pipe.pending
    assert s.events[-3:] == [
        "prefill_slice", "copy_event_wait", "native_history_and_feedback"
    ]
    assert s.pipe.stats["enqueued"] == s.pipe.stats["retired"] == 1


@pytest.mark.parametrize("change", [
    "random", "penalties", "bad_words", "allowed_mask", "thinking",
    "logprobs", "prompt_logprobs", "grammar", "custom", "discard",
    "async_feedback", "routed", "async", "spec", "multiple_tokens",
])
def test_unsupported_batch_falls_back_before_sampling(setup, change):
    s = setup
    runner = s.runner
    meta = runner.input_batch.sampling_metadata
    params = runner.requests["d"].sampling_params
    if change == "random":
        meta.all_greedy = False
    elif change == "penalties":
        meta.no_penalties = False
    elif change == "bad_words":
        meta.bad_words_token_ids = {0: [[1]]}
    elif change == "allowed_mask":
        meta.allowed_token_ids_mask = object()
    elif change == "thinking":
        meta.thinking_budget_state_holder = object()
    elif change == "logprobs":
        meta.max_num_logprobs = 0
    elif change == "prompt_logprobs":
        params.prompt_logprobs = 0
    elif change == "grammar":
        params.structured_outputs = object()
    elif change == "custom":
        meta.logitsprocs.all = [object()]
    elif change == "discard":
        runner.num_discarded_requests = 1
    elif change == "async_feedback":
        runner.input_batch.sampled_token_ids_cpu = object()
    elif change == "routed":
        runner.routed_experts_initialized = True
    elif change == "async":
        runner.use_async_scheduling = True
    elif change == "spec":
        runner.speculative_config = object()
    elif change == "multiple_tokens":
        runner.execute_model_state.scheduler_output.num_scheduled_tokens = {"d": 2}
    assert s.pipe.enqueue() is False
    assert not s.pipe.pending and runner.sample_calls == 0 and not s.events
    assert runner.sample_tokens(None) == [[7]]
    assert runner.sample_calls == 1
    assert s.events == [
        "native_sample", "native_copy_and_wait", "native_history_and_feedback"
    ]


def test_duplicate_enqueue_does_not_resample(setup):
    s = setup
    s.pipe.enqueue()
    with pytest.raises(RuntimeError, match="in-flight"):
        s.pipe.enqueue()
    assert s.runner.sample_calls == 1
    assert s.runner.sample_tokens(None) == [[7]]


@pytest.mark.parametrize("change", ["state", "metadata", "ids", "index", "grammar"])
def test_changed_retirement_identity_fails_without_bookkeeping(setup, change):
    s = setup
    s.pipe.enqueue()
    if change == "state":
        s.runner.execute_model_state = object()
    elif change == "metadata":
        s.runner.input_batch.sampling_metadata = object()
    elif change == "ids":
        s.runner.input_batch.req_ids = ["other"]
    elif change == "index":
        s.runner.input_batch.req_id_to_index = {"d": 1}
    with pytest.raises(RuntimeError, match="state changed"):
        s.runner.sample_tokens(object() if change == "grammar" else None)
    assert s.runner.sample_calls == 1 and s.runner.native_retire_calls == 0
    assert s.pipe.pending
    assert s.runner.requests["d"].output_token_ids == [11]
    with pytest.raises(RuntimeError, match="poisoned"):
        s.runner.sample_tokens(None)


def test_submission_error_keeps_ticket_and_close_retries_failed_stream_join(setup):
    s = setup
    s.runtime.fail_copy = True
    with pytest.raises(RuntimeError, match="copy failed"):
        s.pipe.enqueue()
    assert s.pipe.pending and s.runner.sample_calls == 1
    s.runtime.fail_sync = True
    installed = s.runner.sample_tokens
    with pytest.raises(RuntimeError, match="stream join failed"):
        s.pipe.close()
    assert s.pipe.pending and s.runner.sample_tokens is installed
    s.runtime.fail_sync = False
    s.pipe.close()
    s.pipe.close()
    assert not s.pipe.pending
    assert "sample_tokens" not in vars(s.runner)
    assert s.runner.requests["d"].output_token_ids == [11]
    assert s.events.count("decode_stream_sync") == 2


def test_wrong_lane_cannot_enqueue_or_read_ticket(setup):
    s = setup
    s.scope.lane = "prefill"
    with pytest.raises(RuntimeError, match="not decode"):
        s.pipe.enqueue()
    assert s.runner.sample_calls == 0
    s.scope.lane = "decode"
    s.pipe.enqueue()
    s.scope.lane = "caller"
    with pytest.raises(RuntimeError, match="not decode"):
        s.runner.sample_tokens(None)
    assert "copy_event_wait" not in s.events


def test_close_does_not_overwrite_external_replacement(setup):
    s = setup
    s.pipe.enqueue()
    def foreign(grammar):
        return None
    s.runner.sample_tokens = foreign
    with pytest.raises(RuntimeError, match="method was replaced"):
        s.pipe.retire(None)
    s.pipe.close()
    assert s.runner.sample_tokens is foreign
    assert "_sample" not in vars(s.runner) and "_to_list" not in vars(s.runner)
    assert s.events[-1] == "decode_stream_sync"


def test_consumption_twice_fails_without_second_native_sample(setup):
    s = setup
    s.pipe.enqueue()
    native = s.pipe._native_sample_tokens

    def duplicate(grammar):
        state = s.runner.execute_model_state
        s.runner._sample(state.logits, state.spec_decode_metadata)
        return native(grammar)

    s.pipe._native_sample_tokens = duplicate
    with pytest.raises(RuntimeError, match="consumption mismatch"):
        s.runner.sample_tokens(None)
    assert s.runner.sample_calls == 1
    assert s.runner.requests["d"].output_token_ids == [11]
    assert s.pipe.pending


def test_trace_diagnostics_separate_submission_from_retirement(setup):
    s = setup
    records = []
    s.pipe.diagnostic = records.append
    s.pipe.enqueue()
    assert records[0]["phase"] == "enqueue"
    assert records[0]["submission"] == 1
    assert not records[0]["private_copy_waited"]
    assert not records[0]["native_retirement_completed"]
    s.runner.sample_tokens(None)
    assert records[1]["phase"] == "retire"
    assert records[1]["submission"] == 1
    assert records[1]["private_copy_waited"]
    assert records[1]["native_retirement_completed"]
    assert records[1]["counts"]["enqueued"] == records[1]["counts"]["retired"] == 1


def test_real_copy_runtime_uses_private_pinned_buffer_and_deferred_event(monkeypatch):
    events = []
    state = NS(complete=False)

    class View:
        def copy_(self, source, *, non_blocking):
            assert non_blocking
            self.source = source
            events.append("copy")

        def tolist(self):
            assert state.complete, "CPU bytes cannot be read before completion"
            events.append("read")
            return self.source.values

    view = View()

    class Buffer:
        def __getitem__(self, key):
            assert key == slice(None, 1)
            return view

    class Event:
        def record(self, stream):
            assert stream is decode_stream
            events.append("record")

        def synchronize(self):
            events.append("wait")
            state.complete = True

    original_buffer = object()
    private = Buffer()

    def empty_like(original, *, pin_memory):
        assert original is original_buffer and pin_memory is True
        return private

    decode_stream = NS(synchronize=lambda: events.append("stream_join"))
    monkeypatch.setitem(sys.modules, "torch", NS(
        empty_like=empty_like, npu=NS(Event=Event)
    ))
    runtime = NPUSampleCopyRuntime(
        NS(pin_memory=True, sampled_token_ids_pinned_cpu=original_buffer), decode_stream
    )
    assert runtime.buffer is private and runtime.buffer is not original_buffer
    saved = runtime.enqueue(Tensor([[7]]))
    assert events == ["copy", "record"] and not state.complete
    events.append("prefill_slice")
    assert runtime.materialize(saved) == [[7]]
    assert events == ["copy", "record", "prefill_slice", "wait", "read"]
