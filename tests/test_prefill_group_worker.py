"""Real worker methods with CPU lane objects; no device allocation/compiler."""

import importlib
import os
from types import SimpleNamespace as NS

import pytest
from test_prefill_groups import fixture
from test_worker_graph import (
    overlap_graph_factory as overlap_graph_factory,
)
from test_worker_graph import (
    worker_factory as worker_factory,
)

from vllm_muxpd.lanes import LayerInterleaver


@pytest.mark.parametrize("flag", [0, None, "", []])
def test_worker_rejects_falsey_non_boolean_group_opt_in(
    worker_factory, monkeypatch, flag
):
    module = worker_factory().module
    monkeypatch.setattr(module, "validate_engine_config", lambda _: None)
    config = NS(
        scheduler_config=NS(
            batch_admission_policy_config={"compile_prefill_groups": flag}
        )
    )
    with pytest.raises(ValueError, match="compile_prefill_groups must be a boolean"):
        module._validate_worker_config(config)


@pytest.mark.parametrize("fail", [False, True])
def test_startup_prefill_scope_owns_lane_and_joins_before_restore(
    overlap_graph_factory, monkeypatch, fail
):
    state = overlap_graph_factory()
    worker = state.worker
    worker._configure_execution_backend()
    backend, runtime = worker._execution_backend, state.runtime
    caller = backend.capture()
    joined = []
    stream = worker._streams["prefill"]
    stream.synchronize = lambda: joined.append(backend.active_lane)
    monkeypatch.setattr(
        state.module.torch.npu, "current_stream", runtime.npu.current_stream
    )

    def invoke():
        with worker._prefill_group_startup():
            assert backend.active_lane == "prefill"
            worker._check_prefill_group_lane()
            if fail:
                raise ValueError("startup compilation failed")

    if fail:
        with pytest.raises(ValueError, match="startup compilation"):
            invoke()
    else:
        invoke()
    assert backend.capture() == caller
    assert joined == ([] if fail else ["prefill"])
    assert runtime.npu.records[-1][0] is stream
    assert runtime.npu.waits[-1] == (caller.stream, runtime.npu.records[-1])


def test_group_close_failure_keeps_native_model_and_can_retry(
    worker_factory, monkeypatch
):
    state = worker_factory()
    worker, events = state.worker, []
    for name, stream in worker._streams.items():
        stream.synchronize = lambda name=name: events.append(name)
    state.decode.update_stream.synchronize = lambda: events.append("update")
    fail = True

    def close():
        events.append("groups")
        assert not state.controller.snapshot()["closed"]
        if fail:
            raise RuntimeError("group teardown failed")

    groups = NS(close=close)
    worker._prefill_group_executor = groups
    monkeypatch.setattr(
        state.module.NPUWorker, "shutdown", lambda worker: events.append("native")
    )
    with pytest.raises(RuntimeError, match="group teardown"):
        worker.shutdown()
    assert events == ["prefill", "decode", "update", "groups"]
    assert worker._prefill_group_executor is groups
    assert worker.decode_runner is state.decode
    assert not state.controller.snapshot()["closed"]
    fail = False
    worker.shutdown()
    assert events[-5:] == ["prefill", "decode", "update", "groups", "native"]
    assert worker._prefill_group_executor is None
    assert state.controller.snapshot()["closed"]


def test_real_worker_callback_advances_only_successful_groups_and_unwinds_cancel():
    if os.environ.get("VLLM_VERSION") != "0.25.1":
        pytest.skip("source runtime/env.sh for pinned HUST worker contracts")
    cls = importlib.import_module("vllm_muxpd.worker").MuxPDNPUWorker
    worker = cls.__new__(cls)
    state = fixture()
    state.owner.profile_run()
    interleaver = LayerInterleaver(state.backend)
    worker._interleaver = interleaver
    worker.muxpd_stats = {"slices": 0}
    worker.vllm_config = NS(scheduler_config=NS(batch_admission_policy_config={}))
    state.executor.checkpoint = worker._prefill_group_return
    interleaver.add(
        "prefill",
        lambda: state.owner._model_forward(input_ids=[1], positions=state.positions),
        "P",
    )
    assert interleaver.step("prefill", 4).completed_layers == 4
    assert worker.muxpd_stats["slices"] == 1
    with pytest.raises(RuntimeError, match="out of order"):
        worker._prefill_group_return(1, final=False)
    interleaver.close()
    assert state.owner.model is state.model
    assert not state.executor.active and not state.executor.failed
    assert interleaver.tasks["prefill"].completed_layers == 4
    assert state.executor.calls == [2] + [1] * 9
