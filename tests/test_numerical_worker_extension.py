from types import SimpleNamespace

import pytest

from vllm_muxpd import numerical_workers as workers


def worker_fixture(selector):
    module, name = selector.rsplit(".", 1)
    cls = type(name, (), {"__module__": module})
    return cls()


def test_extension_exposes_only_unique_rpc_method():
    names = [
        name
        for name in dir(workers.NumericalObserverExtension)
        if not name.startswith("__")
    ]
    assert names == ["muxpd_install_numerical_observers"]


@pytest.mark.parametrize(
    "selector,lanes",
    [
        (workers.NATIVE_WORKER, ["native"]),
        (workers.MUXPD_WORKER, ["prefill", "decode"]),
    ],
)
def test_install_once_retains_production_class_and_observes_after_warmup(
    monkeypatch, selector, lanes
):
    events = ["warmup-returned"]
    worker = worker_fixture(selector)
    original_type = type(worker)
    worker.rank = 2
    worker.model_runner = SimpleNamespace(execute_model_state=None)
    worker.decode_runner = SimpleNamespace(execute_model_state=None)
    monkeypatch.setattr(
        workers, "attach_observer", lambda w, r, lane: events.append(lane)
    )
    monkeypatch.setattr(
        workers, "compiler_constants_snapshot", lambda w: events.append("constants")
    )
    result = workers.install_observers(worker)
    assert events == ["warmup-returned", *lanes, "constants"]
    assert type(worker) is original_type
    assert result["worker_selector"] == selector
    assert result["observer_lanes"] == lanes
    assert result["installed"] and result["rank"] == 2
    with pytest.raises(RuntimeError, match="already attempted"):
        workers.install_observers(worker)
    assert events == ["warmup-returned", *lanes, "constants"]


def test_active_serving_is_rejected_before_mutation():
    worker = worker_fixture(workers.NATIVE_WORKER)
    worker.model_runner = SimpleNamespace(execute_model_state=("pending",))
    with pytest.raises(RuntimeError, match="before serving"):
        workers.install_observers(worker)
    assert not hasattr(worker, "_muxpd_numerics_install_state")


def test_failed_partial_install_is_not_retried(monkeypatch):
    worker = worker_fixture(workers.MUXPD_WORKER)
    worker.model_runner = SimpleNamespace(execute_model_state=None)
    worker.decode_runner = SimpleNamespace(execute_model_state=None)
    events = []

    def attach(w, r, lane):
        events.append(lane)
        if lane == "decode":
            raise RuntimeError("diagnostic install failure")

    monkeypatch.setattr(workers, "attach_observer", attach)
    with pytest.raises(RuntimeError, match="diagnostic install failure"):
        workers.install_observers(worker)
    assert worker._muxpd_numerics_install_state == "failed"
    with pytest.raises(RuntimeError, match="already attempted"):
        workers.install_observers(worker)
    assert events == ["prefill", "decode"]


def test_postwarmup_checkpoint_calls_bound_original_once(monkeypatch):
    events = []
    sentinel = object()

    def original(index, *, final):
        events.append(("original", index, final))
        return sentinel

    worker = SimpleNamespace(
        _prefill_group_executor=SimpleNamespace(checkpoint=original)
    )
    monkeypatch.setattr(
        workers,
        "observe_prefill_group_return",
        lambda w, i, *, final: events.append(("observer", i, final)),
    )
    assert workers.install_group_observer(worker)
    assert worker._prefill_group_executor.checkpoint(4, final=False) is sentinel
    assert events == [("observer", 4, False), ("original", 4, False)]
    assert workers.install_group_observer(SimpleNamespace()) is False


def test_native_dynamic_base_registration_has_no_collisions():
    class WorkerBase:
        def load_model(self):
            return "native load"

        def compile_or_warm_up_model(self):
            return "native warmup"

    class ProductionWorker(WorkerBase):
        pass

    original_load = ProductionWorker.load_model
    original_warmup = ProductionWorker.compile_or_warm_up_model
    extension = workers.NumericalObserverExtension
    # Match pinned vLLM worker_base registration: check every non-dunder member,
    # then append the extension as a base to the original worker class.
    for name in dir(extension):
        if not name.startswith("__"):
            assert not hasattr(ProductionWorker, name)
    ProductionWorker.__bases__ += (extension,)
    worker = ProductionWorker()
    assert type(worker) is ProductionWorker
    assert ProductionWorker.load_model is original_load
    assert ProductionWorker.compile_or_warm_up_model is original_warmup
    assert worker.load_model() == "native load"
    assert worker.compile_or_warm_up_model() == "native warmup"
    assert callable(worker.muxpd_install_numerical_observers)
