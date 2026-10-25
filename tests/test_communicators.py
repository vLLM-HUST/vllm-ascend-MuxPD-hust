"""Communicator routing contracts without importing torch or creating groups."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vllm_muxpd.communicators import LaneCollectiveRouter


class Communicator:
    def __init__(self, label, group=None):
        self.label = label
        self.device_group = group if group is not None else object()
        self.calls = []

    def all_reduce(self, value):
        self.calls.append(("reduce", value))
        return self.label, "reduce", value

    def all_gather(self, value, dim=-1):
        self.calls.append(("gather", value, dim))
        return self.label, "gather", value, dim

    def reduce_scatter(self, value, dim=-1):
        self.calls.append(("scatter", value, dim))
        return self.label, "scatter", value, dim


def test_three_methods_route_by_lane_and_restore_native_descriptors():
    backend = SimpleNamespace(active_lane=None)
    native, prefill = Communicator("D"), Communicator("P")
    router = LaneCollectiveRouter(backend).install([(native, prefill)])
    for lane, target in ((None, "D"), ("decode", "D"), ("prefill", "P")):
        backend.active_lane = lane
        assert native.all_reduce(10) == (target, "reduce", 10)
        assert native.all_gather(20, dim=0) == (target, "gather", 20, 0)
        assert native.reduce_scatter(30, 1) == (target, "scatter", 30, 1)
    assert router.calls == {"caller": 3, "decode": 3, "prefill": 3}
    assert len(native.calls) == 6
    assert len(prefill.calls) == 3
    router.restore()
    assert not any(name in vars(native) for name in router.METHODS)
    assert not any(name in vars(prefill) for name in router.METHODS)
    assert native.all_reduce(40) == ("D", "reduce", 40)
    router.restore()


def test_restore_keeps_foreign_override_pending_for_safe_retry():
    native, prefill = Communicator("D"), Communicator("P")
    router = LaneCollectiveRouter(SimpleNamespace(active_lane=None)).install(
        [(native, prefill)]
    )
    wrapper = native.all_gather

    def foreign(value):
        return value

    native.all_gather = foreign
    with pytest.raises(RuntimeError, match="changed externally"):
        router.restore()
    assert native.all_gather is foreign
    assert len(router._installed) == 1  # The other two methods were restored.
    native.all_gather = wrapper
    router.restore()
    assert not router._installed
    assert not any(name in vars(native) for name in router.METHODS)


def test_aliases_share_sibling_group_and_repeated_pairs_install_once():
    native_group, prefill_group = object(), object()
    tp = Communicator("Dtp", native_group)
    ep = Communicator("Dep", native_group)
    ptp = Communicator("Ptp", prefill_group)
    pep = Communicator("Pep", prefill_group)
    backend = SimpleNamespace(active_lane="prefill")
    router = LaneCollectiveRouter(backend).install([(tp, ptp), (ep, pep), (tp, ptp)])
    assert len(router._installed) == 6
    assert tp.all_reduce(1) == ("Ptp", "reduce", 1)
    assert ep.all_gather(2) == ("Pep", "gather", 2, -1)
    assert tp.reduce_scatter(3) == ("Ptp", "scatter", 3, -1)
    router.restore()


@pytest.mark.parametrize(
    "invalid", ["same_group", "native_group_alias", "alias", "duplicate", "cycle"]
)
def test_invalid_group_routing_rolls_back_every_installed_method(invalid):
    native, prefill = Communicator("D"), Communicator("P")
    if invalid == "same_group":
        pairs = [(native, Communicator("P", native.device_group))]
    elif invalid == "native_group_alias":
        other = Communicator("Dother")
        pairs = [(native, Communicator("P", other.device_group)), (other, prefill)]
    elif invalid == "alias":
        pairs = [
            (native, prefill),
            (Communicator("alias", native.device_group), Communicator("other")),
        ]
    elif invalid == "duplicate":
        pairs = [(native, prefill), (native, Communicator("other"))]
    else:
        pairs = [(native, prefill), (prefill, native)]
    with pytest.raises(ValueError):
        LaneCollectiveRouter(SimpleNamespace(active_lane=None)).install(pairs)
    assert not any(name in vars(native) for name in LaneCollectiveRouter.METHODS)
    assert not any(name in vars(prefill) for name in LaneCollectiveRouter.METHODS)


def test_failed_target_poisoning_and_native_instance_method_restoration():
    native, prefill = Communicator("D"), Communicator("P")

    def original(value):
        return "custom-native", value

    native.all_reduce = original

    def fail(value):
        raise RuntimeError("prefill collective failed")

    prefill.all_gather = fail
    backend = SimpleNamespace(active_lane="prefill")
    router = LaneCollectiveRouter(backend).install([(native, prefill)])
    with pytest.raises(RuntimeError, match="prefill collective failed"):
        native.all_gather(1)
    backend.active_lane = "decode"
    with pytest.raises(RuntimeError, match="poisoned"):
        native.all_reduce(2)
    router.restore()
    assert native.all_reduce is original
    assert native.all_reduce(3) == ("custom-native", 3)


def test_unknown_lane_poisoning():
    native, prefill = Communicator("D"), Communicator("P")
    backend = SimpleNamespace(active_lane="unexpected")
    router = LaneCollectiveRouter(backend).install([(native, prefill)])
    with pytest.raises(RuntimeError, match="Unknown MuxPD collective lane"):
        native.all_reduce(1)
    backend.active_lane = None
    with pytest.raises(RuntimeError, match="poisoned"):
        native.all_reduce(2)
    router.restore()


def test_install_failure_and_existing_wrappers_are_not_overwritten():
    native, prefill = Communicator("D"), Communicator("P")
    prefill.reduce_scatter = None
    with pytest.raises(ValueError, match="reduce_scatter"):
        LaneCollectiveRouter(SimpleNamespace(active_lane=None)).install(
            [(native, prefill)]
        )
    assert not any(name in vars(native) for name in LaneCollectiveRouter.METHODS)
    prefill = Communicator("P")

    def gate(value):
        return value

    gate._muxpd_collective_gate = object()
    native.all_reduce = gate
    with pytest.raises(RuntimeError, match="unwrapped communicators"):
        LaneCollectiveRouter(SimpleNamespace(active_lane=None)).install(
            [(native, prefill)]
        )
    assert native.all_reduce is gate


def test_external_replacement_preserved_and_other_methods_restored():
    native, prefill = Communicator("D"), Communicator("P")
    router = LaneCollectiveRouter(SimpleNamespace(active_lane=None)).install(
        [(native, prefill)]
    )

    def replacement(value):
        return "external", value

    native.all_gather = replacement
    with pytest.raises(RuntimeError, match="changed externally"):
        router.restore()
    assert native.all_gather is replacement
    assert "all_reduce" not in vars(native)
    assert "reduce_scatter" not in vars(native)


def test_nested_install_and_cross_thread_dispatch_rejected():
    native, prefill = Communicator("D"), Communicator("P")
    backend = SimpleNamespace(active_lane=None)
    router = LaneCollectiveRouter(backend).install([(native, prefill)])
    with pytest.raises(RuntimeError, match="installed twice"):
        router.install([(native, prefill)])
    with pytest.raises(RuntimeError, match="unwrapped communicators"):
        LaneCollectiveRouter(backend).install([(native, prefill)])
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(native.all_reduce, 1)
        with pytest.raises(RuntimeError, match="one host thread"):
            future.result()
    assert router.calls == {"caller": 0, "decode": 0, "prefill": 0}
    router.restore()
