"""P graph ownership and finite capture invariants without device allocation."""

from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest

from vllm_muxpd.piecewise_prefill import ShortPrefillGraphs


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    uniform: bool = False
    num_reqs: int | None = None


@pytest.fixture
def setup():
    ctx = NS(
        cudagraph_runtime_mode="PIECEWISE",
        capturing=False,
        batch_descriptor=Descriptor(128),
    )
    state = NS(lane="prefill", params=("D", ()), wrappers=[], events=[])

    class Wrapper:
        runtime_mode = "PIECEWISE"

        def __init__(self):
            self.graph_pool = "native-global"
            self.concrete_aclgraph_entries = {}
            state.wrappers.append(self)

        def __call__(self):
            if ctx.cudagraph_runtime_mode != self.runtime_mode:
                state.events.append("bypass")
                return "native"
            desc = ctx.batch_descriptor
            entry = self.concrete_aclgraph_entries.get(desc)
            if entry is None:
                ctx.capturing = True  # pinned wrapper deliberately leaves True
                self.concrete_aclgraph_entries[desc] = NS(
                    aclgraph=object(), input_addresses=[]
                )
                state.events.append("capture")
            else:
                state.events.append("replay")
            return "native"

    native_d = Wrapper()
    native_d.runtime_mode = "FULL"
    runtime = NS(
        wrapper_cls=Wrapper,
        piecewise_mode="PIECEWISE",
        context=lambda: ctx,
        wrappers=lambda: tuple(state.wrappers),
        input_addresses=lambda args: [],
        new_pool=lambda: object(),
        graph_params_snapshot=lambda: state.params,
    )

    def assert_lane():
        if state.lane != "prefill":
            raise RuntimeError("wrong P lane")

    owner = ShortPrefillGraphs(runtime, assert_lane=assert_lane, limit=512)
    result = NS(owner=owner, ctx=ctx, state=state, Wrapper=Wrapper, d=native_d)
    yield result
    owner.close()


def prepare(s):
    with s.owner.profile():
        p = s.Wrapper()
    return p


def capture(s, p, sizes=(128, 512)):
    descs = [Descriptor(n) for n in sizes]
    with s.owner.capture(descs):
        for desc in descs:
            s.ctx.batch_descriptor = desc
            assert p() == "native"
            assert s.ctx.capturing is False


def test_private_pool_entries_full_bypass_and_whole_replay(setup):
    s = setup
    p = prepare(s)
    assert p.graph_pool != s.d.graph_pool
    capture(s, p)
    original_entries = dict(p.concrete_aclgraph_entries)
    s.ctx.cudagraph_runtime_mode = "FULL"
    s.ctx.capturing = True
    s.state.lane = "decode"
    assert p() == "native"
    assert s.ctx.capturing is True
    assert p.concrete_aclgraph_entries == original_entries
    s.state.lane = "prefill"
    s.ctx.cudagraph_runtime_mode = "PIECEWISE"
    s.ctx.capturing = False
    with s.owner.serving():
        assert s.owner.captured
        p()
    assert s.owner.stats == {"captures": 2, "replays": 1, "whole_forwards": 1}


def test_missing_serving_gear_rejected_before_native_capture(setup):
    s = setup
    p = prepare(s)
    capture(s, p)
    s.ctx.batch_descriptor = Descriptor(256)
    with pytest.raises(RuntimeError, match="Serving-time"):
        with s.owner.serving():
            p()
    assert s.state.events == ["capture", "capture"]
    assert Descriptor(256) not in p.concrete_aclgraph_entries


@pytest.mark.parametrize("bad", ["wrong_lane", "unknown_wrapper", "pool", "oversize"])
def test_serving_ownership_failures(setup, bad):
    s = setup
    p = prepare(s)
    capture(s, p)
    with pytest.raises(RuntimeError):
        with s.owner.serving():
            if bad == "wrong_lane":
                s.state.lane = "decode"
            elif bad == "unknown_wrapper":
                p = s.Wrapper()
            elif bad == "pool":
                p.graph_pool = "native-global"
            else:
                s.ctx.batch_descriptor = Descriptor(513)
            p()
    assert s.state.events == ["capture", "capture"]


@pytest.mark.parametrize("calls", [0, 2])
def test_whole_forward_requires_each_subgraph_exactly_once(setup, calls):
    s = setup
    p = prepare(s)
    capture(s, p)
    with pytest.raises(RuntimeError, match="every owned subgraph once"):
        with s.owner.serving():
            for _ in range(calls):
                p()


def test_capture_cannot_mutate_decode_graph_params(setup):
    s = setup
    p = prepare(s)
    with pytest.raises(RuntimeError, match="GraphParams"):
        with s.owner.capture([Descriptor(128)]):
            p()
            s.state.params = ("D", ("P event",))
    assert s.ctx.capturing is False
    assert s.owner.snapshot()["phase"] == "failed"


def test_incomplete_startup_capture_fails_closed(setup):
    s = setup
    p = prepare(s)
    with pytest.raises(RuntimeError, match="every bounded gear"):
        with s.owner.capture([Descriptor(128), Descriptor(512)]):
            p()


def test_close_clears_only_owned_graph_entries_and_restores_method(setup):
    s = setup
    original = s.Wrapper.__call__
    p = prepare(s)
    capture(s, p)
    s.d.concrete_aclgraph_entries[Descriptor(1, True, 1)] = NS(
        aclgraph=object(), input_addresses=[]
    )
    with s.owner.serving():
        p()
    assert s.owner._current
    s.owner.close()
    assert not s.owner._current
    assert s.Wrapper.__call__ is original
    assert not p.concrete_aclgraph_entries
    assert len(s.d.concrete_aclgraph_entries) == 1
    s.owner.close()


def test_new_profile_does_not_claim_preexisting_wrappers(setup):
    s = setup
    foreign = s.Wrapper()
    p = prepare(s)
    assert foreign.graph_pool == "native-global"
    assert p.graph_pool != foreign.graph_pool


def test_serving_checks_addresses_even_when_native_debug_is_off(setup):
    s = setup
    p = prepare(s)
    capture(s, p)
    entry = p.concrete_aclgraph_entries[s.ctx.batch_descriptor]
    entry.input_addresses = [123]
    with pytest.raises(RuntimeError, match="captured addresses"):
        with s.owner.serving():
            p()
    assert s.owner.stats["replays"] == 0


def test_pool_manifest_uses_native_handle_values_without_object_repr():
    from vllm_muxpd.piecewise_prefill import graph_pool_identity

    first, other = tuple([17, 31]), tuple([17, 31])
    assert first is not other
    assert graph_pool_identity(first) == graph_pool_identity(other)
    assert graph_pool_identity(first) != graph_pool_identity((17, 32))

    class Opaque:
        def __repr__(self):
            raise AssertionError("Pool diagnostics must not inspect arbitrary contents")

    opaque = Opaque()
    assert graph_pool_identity(opaque)["kind"] == "python_object_identity_only"
    assert graph_pool_identity(opaque)["object_id"] == hex(id(opaque))


def test_pool_manifest_has_every_p_wrapper_and_bounded_entry_metadata(setup):
    import json

    s = setup
    p = prepare(s)
    capture(s, p)
    snap = s.owner.snapshot()
    assert snap["wrapper_metadata_complete"]
    assert len(snap["wrapper_metadata"]) == 1
    row = snap["wrapper_metadata"][0]
    assert row["runtime_mode"] == "PIECEWISE"
    assert row["pool"] == snap["pool"]
    assert row["entries_complete"]
    assert row["entries"] == [
        {"num_tokens": size, "num_reqs": None, "uniform": False, "captured": True}
        for size in (128, 512)
    ]
    assert json.loads(json.dumps(snap)) == snap
    snap["wrapper_metadata"][0]["entries"].clear()
    assert len(s.owner.snapshot()["wrapper_metadata"][0]["entries"]) == 2
    assert s.owner.pool_matches(p.graph_pool) is True
    assert s.owner.pool_matches(s.d.graph_pool) is False
