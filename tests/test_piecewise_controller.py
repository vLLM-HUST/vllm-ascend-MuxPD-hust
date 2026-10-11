"""Exercise the actual controller with fake native dispatch/capture routines."""

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
from test_decode_graph import Runner, Runtime, Stream, config

from vllm_muxpd.config import MuxPDConfig
from vllm_muxpd.decode_graph import DecodeGraphController, validate_decode_graph_config


@dataclass(frozen=True)
class Desc:
    num_tokens: int
    uniform: bool = False
    num_reqs: int | None = None


@pytest.fixture
def prepared():
    r = Runtime()
    r.piecewise_mode, r.full_decode_mode = "PIECEWISE", "FULL_DECODE_ONLY"
    r.lane, r.events, r.wrappers_list = "caller", [], []
    r.context = lambda: r.forward._forward_context
    r.wrappers = lambda: tuple(r.wrappers_list)
    r.input_addresses = lambda args: []
    r.new_pool = object
    r.graph_params_snapshot = lambda: r.params
    r.validate_piecewise_backend = lambda owner: None
    r.guard_capture_disabled = lambda: r.events.append("capture_disabled")

    class Wrapper:
        def __init__(self, runnable, mode):
            self.runnable, self.runtime_mode = runnable, mode
            self.graph_pool = "native-global"
            self.concrete_aclgraph_entries = {}
            r.wrappers_list.append(self)

        def __call__(self, tokens):
            ctx = r.context()
            if ctx.cudagraph_runtime_mode != self.runtime_mode:
                return self.runnable(tokens)
            desc = ctx.batch_descriptor
            if desc not in self.concrete_aclgraph_entries:
                ctx.capturing = True
                result = self.runnable(tokens)
                self.concrete_aclgraph_entries[desc] = NS(
                    aclgraph=object(), input_addresses=[]
                )
                r.events.append(("capture", self.runtime_mode, tokens, r.lane))
                return result
            r.events.append(("replay", self.runtime_mode, tokens, r.lane))
            return None

    r.wrapper_cls = Wrapper
    r.wrap = lambda raw, decode: Wrapper(raw, "FULL")
    cfg = config()
    cfg.compilation_config.cudagraph_mode = "FULL_AND_PIECEWISE"
    cfg.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 128, 512]
    cfg.compilation_config.max_cudagraph_capture_size = 512
    cfg.scheduler_config.max_num_batched_tokens = 16384
    cfg.scheduler_config.batch_admission_policy_config.update(
        stream_mode="overlap",
        overlap_communication="isolated",
        piecewise_short_prefill_tokens=512,
    )

    class Dispatcher:
        def initialize_cudagraph_keys(self, mode):
            self.cudagraph_keys = {
                "FULL": {Desc(n, True, n) for n in (1, 2, 4)},
                "PIECEWISE": {Desc(n) for n in (1, 2, 4, 128, 512)},
            }
            self.cudagraph_mode = mode

        def get_capture_descs(self):
            return [
                (mode, sorted(descs, key=lambda d: -d.num_tokens))
                for mode, descs in self.cudagraph_keys.items()
                if descs
            ]

    class Model:
        def __init__(self):
            self.parts = []

        def __call__(self, tokens):
            for part in self.parts:
                part(tokens)
                if r.context().cudagraph_runtime_mode == "PIECEWISE":
                    assert r.context().capturing is False
            return None

    class NativeRunner(Runner):
        def __init__(self, lane):
            super().__init__(cfg, r)
            self.lane = lane
            self.cudagraph_dispatcher = Dispatcher()
            self.model = Model()

        def _determine_batch_execution_and_padding(
            self,
            num_tokens,
            force_eager=False,
            force_uniform_decode=None,
        ):
            if force_eager:
                return "NONE"
            if self.lane == "prefill":
                assert force_uniform_decode is False
                return "PIECEWISE"
            return "FULL"

        def _model_forward(self, num_tokens):
            return self.model(num_tokens)

        def _dummy_run(self, num_tokens, force_eager=False):
            mode = self._determine_batch_execution_and_padding(num_tokens, force_eager)
            ctx = r.context()
            ctx.cudagraph_runtime_mode = mode
            ctx.capturing = False
            ctx.batch_descriptor = (
                Desc(num_tokens, True, num_tokens)
                if mode == "FULL"
                else Desc(num_tokens)
            )
            return self._model_forward(num_tokens)

        def profile_run(self):
            assert r.lane == "prefill"
            self.model.parts = [Wrapper(lambda n: None, "PIECEWISE") for _ in range(2)]
            return self._dummy_run(16384, True)

        def capture_model(self):
            r.events.append(
                (
                    "native_capture",
                    self.lane,
                    r.lane,
                    r.npu.current_stream(),
                    r.workspace._manager,
                )
            )
            for mode, descs in self.cudagraph_dispatcher.get_capture_descs():
                for desc in descs:
                    self._dummy_run(desc.num_tokens, True)  # native NONE warmup
                    self._dummy_run(desc.num_tokens)
            return 11 if self.lane == "prefill" else 22

    p, d = NativeRunner("prefill"), NativeRunner("decode")
    pstream, dstream = Stream("prefill"), Stream("decode")
    pworkspace, dworkspace = object(), object()

    @contextmanager
    def p_scope():
        previous = r.lane
        with r.capture_context(p, pstream, pworkspace):
            r.lane = "prefill"
            try:
                yield
            finally:
                r.lane = previous

    # The native capture helper is nested by the controller, as in the worker.
    # Use a simple fake here: fixed_decode_capture_context disallows nesting
    # and worker startup scope restores streams directly rather than using it.
    @contextmanager
    def startup():
        previous = (r.lane, r.npu.current, r.workspace._manager)
        r.lane, r.npu.current, r.workspace._manager = "prefill", pstream, pworkspace
        try:
            yield
        finally:
            r.lane, r.npu.current, r.workspace._manager = previous

    def assert_p():
        assert r.lane == "prefill"
        assert r.npu.current_stream() is pstream
        assert r.workspace._manager is pworkspace

    ctrl = DecodeGraphController(
        p,
        d,
        dstream,
        dworkspace,
        runtime=r,
        prefill_startup_scope=startup,
        assert_prefill_lane=assert_p,
    )
    ctrl.install_prefill_guards()
    ctrl.attach_shared_model()
    p.cudagraph_dispatcher.initialize_cudagraph_keys("FULL_AND_PIECEWISE")
    d.cudagraph_dispatcher.initialize_cudagraph_keys("FULL_AND_PIECEWISE")
    r.params = object()
    result = NS(
        ctrl=ctrl,
        p=p,
        d=d,
        runtime=r,
        startup=startup,
        pstream=pstream,
        dstream=dstream,
        pworkspace=pworkspace,
        dworkspace=dworkspace,
        config=cfg,
    )
    yield result
    ctrl.close()


def test_native_lifecycle_pool_dispatch_capture_and_serving(prepared):
    s = prepared
    s.p.profile_run()
    assert s.ctrl.snapshot()["prefill_piecewise"]["wrappers"] == 2
    assert s.p.cudagraph_dispatcher.cudagraph_mode == "PIECEWISE"
    assert s.d.cudagraph_dispatcher.cudagraph_mode == "FULL_DECODE_ONLY"
    with s.ctrl.redirect_owner_capture():
        assert s.p.capture_model() == 33
    captures = [event for event in s.runtime.events if event[0] == "native_capture"]
    assert [event[1] for event in captures] == ["prefill", "decode"]
    assert captures[0][3:] == (s.pstream, s.pworkspace)
    assert captures[1][3:] == (s.dstream, s.dworkspace)
    before = s.ctrl.snapshot()["prefill_piecewise"]
    assert before["captures"] == 10
    snapshot = s.ctrl.snapshot()
    assert snapshot["prefill_decode_pool_equal"] is False
    assert snapshot["decode_graph_metadata"]["runtime_mode"] == "FULL"
    assert snapshot["decode_graph_metadata"]["pool"] != before["pool"]
    assert [
        entry["num_tokens"] for entry in snapshot["decode_graph_metadata"]["entries"]
    ] == [1, 2, 4]
    assert before["wrapper_metadata_complete"]
    assert all(
        row["runtime_mode"] == "PIECEWISE"
        and row["pool"] == before["pool"]
        and row["entries_complete"]
        for row in before["wrapper_metadata"]
    )
    assert s.d.model.graph_pool != next(iter(s.ctrl._piecewise._wrappers)).graph_pool
    with s.startup():
        with s.ctrl.full_prefill_compile_context():
            s.p._dummy_run(1)
    after = s.ctrl.snapshot()["prefill_piecewise"]
    assert after["captures"] == before["captures"]
    assert after["replays"] == 2 and after["whole_forwards"] == 1
    s.p._dummy_run(12000)
    assert s.runtime.context().cudagraph_runtime_mode == "NONE"
    assert s.ctrl.snapshot()["prefill_piecewise"] == after


def test_capture_failure_disables_global_flag_restores_scopes(prepared, monkeypatch):
    s = prepared
    s.p.profile_run()
    caller = (s.runtime.npu.current_stream(), s.runtime.workspace._manager)

    def fail():
        assert s.runtime.lane == "prefill"
        raise RuntimeError("native P capture failure")

    monkeypatch.setattr(s.ctrl, "_prefill_capture_model", fail)
    with pytest.raises(RuntimeError, match="native P capture failure"):
        s.ctrl.capture_decode()
    assert s.ctrl.snapshot()["failed"]
    assert s.runtime.events[-1] == "capture_disabled"
    assert (s.runtime.npu.current_stream(), s.runtime.workspace._manager) == caller
    assert not s.ctrl._prefill_capture_active and not s.ctrl._compile_prefill_active


def test_serving_missing_piecewise_entry_does_not_recapture(prepared):
    s = prepared
    s.p.profile_run()
    s.ctrl.capture_decode()
    wrapper = next(iter(s.ctrl._piecewise._wrappers))
    wrapper.concrete_aclgraph_entries.pop(Desc(128))
    before = list(s.runtime.events)
    with pytest.raises(RuntimeError, match="Serving-time"):
        with s.startup():
            with s.ctrl.full_prefill_compile_context():
                s.p._dummy_run(128)
    assert s.ctrl.snapshot()["failed"]
    assert not any(e[0] == "capture" for e in s.runtime.events[len(before) :])


@pytest.mark.parametrize("value", [-1, True, 1.5, "512"])
def test_piecewise_limit_typed(value):
    with pytest.raises(ValueError, match="piecewise_short_prefill_tokens"):
        MuxPDConfig(piecewise_short_prefill_tokens=value)


@pytest.mark.parametrize(
    "change",
    [
        {"stream_mode": "fenced", "overlap_communication": "shared"},
        {"decode_graph": False},
        {"overlap_communication": "shared"},
    ],
)
def test_piecewise_requires_supported_combination(change):
    kwargs = dict(
        execution="decoupled",
        decode_graph=True,
        stream_mode="overlap",
        overlap_communication="isolated",
        piecewise_short_prefill_tokens=512,
    )
    kwargs.update(change)
    with pytest.raises(ValueError):
        MuxPDConfig(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cudagraph_copy_inputs", True),
        ("use_inductor_graph_partition", True),
        ("cudagraph_mode", "FULL_DECODE_ONLY"),
        ("cudagraph_capture_sizes", [1, 2, 4, 128]),
    ],
)
def test_piecewise_graph_config_fail_closed(prepared, field, value):
    cfg = prepared.config
    setattr(cfg.compilation_config, field, value)
    with pytest.raises(ValueError):
        validate_decode_graph_config(
            cfg, cfg.scheduler_config.batch_admission_policy_config
        )


def test_native_dispatcher_contract_without_device_allocation():
    import os

    if os.environ.get("VLLM_VERSION") != "0.25.1":
        pytest.skip("source runtime/env.sh for pinned native dispatcher contract")
    from vllm.config import CUDAGraphMode as Mode
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    compilation = NS(
        cudagraph_mode=Mode.FULL_AND_PIECEWISE,
        is_attention_compiled_piecewise=lambda: True,
        cudagraph_capture_sizes=[1, 2, 4, 8, 128, 256, 512],
        max_cudagraph_capture_size=512,
        compile_sizes=[],
    )
    cfg = NS(
        compilation_config=compilation,
        num_speculative_tokens=0,
        lora_config=None,
        scheduler_config=NS(max_num_seqs=8),
    )
    ctrl = DecodeGraphController.__new__(DecodeGraphController)
    ctrl._patches = []
    ctrl.piecewise_short_prefill_tokens = 512
    ctrl.runtime = NS(
        piecewise_mode=Mode.PIECEWISE,
        full_mode=Mode.FULL,
        full_decode_mode=Mode.FULL_DECODE_ONLY,
    )
    for prefill in (True, False):
        dispatcher = CudagraphDispatcher(cfg)
        runner = NS(cudagraph_dispatcher=dispatcher, vllm_config=cfg)
        ctrl._isolate_dispatcher(runner, prefill=prefill)
        dispatcher.initialize_cudagraph_keys(Mode.FULL_AND_PIECEWISE, 1)
        groups = dispatcher.get_capture_descs()
        assert len(groups) == 1
        mode, descs = groups[0]
        assert mode == (Mode.PIECEWISE if prefill else Mode.FULL)
        assert [d.num_tokens for d in descs] == (
            [512, 256, 128, 8, 4, 2, 1] if prefill else [8, 4, 2, 1]
        )
        assert dispatcher.dispatch(1, uniform_decode=True)[0] == mode
        assert dispatcher.dispatch(129, uniform_decode=False)[0] == (
            Mode.PIECEWISE if prefill else Mode.NONE
        )
        assert dispatcher.dispatch(513, uniform_decode=False)[0] == Mode.NONE
