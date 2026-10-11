"""No NPU or compiler execution: exercise ownership and live group suspension."""

import ast
import os
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vllm_muxpd.config import MuxPDConfig
from vllm_muxpd.lanes import LayerInterleaver
from vllm_muxpd.prefill_groups import PrefillGroupExecutor, clone_group_config


class Mode(Enum):
    NONE = 0
    FULL_DECODE_ONLY = 1


class FakeGroup:
    def __init__(self, *, layers, vllm_config):
        self.layers, self.vllm_config = layers, vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.compile_calls = 0

    def __call__(self, positions, hidden, residual):
        self.compile_calls += 1
        assert self.compile_calls == 1
        self.aot_compiled_fn = lambda group, *args: group.forward(*args)
        return self.aot_compiled_fn(self, positions, hidden, residual)

    def run(self, positions, hidden, residual):
        for layer in self.layers:
            hidden, residual = layer.forward(
                positions=positions, hidden_states=hidden, residual=residual
            )
        return hidden, residual


class G0(FakeGroup):
    def forward(self, positions, hidden, residual):
        return self.run(positions, hidden, residual)


class G1(FakeGroup):
    def forward(self, positions, hidden, residual):
        return self.run(positions, hidden, residual)


class G2(FakeGroup):
    def forward(self, positions, hidden, residual):
        return self.run(positions, hidden, residual)


class G3(FakeGroup):
    def forward(self, positions, hidden, residual):
        return self.run(positions, hidden, residual)


class G4(FakeGroup):
    def forward(self, positions, hidden, residual):
        return self.run(positions, hidden, residual)


class G5(FakeGroup):
    def forward(self, positions, hidden, residual):
        return self.run(positions, hidden, residual)


class G6(FakeGroup):
    def forward(self, positions, hidden, residual):
        return self.run(positions, hidden, residual)


class G7(FakeGroup):
    def forward(self, positions, hidden, residual):
        return self.run(positions, hidden, residual)


class G8(FakeGroup):
    def forward(self, positions, hidden, residual):
        return self.run(positions, hidden, residual)


class G9(FakeGroup):
    def forward(self, positions, hidden, residual):
        return self.run(positions, hidden, residual)


CLASSES = (G0, G1, G2, G3, G4, G5, G6, G7, G8, G9)


class Backend:
    state = "caller"

    def capture(self):
        return self.state

    def restore(self, state):
        self.state = state

    def fence(self):
        pass


def fixture():
    backend, ctx, config_stack = (
        Backend(),
        NS(cudagraph_runtime_mode=Mode.NONE, capturing=False, skip_compiled=True),
        [],
    )
    seen = []

    class Layer:
        def __init__(self, index):
            self.index = index

        def forward(self, *, positions, hidden_states, residual):
            seen.append((self.index, residual))
            return (
                hidden_states + 1,
                hidden_states if residual is None else residual + 1,
            )

    layers = [Layer(i) for i in range(40)]
    config = NS(
        compilation_config=NS(
            cudagraph_mode=Mode.FULL_DECODE_ONLY,
            static_forward_context={"layer": layers[0]},
            traced_files={"D.py"},
            pass_config={"x": []},
        ),
        scheduler_config=NS(max_num_batched_tokens=32),
    )
    model = object()
    owner = NS(vllm_config=config, model=model)
    positions = NS(shape=(3, 16))
    owner._model_forward = lambda **kwargs: owner.model(**kwargs)
    owner.profile_run = lambda: owner._model_forward(input_ids=[1], positions=positions)
    backbone = NS(
        layers=layers,
        start_layer=0,
        end_layer=40,
        aux_hidden_state_layers=(),
        embed_input_ids=lambda _: 0,
        norm=lambda h, r: (h + 100, r),
    )

    @contextmanager
    def cfg_scope(value):
        config_stack.append(value)
        try:
            yield
        finally:
            assert config_stack.pop() is value

    @contextmanager
    def startup_scope():
        previous, backend.state = backend.state, "P"
        try:
            yield
        finally:
            backend.state = previous

    def lane_check():
        assert backend.state == "P"

    checkpoint = []
    runtime = NS(
        classes=CLASSES,
        context=lambda: ctx,
        config_context=cfg_scope,
        partition_context=lambda _: nullcontext(),
        cache_root="/fake/cache",
    )
    executor = PrefillGroupExecutor(
        owner,
        backbone,
        checkpoint=lambda index, final: checkpoint.append((index, final)),
        lane_check=lane_check,
        startup_scope=startup_scope,
        is_startup=lambda: False,
        runtime=runtime,
    )
    executor.install()
    return NS(
        executor=executor,
        owner=owner,
        backbone=backbone,
        model=model,
        backend=backend,
        ctx=ctx,
        stack=config_stack,
        seen=seen,
        checkpoints=checkpoint,
        positions=positions,
    )


def test_config_opt_in_and_combinations():
    assert MuxPDConfig().compile_prefill_groups is False
    valid = dict(
        compile_prefill_groups=True,
        decode_graph=True,
        execution="decoupled",
        stream_mode="overlap",
        overlap_communication="isolated",
        layers_per_slice=4,
    )
    assert MuxPDConfig(**valid).compile_prefill_groups
    for change in [
        dict(layers_per_slice=1),
        dict(prefill_layer_token_budget=100),
        dict(adaptive_layers=True),
        dict(compile_full_prefill=True),
        dict(decode_graph=False),
        dict(stream_mode="fenced"),
        dict(overlap_communication="shared"),
    ]:
        with pytest.raises(ValueError):
            MuxPDConfig(**(valid | change))


def test_config_clone_retains_layer_identity_and_independent_mutable_state():
    class Module:
        def __deepcopy__(self, memo):
            raise AssertionError("Must not copy live layer/weight storage")

    layer = Module()
    cfg = NS(
        compilation_config=NS(
            cudagraph_mode=Mode.FULL_DECODE_ONLY,
            static_forward_context={"layer": layer},
            traced_files={"D"},
            pass_config={"nested": []},
        ),
        scheduler_config=NS(max_num_batched_tokens=32),
    )
    a, b = (clone_group_config(cfg, i, "/cache") for i in (0, 1))
    assert a.compilation_config.static_forward_context["layer"] is layer
    assert (
        a.compilation_config.static_forward_context
        is not cfg.compilation_config.static_forward_context
    )
    a.compilation_config.pass_config["nested"].append(1)
    assert (
        not cfg.compilation_config.pass_config["nested"]
        and not b.compilation_config.pass_config["nested"]
    )
    assert a.compilation_config.cache_dir != b.compilation_config.cache_dir
    assert cfg.compilation_config.cudagraph_mode is Mode.FULL_DECODE_ONLY


def test_source_has_ten_real_distinct_forward_definitions():
    path = Path(__file__).parents[1] / "src/vllm_muxpd/prefill_group_models.py"
    tree = ast.parse(path.read_text())
    groups = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name.startswith("PrefillGroup")
    ]
    forwards = [
        next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        for cls in groups
    ]
    assert len(groups) == len({f.lineno for f in forwards}) == 10


def test_startup_direct_serving_no_recompile_and_original_model_retained(monkeypatch):
    state = fixture()
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/D/cache")
    assert state.owner.profile_run() == 140
    assert state.executor.ready and state.owner.model is state.model
    assert state.executor.calls == [1] * 10 and not state.checkpoints
    state.backend.state = "P"
    assert state.owner._model_forward(input_ids=[1], positions=state.positions) == 140
    assert state.executor.calls == [2] * 10
    assert [g.compile_calls for g in state.executor.groups] == [1] * 10
    assert state.checkpoints == [(i, i == 9) for i in range(10)]
    assert state.seen[0] == state.seen[40] == (0, None)
    assert state.owner.model is state.model and state.stack == []
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/D/cache"


def test_group_yield_restores_config_and_cancel_unwinds_proxy_without_retirement():
    state = fixture()
    state.owner.profile_run()
    interleaver = LayerInterleaver(state.backend)

    def returned(index, final):
        assert not state.stack
        assert interleaver.active.completed_layers == index * 4
        interleaver.checkpoint_group(4, final=final)

    state.executor.checkpoint = returned
    interleaver.add(
        "P",
        lambda: state.owner._model_forward(input_ids=[1], positions=state.positions),
        "P",
    )
    assert interleaver.step("P", 4).completed_layers == 4
    assert state.backend.state == "caller" and state.stack == []
    assert state.owner.model is not state.model and state.executor.active
    with pytest.raises(RuntimeError, match="active"):
        state.executor.close()
    interleaver.close()
    assert state.owner.model is state.model and not state.executor.active
    assert not state.executor.failed
    assert state.executor.calls == [2] + [1] * 9
    state.executor.close()
    state.executor.close()
    assert state.executor.closed and not state.executor.groups


def test_ten_atomic_steps_complete_actual_40_layers_then_norm():
    state = fixture()
    state.owner.profile_run()
    interleaver = LayerInterleaver(state.backend)
    state.executor.checkpoint = lambda index, final: interleaver.checkpoint_group(
        4, final=final
    )
    interleaver.add(
        "P",
        lambda: state.owner._model_forward(input_ids=[1], positions=state.positions),
        "P",
    )
    for i in range(9):
        assert interleaver.step("P", 4).completed_layers == (i + 1) * 4
        assert not interleaver.tasks["P"].done
    assert interleaver.step("P", 4) is None
    assert interleaver.tasks["P"].result == 140
    assert interleaver.tasks["P"].completed_layers == 40
    assert state.owner.model is state.model


@pytest.mark.parametrize("corrupt", ["handle", "shape", "context"])
def test_serving_rejects_identity_or_shape_or_graph_context_change(corrupt):
    state = fixture()
    state.owner.profile_run()
    state.backend.state = "P"
    if corrupt == "handle":
        state.executor.groups[0].aot_compiled_fn = lambda *a: (0, 0)
    elif corrupt == "shape":
        state.positions.shape = (3, 33)
    else:
        state.ctx.capturing = True
    with pytest.raises((RuntimeError, ValueError)):
        state.owner._model_forward(input_ids=[1], positions=state.positions)
    assert state.executor.failed and state.owner.model is state.model
    assert state.executor.calls == [1] * 10 and not state.checkpoints
    state.executor.close()


def test_four_layer_budget_cannot_report_partial_group_as_single_layer():
    state = fixture()
    state.owner.profile_run()
    interleaver = LayerInterleaver(state.backend)
    state.executor.checkpoint = lambda index, final: interleaver.checkpoint_group(
        4, final=final
    )
    interleaver.add(
        "P",
        lambda: state.owner._model_forward(input_ids=[1], positions=state.positions),
        "P",
    )
    with pytest.raises(RuntimeError, match="exceeds"):
        interleaver.step("P", 1)
    assert interleaver.tasks["P"].completed_layers == 0
    assert state.owner.model is state.model
