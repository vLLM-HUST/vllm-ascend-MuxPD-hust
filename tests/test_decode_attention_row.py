"""CPU tests of the opt-in 'row' and 'bsh' decode attention layouts (stub NPU).

They check the arguments of every kernel call (one native TND M=1 call per real
row, native causal mask/sparse/window arguments), padding handling, graph
capture bookkeeping and replay updates. Numerical equality with native M=1 is a
device property and is not testable here.
"""

import sys
import types
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


from vllm_muxpd import decode_attention as da  # noqa: E402

NH, HS, MASK = 2, 4, "causal-mask"


class _Kernel:
    def __init__(self, log):
        self.log = log

    def __call__(self, **kw):
        self.log.append(("eager", kw))
        q = kw["query"]
        return torch.full((q.shape[0], NH, HS), float(len(self.log))), None

    def out(self, **kw):
        self.log.append(("out", kw))


class _Event:
    def __init__(self):
        self.records = 0

    def wait(self, stream):
        pass

    def reset(self, stream):
        pass

    def record(self, stream):
        self.records += 1


def _install(monkeypatch, layout):
    log, groups, updates = [], [], []
    kernel = _Kernel(log)
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu_fused_infer_attention_score = kernel
    torch_npu._npu_fused_infer_attention_score_get_max_workspace = lambda **kw: (
        "workspace",
        kw["actual_seq_lengths_kv"],
    )
    torch_npu.npu = SimpleNamespace(current_stream=lambda: "stream")

    @contextmanager
    def stream_ctx(s):
        yield

    fake_npu = SimpleNamespace(
        ExternalEvent=_Event,
        graph_task_group_begin=lambda s: groups.append("begin"),
        graph_task_group_end=lambda s: f"handle{len(groups)}",
        graph_task_update_begin=lambda s, h: updates.append(h),
        graph_task_update_end=lambda s: None,
        stream=stream_ctx,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    graph = SimpleNamespace(
        attn_params=defaultdict(list),
        handles=defaultdict(list),
        events=defaultdict(list),
    )
    extra = SimpleNamespace(capturing=False, is_draft_model=False)
    state = SimpleNamespace(DecodeOnly="decode", ChunkedPrefill="chunked")
    attn_type = SimpleNamespace(DECODER="decoder")
    native = []

    class Impl:
        def forward(self, layer, *a, **k):
            return self.forward_fused_infer_attention(*a, **k)

        def forward_fused_infer_attention(self, q, k, v, md, out, kv_cache=None):
            native.append("fia")
            return out

        def full_graph_fia(self, q, k, v, md, out, layer=None):
            native.append("graph")
            return out, 0

        @staticmethod
        def update_graph_params(*a, **k):
            native.append("update")

        def _get_fia_params(self, key, value, md, kv_cache=None):
            return key, value, 128, md.block_tables, md.seq_lens_list

    av1 = types.ModuleType("vllm_ascend.attention.attention_v1")
    av1.AscendAttentionBackendImpl = Impl
    av1.AscendAttentionState = state
    av1._EXTRA_CTX = extra
    av1.AttentionType = attn_type
    acl = types.ModuleType("vllm_ascend.compilation.acl_graph")
    acl.get_graph_params = lambda: graph
    backend = types.ModuleType("vllm.v1.attention.backend")
    backend.AttentionType = attn_type
    for name, mod in {
        "torch_npu": torch_npu,
        "vllm_ascend.attention.attention_v1": av1,
        "vllm_ascend.compilation.acl_graph": acl,
        "vllm.v1.attention.backend": backend,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
        # `from pkg import mod` reads the attribute of an already imported parent.
        parent_name, _, attr = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, attr, mod, raising=False)
    monkeypatch.setattr(da, "_INSTALLED", {})
    monkeypatch.setattr(da, "STATS", dict.fromkeys(da.STATS, 0))
    assert da.install(layout) is True

    impl = Impl()
    impl.sinks = impl.sliding_window = None
    impl.enable_c8_quant = False
    impl.attn_type = "decoder"
    impl.num_heads, impl.num_kv_heads, impl.head_size, impl.scale = NH, 1, HS, 0.5
    impl.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    impl._muxpd_layer_name = "layers.3.attn"
    return SimpleNamespace(
        impl=impl, log=log, graph=graph, extra=extra, native=native, updates=updates
    )


@pytest.fixture
def env(monkeypatch):
    return _install(monkeypatch, "row")


@pytest.fixture
def bsh_env(monkeypatch):
    return _install(monkeypatch, "bsh")


def _md(qcum, seq_lens, rows=4, state="decode"):
    bt = torch.arange(rows * 3, dtype=torch.int32).view(rows, 3) + 1
    return SimpleNamespace(
        attn_state=state,
        causal=True,
        actual_seq_lengths_q=qcum,
        seq_lens_list=seq_lens,
        block_tables=bt,
        attn_mask=MASK,
    )


def _assert_native_row_args(kw, kv_len):
    assert kw["input_layout"] == "TND"
    assert kw["actual_seq_lengths"] == [1]
    assert kw["actual_seq_lengths_kv"] == [kv_len]
    assert kw["sparse_mode"] == 3
    assert kw["atten_mask"] == MASK
    assert kw["pre_tokens"] == kw["next_tokens"] == da.SWA_INT_MAX
    assert kw["query"].shape[0] == 1 and kw["block_table"].shape[0] == 1


def test_eager_one_native_call_per_real_row_and_zero_padding(env):
    md = _md([1, 2, 3, 6], [17, 9, 30])  # 3 real rows + TND padding request
    q = torch.randn(6, NH, HS)
    out = torch.full((6, NH, HS), 7.0)
    env.impl.forward_fused_infer_attention(q, "k", "v", md, out)
    assert [c for c, _ in env.log] == ["eager"] * 3
    for r, (_, kw) in enumerate(env.log):
        _assert_native_row_args(kw, [17, 9, 30][r])
        assert torch.equal(kw["query"], q[r : r + 1])
        assert torch.equal(kw["block_table"], md.block_tables[r : r + 1])
        assert torch.all(out[r] == r + 1)
    assert torch.all(out[3:] == 0)
    assert env.native == []


def test_non_decode_and_capture_eager_calls_pass_through(env):
    out = torch.zeros(5, NH, HS)
    md = _md([5], [5], state="chunked")
    env.impl.forward_fused_infer_attention(torch.randn(5, NH, HS), "k", "v", md, out)
    md2 = _md([2, 4], [4, 1])  # multi-token queries: not a decode batch
    env.impl.forward_fused_infer_attention(torch.randn(3, NH, HS), "k", "v", md2, out)
    env.extra.capturing = True
    env.impl.forward_fused_infer_attention(
        torch.randn(2, NH, HS), "k", "v", _md([1, 2], [3, 4]), out
    )
    assert env.native == ["fia", "fia", "fia"] and env.log == []


def test_capture_then_replay_updates_rows_and_skips_unchanged_padding(env):
    q = torch.randn(4, NH, HS)
    out = torch.zeros(4, NH, HS)
    _, total = env.impl.full_graph_fia(q, "k", "v", _md([1, 2, 4], [11, 12]), out)
    assert total == 4
    params = env.graph.attn_params[4]
    assert [p.row for p in params] == [0, 1, 2, 3]
    assert [p.padded for p in params] == [False, False, True, True]
    captured = [kw for c, kw in env.log if c == "out"]
    assert [kw["actual_seq_lengths_kv"] for kw in captured] == [[11], [12], [1], [1]]
    assert all(torch.all(kw["block_table"] == 0) for kw in captured[2:])
    assert len({id(kw["workspace"]) for kw in captured}) == 1
    assert len(env.graph.handles[4]) == len(env.graph.events[4]) == 4

    env.log.clear()
    ctx = SimpleNamespace(
        attn_metadata={"layers.3.attn": _md([1, 2, 3, 4], [5, 6, 7, 2])}
    )
    type(env.impl).update_graph_params("upd", ctx, 4, None)
    updated = [kw for c, kw in env.log if c == "out"]
    # Four real rows: every task leaves padding and is updated.
    assert [kw["actual_seq_lengths_kv"] for kw in updated] == [[5], [6], [7], [2]]
    for r, kw in enumerate(updated):
        _assert_native_row_args(kw, [5, 6, 7, 2][r])
        assert torch.equal(
            kw["block_table"],
            ctx.attn_metadata["layers.3.attn"].block_tables[r : r + 1],
        )
    assert [p.padded for p in params] == [False] * 4
    assert all(e.records == 1 for e in env.graph.events[4])

    for step, skipped in ((1, 0), (2, 3)):
        env.log.clear()
        ctx = SimpleNamespace(attn_metadata={"layers.3.attn": _md([1, 4], [8 + step])})
        type(env.impl).update_graph_params("upd", ctx, 4, None)
        updated = [kw for c, kw in env.log if c == "out"]
        if step == 1:
            # Row 0 real; rows 1-3 return to padding and are rebound once.
            assert [kw["actual_seq_lengths_kv"] for kw in updated] == [
                [9],
                [1],
                [1],
                [1],
            ]
            assert all(torch.all(kw["block_table"] == 0) for kw in updated[1:])
        else:
            # Unchanged padding tasks are skipped but their events still fire.
            assert [kw["actual_seq_lengths_kv"] for kw in updated] == [[10]]
        assert da.STATS["row_updates_skipped"] == skipped
        assert all(e.records == 1 + step for e in env.graph.events[4])


def test_uniform_graph_padding_rows_are_padding_in_eager_and_replay(env):
    # Shipped FULL_DECODE_ONLY preset: padding = extra [1..B] rows with KV length 0.
    q = torch.randn(4, NH, HS)
    out = torch.full((4, NH, HS), 7.0)
    env.impl.forward_fused_infer_attention(
        q, "k", "v", _md([1, 2, 3, 4], [5, 6, 0, 0]), out
    )
    assert [kw["actual_seq_lengths_kv"] for _, kw in env.log] == [[5], [6]]
    assert torch.all(out[2:] == 0)

    env.log.clear()
    env.impl.full_graph_fia(q, "k", "v", _md([1, 2, 3, 4], [5, 6, 0, 0]), out)
    params = env.graph.attn_params[4]
    assert [p.padded for p in params] == [False, False, True, True]
    captured = [kw for c, kw in env.log if c == "out"]
    assert [kw["actual_seq_lengths_kv"] for kw in captured] == [[5], [6], [1], [1]]

    env.log.clear()
    ctx = SimpleNamespace(
        attn_metadata={"layers.3.attn": _md([1, 2, 3, 4], [8, 0, 0, 0])}
    )
    type(env.impl).update_graph_params("upd", ctx, 4, None)
    updated = [kw for c, kw in env.log if c == "out"]
    # Row 0 real, row 1 newly padding (rebound once), rows 2-3 unchanged padding.
    assert [kw["actual_seq_lengths_kv"] for kw in updated] == [[8], [1]]
    assert da.STATS["row_updates_skipped"] == 2
    assert all(e.records == 1 for e in env.graph.events[4])


def test_replay_rejects_non_decode_or_wrong_size_metadata(env):
    out = torch.zeros(2, NH, HS)
    env.impl.full_graph_fia(torch.randn(2, NH, HS), "k", "v", _md([1, 2], [3, 4]), out)
    for qcum in ([1, 2, 3], [2]):
        ctx = SimpleNamespace(attn_metadata={"layers.3.attn": _md(qcum, [3, 4, 5])})
        with pytest.raises(RuntimeError, match="not a decode batch"):
            type(env.impl).update_graph_params("upd", ctx, 2, None)


def test_sizes_without_row_params_use_native_update(env):
    type(env.impl).update_graph_params(
        "upd", SimpleNamespace(attn_metadata={}), 8, None
    )
    assert env.native == ["update"]


def test_ineligible_attention_keeps_native_graph_path(env):
    env.impl.sliding_window = 128
    out = torch.zeros(2, NH, HS)
    env.impl.full_graph_fia(torch.randn(2, NH, HS), "k", "v", _md([1, 2], [3, 4]), out)
    assert env.native == ["graph"] and env.graph.attn_params == {}


def test_second_install_with_other_layout_is_rejected(env):
    assert da.install("row") is True
    with pytest.raises(RuntimeError, match="installed differently"):
        da.install("bsh")


def _assert_bsh_args(kw, kv_lens):
    total = len(kv_lens)
    assert kw["input_layout"] == "BSH"
    assert kw["sparse_mode"] == 0
    assert kw["actual_seq_lengths_kv"] == kv_lens
    assert kw["query"].shape == (total, 1, NH * HS)
    assert kw["block_table"].shape[0] == total
    assert kw["num_heads"] == NH and kw["num_key_value_heads"] == 1
    assert kw["scale"] == 0.5 and kw["block_size"] == 128
    assert "atten_mask" not in kw and "actual_seq_lengths" not in kw


def test_bsh_eager_is_one_call_with_padded_rows_bound_to_block_zero(bsh_env):
    md = _md([1, 2, 3, 6], [17, 9, 30])  # 3 real rows + TND padding request
    out = torch.full((6, NH, HS), 7.0)
    bsh_env.impl.forward_fused_infer_attention(
        torch.randn(6, NH, HS), "k", "v", md, out
    )
    assert [c for c, _ in bsh_env.log] == ["eager"]
    kw = bsh_env.log[0][1]
    _assert_bsh_args(kw, [17, 9, 30, 1, 1, 1])
    assert torch.equal(kw["block_table"][:3], md.block_tables[:3])
    assert torch.all(kw["block_table"][3:] == 0)
    assert torch.all(out == 1.0)
    assert bsh_env.native == [] and da.STATS["eager_calls"] == 1


def test_bsh_eager_short_block_table_is_zero_padded(bsh_env):
    md = _md([1, 2, 5], [4, 6], rows=2)
    out = torch.zeros(5, NH, HS)
    bsh_env.impl.forward_fused_infer_attention(
        torch.randn(5, NH, HS), "k", "v", md, out
    )
    kw = bsh_env.log[0][1]
    _assert_bsh_args(kw, [4, 6, 1, 1, 1])
    assert torch.equal(kw["block_table"][:2], md.block_tables)
    assert torch.all(kw["block_table"][2:] == 0)


def test_bsh_non_decode_and_capture_eager_calls_pass_through(bsh_env):
    out = torch.zeros(5, NH, HS)
    bsh_env.impl.forward_fused_infer_attention(
        torch.randn(5, NH, HS), "k", "v", _md([5], [5], state="chunked"), out
    )
    bsh_env.extra.capturing = True
    bsh_env.impl.forward_fused_infer_attention(
        torch.randn(2, NH, HS), "k", "v", _md([1, 2], [3, 4]), out
    )
    assert bsh_env.native == ["fia", "fia"] and bsh_env.log == []


def test_bsh_capture_then_replay_rebinds_metadata(bsh_env):
    out = torch.zeros(4, NH, HS)
    _, total = bsh_env.impl.full_graph_fia(
        torch.randn(4, NH, HS), "k", "v", _md([1, 2, 4], [11, 12]), out
    )
    assert total == 4
    (param,) = bsh_env.graph.attn_params[4]
    assert isinstance(param, da.BSHGraphParam) and param.batch == 4
    assert param.layer_name == "layers.3.attn"
    (captured,) = [kw for c, kw in bsh_env.log if c == "out"]
    _assert_bsh_args(captured, [11, 12, 1, 1])
    assert captured["workspace"] == ("workspace", [64] * 4)
    assert len(bsh_env.graph.handles[4]) == len(bsh_env.graph.events[4]) == 1

    bsh_env.log.clear()
    md = _md([1, 2, 3, 4], [5, 6, 7, 2])
    type(bsh_env.impl).update_graph_params(
        "upd", SimpleNamespace(attn_metadata={"layers.3.attn": md}), 4, None
    )
    (updated,) = [kw for c, kw in bsh_env.log if c == "out"]
    _assert_bsh_args(updated, [5, 6, 7, 2])
    assert torch.equal(updated["block_table"], md.block_tables[:4])
    assert updated["workspace"] is param.workspace
    assert updated["out"] == [param.output, param.softmax_lse]
    assert bsh_env.graph.events[4][0].records == 1
    assert da.STATS["graph_updates"] == 1


def test_bsh_replay_rejects_wrong_metadata_and_short_tables(bsh_env):
    bsh_env.impl.full_graph_fia(
        torch.randn(2, NH, HS), "k", "v", _md([1, 2], [3, 4]), torch.zeros(2, NH, HS)
    )
    for qcum in ([1, 2, 3], [2]):
        ctx = SimpleNamespace(attn_metadata={"layers.3.attn": _md(qcum, [3, 4, 5])})
        with pytest.raises(RuntimeError, match="not a decode batch"):
            type(bsh_env.impl).update_graph_params("upd", ctx, 2, None)
    ctx = SimpleNamespace(attn_metadata={"layers.3.attn": _md([1, 2], [3, 4], rows=1)})
    with pytest.raises(RuntimeError, match="too few rows"):
        type(bsh_env.impl).update_graph_params("upd", ctx, 2, None)


def test_bsh_capture_rejects_short_block_table(bsh_env):
    with pytest.raises(RuntimeError, match="too few rows"):
        bsh_env.impl.full_graph_fia(
            torch.randn(4, NH, HS),
            "k",
            "v",
            _md([1, 2, 4], [3, 4], rows=2),
            torch.zeros(4, NH, HS),
        )


def test_bsh_sizes_without_bsh_params_and_ineligible_calls_stay_native(bsh_env):
    type(bsh_env.impl).update_graph_params(
        "upd", SimpleNamespace(attn_metadata={}), 8, None
    )
    bsh_env.impl.sliding_window = 128
    bsh_env.impl.full_graph_fia(
        torch.randn(2, NH, HS), "k", "v", _md([1, 2], [3, 4]), torch.zeros(2, NH, HS)
    )
    assert bsh_env.native == ["update", "graph"]
    assert bsh_env.graph.attn_params == {}


@pytest.mark.parametrize("layout", ["row", "bsh"])
def test_layout_and_native_params_in_one_size_are_rejected(monkeypatch, layout):
    env = _install(monkeypatch, layout)
    env.impl.full_graph_fia(
        torch.randn(2, NH, HS), "k", "v", _md([1, 2], [3, 4]), torch.zeros(2, NH, HS)
    )
    env.graph.attn_params[2].append(SimpleNamespace(layer_name="native"))
    env.graph.handles[2].append("native-handle")
    env.graph.events[2].append(_Event())
    ctx = SimpleNamespace(attn_metadata={"layers.3.attn": _md([1, 2], [3, 4])})
    with pytest.raises(RuntimeError, match="mixes"):
        type(env.impl).update_graph_params("upd", ctx, 2, None)
