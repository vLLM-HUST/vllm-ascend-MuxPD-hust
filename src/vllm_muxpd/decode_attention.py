"""Opt-in batch-invariant decode attention layout (policy ``decode_attention_layout``).

The native Ascend decode path calls ``npu_fused_infer_attention_score`` in TND
layout. Its kernel tiling depends on the number of rows, so one request's
attention output changes with the decode batch size (measured on 910B2 for the
Qwen3.5 TP2 shapes). The BSH layout with S=1 (one batch entry per decode row)
is row invariant for 1..16 rows, eager and captured, at about 3% higher kernel
cost, but its numerics differ from native TND. ``"row"`` runs every decode row
as its own native TND M=1 call (same mask/sparse/window arguments as the native
graph path), so each row is bit-identical to the same request decoded alone;
with ``max_prefill_requests: 1`` the concurrency-16 fixed200 output equalled the
native concurrency-1 output on 200/200 cases. Its graph-mode cost is high (one
host task update per row and layer). ``"tnd"`` (the default) installs nothing.

Scope, fail-closed:
* only ``AscendAttentionState.DecodeOnly`` calls of the base FIA path without
  sinks, sliding window, C8 quantization or non-causal attention; every other
  call is passed through unchanged (prefill and chunked prefill keep TND);
* graph update: a captured size whose attention params are BSH is updated here;
  mixing BSH and TND params in one size raises.
Padding: FULL_DECODE_ONLY graph padding arrives as extra single-token rows with
KV length 0 (native uniform padding), FULL mode as one dummy TND request
holding the padded tokens. ``row`` treats both as padding: no kernel call in
eager mode, a fixed KV-length-1 / block-row-0 task in captured graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LAYOUTS = ("tnd", "bsh", "row")
# Native attention_v1 passes this for both window bounds on the causal FIA graph path.
SWA_INT_MAX = 2147483647
_INSTALLED: dict[str, Any] = {}
STATS = {
    "eager_calls": 0,
    "captured_calls": 0,
    "graph_updates": 0,
    "passthrough": 0,
    "row_updates_skipped": 0,
}


def validate_layout(value: Any) -> str:
    if value not in LAYOUTS:
        raise ValueError(f"MuxPD decode_attention_layout must be one of {LAYOUTS}")
    return value


def decode_rows(actual_seq_lengths_q: list[int]) -> tuple[int, int] | None:
    """(real single-token rows, total query tokens) for a decode layout, else None.

    Accepts [1..m] or [1..m, B] (TND padding request holding B-m tokens).
    """
    q = [int(x) for x in actual_seq_lengths_q]
    if not q:
        return None
    m = 0
    while m < len(q) and q[m] == m + 1:
        m += 1
    if m == 0:
        return None
    if m == len(q):
        return m, m
    if m == len(q) - 1 and q[-1] > m:
        return m, q[-1]
    return None


def padded_kv_lengths(seq_lens: list[int], real: int, total: int) -> list[int]:
    return [int(x) for x in seq_lens[:real]] + [1] * (total - real)


@dataclass
class BSHGraphParam:
    query: Any
    key: Any
    value: Any
    block_table: Any
    block_size: int
    num_kv_heads: int
    num_heads: int
    scale: float
    output: Any
    softmax_lse: Any
    workspace: Any
    layer_name: str
    batch: int


@dataclass
class RowGraphParam:
    """One captured native TND M=1 decode attention call (one row of one layer)."""

    query: Any
    key: Any
    value: Any
    attn_mask: Any
    block_size: int
    num_kv_heads: int
    num_heads: int
    scale: float
    output: Any
    softmax_lse: Any
    workspace: Any
    layer_name: str
    batch: int
    row: int
    padded: bool = False  # last update bound the persistent padding row


def row_plan(
    real: int, total: int, seq_lens: list[int], table_rows: int
) -> list[tuple[int | None, int]]:
    """(block-table row, or None for padding, KV length) for each of `total` rows.

    A single-token row with KV length 0 is native uniform graph padding.
    """
    if table_rows < real:
        raise RuntimeError("MuxPD row decode attention block table has too few rows")
    plan: list[tuple[int | None, int]] = []
    for r in range(total):
        kv_len = int(seq_lens[r]) if r < real and r < len(seq_lens) else 0
        plan.append((r, kv_len) if kv_len > 0 else (None, 1))
    return plan


def install(layout: str) -> bool:
    """Patch the Ascend attention impl class in this process. Idempotent."""
    validate_layout(layout)
    if layout == "tnd":
        return False
    if _INSTALLED:
        if _INSTALLED["layout"] != layout:
            raise RuntimeError(
                "MuxPD decode attention layout already installed differently"
            )
        return True

    import torch
    import torch_npu
    from vllm.v1.attention.backend import AttentionType  # noqa: F401  (import check)
    from vllm_ascend.attention import attention_v1 as av1
    from vllm_ascend.compilation.acl_graph import get_graph_params

    impl = av1.AscendAttentionBackendImpl
    state_cls = av1.AscendAttentionState
    extra = av1._EXTRA_CTX
    orig_forward = impl.forward
    orig_fia = impl.forward_fused_infer_attention
    orig_full_graph_fia = impl.full_graph_fia
    orig_update = impl.update_graph_params
    # One workspace per captured size, shared by all layers (same shapes), like
    # the native graph_params.workspaces cache; per-layer buffers exhaust HBM.
    workspaces: dict[int, Any] = {}

    def eligible(self, attn_metadata) -> tuple[int, int] | None:
        if (
            attn_metadata.attn_state != state_cls.DecodeOnly
            or self.sinks is not None
            or self.sliding_window is not None
            or self.enable_c8_quant
            or not attn_metadata.causal
            or self.attn_type != av1.AttentionType.DECODER
            or extra.is_draft_model
        ):
            return None
        return decode_rows(attn_metadata.actual_seq_lengths_q)

    def bsh_block_table(block_table, real: int, total: int):
        if block_table.shape[0] >= total:
            bt = block_table[:total].clone()
            if total > real:
                bt[real:] = 0
            return bt
        pad = block_table.new_zeros(
            (total - min(real, block_table.shape[0]), block_table.shape[1])
        )
        return torch.cat([block_table[:real], pad], 0)

    def bsh_kwargs(self, query, key, value, block_table, block_size, kv_lens):
        b = query.shape[0]
        return dict(
            query=query.view(b, 1, self.num_heads * self.head_size),
            key=key,
            value=value,
            block_table=block_table,
            input_layout="BSH",
            block_size=block_size,
            actual_seq_lengths_kv=kv_lens,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=0,
        )

    def forward(self, layer, *args, **kwargs):
        if layer is not None:
            self._muxpd_layer_name = layer.layer_name
        return orig_forward(self, layer, *args, **kwargs)

    def forward_fused_infer_attention(
        self, query, key, value, attn_metadata, output, kv_cache=None
    ):
        rows = eligible(self, attn_metadata)
        if rows is None:
            STATS["passthrough"] += 1
            return orig_fia(self, query, key, value, attn_metadata, output, kv_cache)
        if extra.capturing:
            return orig_fia(self, query, key, value, attn_metadata, output, kv_cache)
        real, total = rows
        key, value, block_size, block_table, kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        q = query[:total]
        attn, _ = torch_npu.npu_fused_infer_attention_score(
            **bsh_kwargs(
                self,
                q,
                key,
                value,
                bsh_block_table(block_table, real, total),
                block_size,
                padded_kv_lengths(kv, real, total),
            )
        )
        output[:total] = attn.view(total, self.num_heads, self.head_size)
        STATS["eager_calls"] += 1
        return output

    def full_graph_fia(self, query, key, value, attn_metadata, output, layer=None):
        rows = eligible(self, attn_metadata)
        if rows is None:
            return orig_full_graph_fia(
                self, query, key, value, attn_metadata, output, layer
            )
        real, total = rows
        key, value, block_size, block_table, kv = self._get_fia_params(
            key, value, attn_metadata
        )
        graph_params = get_graph_params()
        q = query[:total]
        if block_table.shape[0] < total:
            raise RuntimeError("MuxPD BSH decode capture block table has too few rows")
        buf = block_table[:total]
        kv_lens = padded_kv_lengths(kv, real, total)
        out = output[:total].view(total, 1, self.num_heads * self.head_size)
        lse = torch.empty(1, dtype=query.dtype, device=query.device)
        workspace = workspaces.get(total)
        if workspace is None:
            max_kv = [self.vllm_config.model_config.max_model_len] * total
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                **bsh_kwargs(self, q, key, value, buf, block_size, max_kv)
            )
            workspaces[total] = workspace
        stream = torch_npu.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[total].append(event)
        layer_name = getattr(self, "_muxpd_layer_name", None)
        if layer_name is None:
            raise RuntimeError(
                "MuxPD BSH decode capture needs the attention layer name"
            )
        graph_params.attn_params[total].append(
            BSHGraphParam(
                q,
                key,
                value,
                buf,
                block_size,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                out,
                lse,
                workspace,
                layer_name,
                total,
            )
        )
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            **bsh_kwargs(self, q, key, value, buf, block_size, kv_lens),
            workspace=workspace,
            out=[out, lse],
        )
        graph_params.handles[total].append(torch.npu.graph_task_group_end(stream))
        STATS["captured_calls"] += 1
        return output[:total].view(total, self.num_heads, self.head_size), total

    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        graph_params = get_graph_params()
        params = graph_params.attn_params.get(num_tokens, []) if graph_params else []
        kinds = {isinstance(p, BSHGraphParam) for p in params}
        if True not in kinds or extra.is_draft_model:
            return orig_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
            )
        if kinds != {True}:
            raise RuntimeError(
                "MuxPD BSH decode graph mixes BSH and TND attention params"
            )
        attn_metadata = forward_context.attn_metadata
        with torch.npu.stream(update_stream):
            for p, handle, event in zip(
                params,
                graph_params.handles[num_tokens],
                graph_params.events[num_tokens],
            ):
                md = attn_metadata[p.layer_name]
                rows = decode_rows(md.actual_seq_lengths_q)
                if rows is None or rows[1] != p.batch:
                    raise RuntimeError(
                        "MuxPD BSH decode replay metadata is not a decode batch"
                    )
                real, total = rows
                if md.block_tables.shape[0] < total:
                    raise RuntimeError(
                        "MuxPD BSH decode replay block table has too few rows"
                    )
                # Rebind to the metadata tensor (written before the forward on the
                # main stream), as native FIA replay does. No cross-stream wait:
                # the replay already queued on the main stream waits on `event`.
                torch.npu.graph_task_update_begin(update_stream, handle)
                torch_npu.npu_fused_infer_attention_score.out(
                    query=p.query.view(total, 1, p.num_heads * p.query.shape[-1]),
                    key=p.key,
                    value=p.value,
                    block_table=md.block_tables[:total],
                    input_layout="BSH",
                    block_size=p.block_size,
                    actual_seq_lengths_kv=padded_kv_lengths(
                        md.seq_lens_list, real, total
                    ),
                    num_key_value_heads=p.num_kv_heads,
                    num_heads=p.num_heads,
                    scale=p.scale,
                    sparse_mode=0,
                    workspace=p.workspace,
                    out=[p.output, p.softmax_lse],
                )
                torch.npu.graph_task_update_end(update_stream)
                event.record(update_stream)
        STATS["graph_updates"] += 1

    # ---- "row": every decode row is its own native TND M=1 call ----
    # Bit-identical to what the same request computes alone (concurrency 1) on
    # the native graph path, whatever the batch composition. Shared, sequential
    # (one stream) workspace, LSE and padding block-table buffers.
    row_shared: dict[Any, Any] = {}

    def row_kwargs(self_or_p, q, key, value, mask, bt, block_size, kv_len):
        return dict(
            query=q,
            key=key,
            value=value,
            atten_mask=mask,
            block_table=bt,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=[1],
            actual_seq_lengths_kv=[kv_len],
            num_key_value_heads=self_or_p.num_kv_heads,
            num_heads=self_or_p.num_heads,
            scale=self_or_p.scale,
            sparse_mode=3,
            pre_tokens=SWA_INT_MAX,
            next_tokens=SWA_INT_MAX,
        )

    def pad_table(block_table):
        k = ("pad", block_table.device, block_table.shape[1], block_table.dtype)
        if k not in row_shared:
            row_shared[k] = block_table.new_zeros((1, block_table.shape[1]))
        return row_shared[k]

    def row_forward(self, query, key, value, attn_metadata, output, rows, kv_cache):
        real, total = rows
        key, value, block_size, block_table, kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        mask = attn_metadata.attn_mask
        for r, (bt_row, kv_len) in enumerate(
            row_plan(real, total, kv, block_table.shape[0])
        ):
            if bt_row is None:
                output[r : r + 1].zero_()
                continue
            attn, _ = torch_npu.npu_fused_infer_attention_score(
                **row_kwargs(
                    self,
                    query[r : r + 1],
                    key,
                    value,
                    mask,
                    block_table[bt_row : bt_row + 1],
                    block_size,
                    kv_len,
                )
            )
            output[r : r + 1] = attn.view(1, self.num_heads, self.head_size)
        STATS["eager_calls"] += 1
        return output

    def row_capture(self, query, key, value, attn_metadata, output, rows):
        real, total = rows
        key, value, block_size, block_table, kv = self._get_fia_params(
            key, value, attn_metadata
        )
        layer_name = getattr(self, "_muxpd_layer_name", None)
        if layer_name is None:
            raise RuntimeError(
                "MuxPD row decode capture needs the attention layer name"
            )
        graph_params = get_graph_params()
        mask = attn_metadata.attn_mask
        pad_bt = pad_table(block_table)
        wk = ("ws", query.device)
        if wk not in row_shared:
            row_shared[wk] = (
                torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    **row_kwargs(
                        self,
                        query[:1],
                        key,
                        value,
                        mask,
                        pad_bt,
                        block_size,
                        self.vllm_config.model_config.max_model_len,
                    )
                )
            )
            row_shared[("lse", query.device)] = torch.empty(
                1, dtype=query.dtype, device=query.device
            )
        workspace = row_shared[wk]
        lse = row_shared[("lse", query.device)]
        stream = torch_npu.npu.current_stream()
        for r, (bt_row, kv_len) in enumerate(
            row_plan(real, total, kv, block_table.shape[0])
        ):
            bt = pad_bt if bt_row is None else block_table[bt_row : bt_row + 1]
            q = query[r : r + 1]
            out = output[r : r + 1]
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[total].append(event)
            graph_params.attn_params[total].append(
                RowGraphParam(
                    q,
                    key,
                    value,
                    mask,
                    block_size,
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    out,
                    lse,
                    workspace,
                    layer_name,
                    total,
                    r,
                    bt_row is None,
                )
            )
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                **row_kwargs(self, q, key, value, mask, bt, block_size, kv_len),
                workspace=workspace,
                out=[out, lse],
            )
            graph_params.handles[total].append(torch.npu.graph_task_group_end(stream))
        STATS["captured_calls"] += 1
        return output[:total], total

    def row_update(params, handles, events, attn_metadata, update_stream):
        with torch.npu.stream(update_stream):
            for p, handle, event in zip(params, handles, events):
                md = attn_metadata[p.layer_name]
                rows = decode_rows(md.actual_seq_lengths_q)
                if rows is None or rows[1] != p.batch:
                    raise RuntimeError(
                        "MuxPD row decode replay metadata is not a decode batch"
                    )
                real, total = rows
                bt_row, kv_len = row_plan(
                    real, total, md.seq_lens_list, md.block_tables.shape[0]
                )[p.row]
                padded = bt_row is None
                bt = (
                    pad_table(md.block_tables)
                    if padded
                    else md.block_tables[bt_row : bt_row + 1]
                )
                if padded and p.padded:
                    # Unchanged padding task: keep its parameters, still release it.
                    STATS["row_updates_skipped"] += 1
                    event.record(update_stream)
                    continue
                torch.npu.graph_task_update_begin(update_stream, handle)
                torch_npu.npu_fused_infer_attention_score.out(
                    **row_kwargs(
                        p,
                        p.query,
                        p.key,
                        p.value,
                        p.attn_mask,
                        bt,
                        p.block_size,
                        kv_len,
                    ),
                    workspace=p.workspace,
                    out=[p.output, p.softmax_lse],
                )
                torch.npu.graph_task_update_end(update_stream)
                event.record(update_stream)
                p.padded = padded

    if layout == "row":

        def forward_fused_infer_attention(  # noqa: F811
            self, query, key, value, attn_metadata, output, kv_cache=None
        ):
            rows = eligible(self, attn_metadata)
            if rows is None:
                STATS["passthrough"] += 1
                return orig_fia(
                    self, query, key, value, attn_metadata, output, kv_cache
                )
            if extra.capturing:
                return orig_fia(
                    self, query, key, value, attn_metadata, output, kv_cache
                )
            return row_forward(
                self, query, key, value, attn_metadata, output, rows, kv_cache
            )

        def full_graph_fia(  # noqa: F811
            self, query, key, value, attn_metadata, output, layer=None
        ):
            rows = eligible(self, attn_metadata)
            if rows is None:
                return orig_full_graph_fia(
                    self, query, key, value, attn_metadata, output, layer
                )
            return row_capture(self, query, key, value, attn_metadata, output, rows)

    param_cls = RowGraphParam if layout == "row" else BSHGraphParam

    def update_dispatch(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        graph_params = get_graph_params()
        params = graph_params.attn_params.get(num_tokens, []) if graph_params else []
        kinds = {isinstance(p, param_cls) for p in params}
        if layout == "bsh" or True not in kinds or extra.is_draft_model:
            return update_graph_params(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
            )
        if kinds != {True}:
            raise RuntimeError(
                "MuxPD row decode graph mixes row and native attention params"
            )
        row_update(
            params,
            graph_params.handles[num_tokens],
            graph_params.events[num_tokens],
            forward_context.attn_metadata,
            update_stream,
        )
        STATS["graph_updates"] += 1

    impl.forward = forward
    impl.forward_fused_infer_attention = forward_fused_infer_attention
    impl.full_graph_fia = full_graph_fia
    impl.update_graph_params = staticmethod(update_dispatch)
    _INSTALLED.update(
        layout=layout,
        impl=impl,
        originals=(orig_forward, orig_fia, orig_full_graph_fia, orig_update),
    )
    return True
