import pytest

from vllm_muxpd import decode_attention as da
from vllm_muxpd.config import MuxPDConfig


@pytest.mark.parametrize(
    "qcum,expected",
    [
        ([1], (1, 1)),
        ([1, 2, 3], (3, 3)),
        ([1, 2, 3, 8], (3, 8)),  # TND padding request holding 5 tokens
        ([1, 3], (1, 3)),
        ([5], None),  # single prefill
        ([1, 2, 5, 6], None),  # mixed
        ([2, 4], None),  # multi-token queries
        ([], None),
    ],
)
def test_decode_rows(qcum, expected):
    assert da.decode_rows(qcum) == expected


def test_padded_kv_lengths_pads_with_one():
    assert da.padded_kv_lengths([9, 8, 77], 2, 4) == [9, 8, 1, 1]
    assert da.padded_kv_lengths([9, 8], 2, 2) == [9, 8]


def test_layout_default_is_native():
    assert MuxPDConfig().decode_attention_layout == "tnd"
    assert da.install("tnd") is False


def test_layout_validation():
    MuxPDConfig.from_mapping(
        {"execution": "decoupled", "decode_attention_layout": "bsh"}
    )
    with pytest.raises(ValueError, match="requires execution='decoupled'"):
        MuxPDConfig.from_mapping({"decode_attention_layout": "bsh"})
    MuxPDConfig.from_mapping(
        {"execution": "decoupled", "decode_attention_layout": "row"}
    )
    with pytest.raises(ValueError, match="requires execution='decoupled'"):
        MuxPDConfig.from_mapping({"decode_attention_layout": "row"})
    with pytest.raises(ValueError, match="must be 'tnd', 'bsh' or 'row'"):
        MuxPDConfig.from_mapping(
            {"execution": "decoupled", "decode_attention_layout": "BSH"}
        )
    with pytest.raises(ValueError):
        da.validate_layout("bnsd")


def test_row_plan_real_rows_then_padding():
    assert da.row_plan(2, 4, [17, 9, 555], 3) == [(0, 17), (1, 9), (None, 1), (None, 1)]


def test_row_plan_native_uniform_padding_rows_have_zero_kv():
    # FULL_DECODE_ONLY pads with single-token rows whose seq_lens are 0.
    assert da.row_plan(4, 4, [17, 9, 0, 0], 4) == [
        (0, 17),
        (1, 9),
        (None, 1),
        (None, 1),
    ]
    assert da.row_plan(3, 3, [5, 6, 7], 3) == [(0, 5), (1, 6), (2, 7)]


def test_row_plan_rejects_short_block_table():
    with pytest.raises(RuntimeError, match="too few rows"):
        da.row_plan(3, 4, [1, 2, 3], 2)


def test_layouts_include_row():
    assert da.validate_layout("row") == "row"
    assert "row" in da.LAYOUTS
