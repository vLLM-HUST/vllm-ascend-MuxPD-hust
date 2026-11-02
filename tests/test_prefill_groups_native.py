"""Pinned source contracts only; no torch.compile call or device allocation."""

import importlib
import os
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def native():
    if os.environ.get("VLLM_VERSION") != "0.25.1":
        pytest.skip("source runtime/env.sh for pinned compiler class contracts")
    import torch
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.config import CompilationMode

    module = importlib.import_module("vllm_muxpd.prefill_group_models")
    return NS(
        torch=torch,
        wrapper=TorchCompileWithNoGuardsWrapper,
        mode=CompilationMode,
        classes=module.GROUP_CLASSES,
    )


def test_real_decorated_classes_have_isolated_code_and_share_only_loaded_layers(
    native, monkeypatch
):
    # The native wrapper constructor would create a lazy torch.compile object.
    # Replace it before constructing any group; this fixture must never compile.
    seen = []
    monkeypatch.setattr(native.wrapper, "__init__", lambda self, **kw: seen.append(kw))
    config = NS(compilation_config=NS(mode=native.mode.VLLM_COMPILE))
    layers = [native.torch.nn.Identity() for _ in range(40)]
    groups = [
        cls(layers=layers[i * 4 : i * 4 + 4], vllm_config=config)
        for i, cls in enumerate(native.classes)
    ]
    assert len(groups) == len(seen) == 10
    assert len({type(group).forward.__code__ for group in groups}) == 10
    assert len({type(group).forward.__qualname__ for group in groups}) == 10
    assert all(
        tuple(group.layers) == tuple(layers[i * 4 : i * 4 + 4])
        for i, group in enumerate(groups)
    )
    assert all(isinstance(group, native.wrapper) for group in groups)
    assert all(not list(group.parameters()) for group in groups)


def test_real_worker_progress_snapshot_uses_groups_including_cancel_before_first_return(
    native,
):
    cls = importlib.import_module("vllm_muxpd.decoupled_worker").DecoupledMuxPDNPUWorker
    worker = cls.__new__(cls)
    worker._prefill_groups_enabled = True
    worker._decoder_layer_count = 40
    worker._prefill_full_forward_return = False
    for layers, done in [(0, False), (4, False), (36, False), (40, True)]:
        worker._prefill_task = NS(
            tasks={"prefill": NS(completed_layers=layers, done=done)}
        )
        worker._snapshot_prefill_progress()
        assert worker._iteration_prefill_layers_completed == layers
        assert worker._iteration_prefill_layers_total == 40
        assert worker._iteration_prefill_progress_source == "group_forward_return"


@pytest.mark.parametrize(
    "progress,completed,cancelled,paused",
    [
        (4, set(), set(), False),
        (40, {"P"}, set(), False),
        (4, set(), {"P"}, False),
        (0, set(), {"P"}, False),
        (0, set(), set(), True),
    ],
)
def test_native_scheduler_group_receipts_preserve_ack_and_cancel_lifetime(
    native,
    progress,
    completed,
    cancelled,
    paused,
):
    cls = importlib.import_module(
        "vllm_muxpd.decoupled_scheduler"
    ).DecoupledMuxPDScheduler
    scheduler = cls.__new__(cls)
    scheduler._prefill_cohort = NS(pending_ids={"P"})
    scheduler._decoder_layer_count = 40
    scheduler.requests = {} if cancelled else {"P": object()}
    scheduler._cancelled_prefill_ids = lambda: cancelled
    scheduler.muxpd_policy = NS(
        config=NS(compile_prefill_groups=True, compile_full_prefill=False)
    )
    step = NS(
        cohort_id=1,
        expected_layers=progress,
        decode_output=NS(num_scheduled_tokens={"D": 1}),
        output=NS(
            muxpd_finish_prefill=bool(completed),
            muxpd_pause_prefill=paused,
            muxpd_cancel_prefill_request_ids=cancelled,
        ),
    )
    order = ["D"] + sorted(completed)
    output = NS(
        muxpd_prefill_cohort_id=1,
        muxpd_completed_prefill_ids=completed,
        muxpd_cancelled_prefill_ids=cancelled,
        req_ids=order,
        req_id_to_index={rid: i for i, rid in enumerate(order)},
        sampled_token_ids=[[1] for _ in order],
        muxpd_prefill_layers_completed=progress,
        muxpd_prefill_layers_total=40,
        muxpd_prefill_progress_source="group_forward_return",
    )
    assert scheduler._validate_receipt(step, output) == (
        completed,
        cancelled,
        progress,
        "group_forward_return",
    )
    output.muxpd_prefill_progress_source = "layer_hooks"
    with pytest.raises(RuntimeError, match="source/config"):
        scheduler._validate_receipt(step, output)
