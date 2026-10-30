"""Execute actual worker lifecycle methods on CPU sentinel dependencies."""

import ast
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest
from test_prefill_collective_layout import fixture, output

from vllm_muxpd import prefill_collective_layout as layout_module


def worker_method(filename, class_name, method, base=object, **namespace):
    tree = ast.parse((Path(layout_module.__file__).parent / filename).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    function = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method
    )
    generated = ast.Module(
        body=[
            ast.ClassDef(
                name="Probe",
                bases=[ast.Name(id="Base", ctx=ast.Load())],
                keywords=[],
                body=[function],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    values = {
        "Base": base,
        "Any": Any,
        "json": json,
        "torch": NS(inference_mode=lambda: lambda fn: fn),
        "__package__": "vllm_muxpd",
        **namespace,
    }
    exec(
        compile(
            ast.fix_missing_locations(generated), "actual-worker-lifecycle", "exec"
        ),
        values,
    )
    return values["Probe"]


@pytest.mark.parametrize("enabled", [False, True])
def test_install_only_after_native_warmup_capture_exit(monkeypatch, enabled):
    events = []

    class Base:
        def compile_or_warm_up_model(self):
            events.append("native-warmup-and-seed-reset")
            return "native-result"

    @contextmanager
    def capture():
        events.append("capture-enter")
        yield
        events.append("capture-exit")

    def install(worker):
        assert events[-1] == "capture-exit"
        events.append("install")
        return {}

    monkeypatch.setattr(layout_module, "install_on_worker", install)
    cls = worker_method(
        "worker.py",
        "MuxPDNPUWorker",
        "compile_or_warm_up_model",
        Base,
        logger=NS(info=lambda *args: None),
    )
    worker = cls()
    worker.rank = 0
    worker._decode_graph_controller = NS(
        redirect_owner_capture=capture, snapshot=lambda: {}
    )
    worker._interleaver = worker._prefill_task = None
    worker._pending_lanes = []
    worker._awaiting_sample = False
    worker._prefill_collective_layout_mode = "prefix" if enabled else "off"
    assert worker.compile_or_warm_up_model() == "native-result"
    assert events == [
        "capture-enter",
        "native-warmup-and-seed-reset",
        "capture-exit",
    ] + (["install"] if enabled else [])


def test_failed_native_warmup_never_installs(monkeypatch):
    class Base:
        def compile_or_warm_up_model(self):
            raise RuntimeError("warmup failure")

    @contextmanager
    def capture():
        yield

    monkeypatch.setattr(
        layout_module,
        "install_on_worker",
        lambda _: pytest.fail("install after warmup failure"),
    )
    cls = worker_method("worker.py", "MuxPDNPUWorker", "compile_or_warm_up_model", Base)
    worker = cls()
    worker._decode_graph_controller = NS(redirect_owner_capture=capture)
    worker._interleaver = worker._prefill_task = None
    worker._pending_lanes = []
    worker._awaiting_sample = False
    worker._prefill_collective_layout_mode = "prefix"
    with pytest.raises(RuntimeError, match="warmup failure"):
        worker.compile_or_warm_up_model()


@pytest.mark.parametrize("fail_sync", [False, True])
def test_actual_abort_retires_only_after_stream_sync_and_preserves_owner(fail_sync):
    worker, layout, _, _, _ = fixture()
    worker.execute_model(output())
    events = []
    task = worker._prefill_task
    task.close = lambda: (
        events.append("close"),
        setattr(task.tasks["prefill"], "done", True),
    )

    def sync():
        events.append("sync")
        if fail_sync:
            raise RuntimeError("sync failure")

    worker._streams = {"prefill": NS(synchronize=sync)}
    worker._snapshot_prefill_progress = lambda: events.append("snapshot")
    worker._prefill_order = ["P_A", "P_B"]
    worker._cancelled_ack = set()
    worker._retire_finished = lambda _: events.append("retire")
    cls = worker_method(
        "decoupled_worker.py", "DecoupledMuxPDNPUWorker", "_abort_prefill"
    )
    if fail_sync:
        with pytest.raises(RuntimeError, match="sync failure"):
            cls._abort_prefill(worker)
        assert layout.active is not None and worker._prefill_task is task
        assert not worker._cancelled_ack and events == ["snapshot", "close", "sync"]
    else:
        cls._abort_prefill(worker)
        assert layout.active is None and worker._prefill_task is None
        assert worker._cancelled_ack == {"P_A", "P_B"}
        assert events == ["snapshot", "close", "sync", "retire"]


@pytest.mark.parametrize("fail", [False, True])
def test_actual_shutdown_uninstalls_before_graph_router_group_destruction(
    monkeypatch, fail
):
    events = []

    class Base:
        def shutdown(self):
            events.append("native-shutdown")

    def uninstall(_worker):
        events.append("layout-drain-and-uninstall")
        if fail:
            raise RuntimeError("layout cleanup failure")
        return {}

    monkeypatch.setattr(layout_module, "shutdown_on_worker", uninstall)
    cls = worker_method(
        "worker.py",
        "MuxPDNPUWorker",
        "shutdown",
        Base,
        logger=NS(info=lambda *args: None),
    )
    worker = cls()
    worker.rank = 0
    worker._prefill_collective_layout = object()
    worker._interleaver = None
    worker._hook_handles = []
    worker._decode_graph_controller = NS(
        snapshot=lambda: {}, close=lambda: events.append("graph-close")
    )
    worker.decode_runner = NS()
    worker._streams = {}
    worker._release_overlap_communication = lambda: events.append("router-and-groups")
    if fail:
        with pytest.raises(RuntimeError, match="layout cleanup failure"):
            worker.shutdown()
        assert events == ["layout-drain-and-uninstall"]
        assert worker._decode_graph_controller is not None
        assert not worker._muxpd_shutdown_in_progress
    else:
        worker.shutdown()
        assert events == [
            "layout-drain-and-uninstall",
            "graph-close",
            "router-and-groups",
            "native-shutdown",
        ]
        assert worker._muxpd_shutdown_complete


def test_second_warmup_is_refused_before_native_work():
    class Base:
        def compile_or_warm_up_model(self):
            pytest.fail("native work after prefix startup")

    cls = worker_method("worker.py", "MuxPDNPUWorker", "compile_or_warm_up_model", Base)
    worker = cls()
    worker._decode_graph_controller = object()
    worker._prefill_collective_layout = object()
    with pytest.raises(RuntimeError, match="already finalized"):
        worker.compile_or_warm_up_model()
