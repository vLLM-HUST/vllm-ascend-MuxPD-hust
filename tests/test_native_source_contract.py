"""Optional CPU execution of exact current-source contracts, without its binaries.

Set MUXPD_CURRENT_CORE_SOURCE to an exported 62c941c native source tree.
AST extraction keeps the original method bodies and schemas; it does not claim
that importing or executing the full October 25 torch/NPU stack succeeded.
"""

import ast
import os
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from vllm_muxpd.native_compat import runner_contract, scheduler_contract


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            methods = names[node.name]
            if methods is not None:
                node.bases = []
                node.body = [
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name in methods
                ]
            classes.append(node)
    assert len(classes) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *classes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def current_source(monkeypatch):
    source = os.environ.get("MUXPD_CURRENT_CORE_SOURCE")
    if not source:
        pytest.skip("set MUXPD_CURRENT_CORE_SOURCE for current-source CPU contracts")
    root = Path(source)
    module = ModuleType("muxpd_current_source_fixture")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = vars(module)
    namespace.update(dataclass=dataclass, field=field, cached_property=cached_property)
    # The event module itself only uses the standard library.
    exec(
        compile((root / "vllm/v1/events.py").read_text(), "native-events", "exec"),
        namespace,
    )
    extract(
        root / "vllm/v1/core/sched/output.py",
        {
            "SchedulerOutput": None,
            "CachedRequestData": None,
            "NewRequestData": None,
            "ScheduledEncoderInputStats": None,
        },
        namespace,
    )
    extract(root / "vllm/v1/outputs.py", {"ModelRunnerOutput": None}, namespace)
    extract(
        root / "vllm/v1/core/sched/scheduler.py",
        {
            "Scheduler": {
                "_free_request_blocks",
                "_request_blocks_can_be_freed",
                "_request_kv_block_count",
                "_free_blocks",
                "_free_request",
                "_preempt_request",
                "_handle_blocked_waiting_request",
            }
        },
        namespace,
    )
    module.contract = scheduler_contract(
        module.Scheduler,
        module.SchedulerOutput,
        module.ModelRunnerOutput,
        module.CachedRequestData,
        module.NewRequestData,
    )
    return module


def test_current_exact_source_signatures_and_schemas(current_source):
    assert current_source.contract.name == "hust-oct25"


def test_current_exact_free_finish_and_deferred_event_order(current_source):
    native = current_source
    observed = []
    native.EventBus.register_sink(NS(emit=observed.append))
    freed = []

    class Scheduler(native.Scheduler):
        def _free_request_blocks(self, request):
            if request.request_id == "held":
                return native.contract.held_result([1, 2])
            return super()._free_request_blocks(request)

    scheduler = Scheduler()
    scheduler.defer_block_free = False
    scheduler.aux_output_connector = scheduler.ec_connector = None
    scheduler._inflight_prefills = set()
    scheduler._set_kv_fetch_stage = lambda *args: None
    scheduler._connector_finished = lambda *args: (False, None)
    scheduler.encoder_cache_manager = NS(free=lambda request: None)
    scheduler.kv_cache_manager = NS(
        free=lambda request: freed.append(request.request_id),
        get_block_ids=lambda rid: ([1, 2],),
    )
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None

    class Request:
        session_id = "session"
        num_prompt_tokens = 32
        num_output_tokens = 1

        def is_finished(self):
            return True

        def get_finished_reason(self):
            return "abort"

        def __init__(self, rid):
            self.request_id = rid

    requests = {rid: Request(rid) for rid in ("immediate", "held")}
    scheduler.requests = requests.copy()
    scheduler._free_request(requests["immediate"])
    assert freed == ["immediate"]
    assert [type(event).__name__ for event in observed] == [
        "RequestKvReclaimed",
        "RequestFinished",
    ]
    assert not observed[-1].kv_reclaim_deferred
    scheduler._free_request(requests["held"])
    assert observed[-1].kv_reclaim_deferred and observed[-1].kv_blocks == 2
    assert not scheduler.requests and len(observed) == 3
    native.contract.emit_reclaimed("held", "session", 2)
    assert observed[-1].path == "deferred" and observed[-1].request_id == "held"
    assert observed[-1].ts_monotonic_ns >= observed[-2].ts_monotonic_ns


def test_current_signature_drift_fails_closed(current_source, monkeypatch):
    native = current_source

    def changed(self, request, timestamp, drop_stale_output=True):
        raise AssertionError("must not call")

    monkeypatch.setattr(native.Scheduler, "_preempt_request", changed)
    with pytest.raises(RuntimeError, match="preemption default"):
        scheduler_contract(
            native.Scheduler,
            native.SchedulerOutput,
            native.ModelRunnerOutput,
            native.CachedRequestData,
            native.NewRequestData,
        )


def test_current_ascend_exact_cache_signatures(current_source):
    source = os.environ.get("MUXPD_CURRENT_ASCEND_SOURCE")
    if not source:
        pytest.skip("set MUXPD_CURRENT_ASCEND_SOURCE for Ascend contract inspection")
    extract(
        Path(source) / "vllm_ascend/worker/model_runner_v1.py",
        {"NPUModelRunner": {"initialize_kv_cache", "initialize_kv_cache_tensors"}},
        vars(current_source),
    )
    assert runner_contract(current_source.NPUModelRunner) == "hust-oct25"
