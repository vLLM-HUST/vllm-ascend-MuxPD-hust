from types import SimpleNamespace as NS

import numpy as np
import pytest

from vllm_muxpd.prefill_collective_layout import (
    ATTRIBUTE,
    HISTORY_SIZE,
    Layout,
    TorchRuntime,
    shutdown_on_worker,
)


class Tensor:
    dtype = "torch.bfloat16"
    device = "npu:0"

    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    @property
    def shape(self):
        return self.array.shape

    def new_zeros(self, shape):
        return Tensor(np.zeros(shape, np.float32))

    def __getitem__(self, key):
        result = Tensor.__new__(Tensor)
        result.array = self.array[key]
        return result

    def copy_(self, other):
        self.array[...] = other.array
        return self

    def is_contiguous(self):
        return self.array.flags.c_contiguous


class Runtime:
    transform = TorchRuntime.transform

    def __init__(self):
        self.current = NS(
            skip_compiled=True,
            capturing=False,
            in_profile_run=False,
            cudagraph_runtime_mode=NS(name="NONE"),
        )
        self.current_coroutine = object()

    def context(self):
        return self.current

    def coroutine(self):
        return self.current_coroutine


class Communicator:
    def __init__(self):
        self.device_group = object()
        self.inputs = []

    def all_reduce(self, tensor):
        self.inputs.append(tensor.array.copy())
        tensor.array *= 4
        return tensor


class Group:
    def __init__(self, name):
        self.unique_name, self.ranks, self.rank_in_group = name, [0, 1, 2, 3], 0
        self.world_size, self.use_custom_op_call = 4, True
        self.device_communicator = Communicator()
        self.device_group = self.device_communicator.device_group

    def _all_reduce_out_place(self, input_):
        return self.device_communicator.all_reduce(input_)


class Worker:
    def __init__(self, runtime, group):
        self.rank, self.runtime, self.group = 0, runtime, group
        self._execution_backend = NS(active_lane=None)
        self._prefill_task = self._prefill_cohort_id = self._interleaver = None
        self._prefill_compiled = self._awaiting_sample = False
        self._pending_lanes = []
        self.model_runner = NS()
        self.decode_runner = NS()
        self.tensor = None
        self.fail_execute = self.fail_clear = False
        self.idle_runners()

    def idle_runners(self):
        for runner in (self.model_runner, self.decode_runner):
            runner.requests = {}
            runner.input_batch = NS(num_reqs=0, req_ids=[None], req_id_to_index={})
            runner.execute_model_state = None

    def execute_model(self, scheduler_output):
        output = scheduler_output
        if self.fail_execute:
            raise RuntimeError("original execute failure")
        if output.muxpd_prefill_request_ids:
            ids = [
                rid
                for rid in output.num_scheduled_tokens
                if rid in output.muxpd_prefill_request_ids
            ]
            qsl, packed, positions, requests = [0], [], [], {}
            for i, rid in enumerate(ids):
                tokens = [i + 11] * output.num_scheduled_tokens[rid]
                qsl.append(qsl[-1] + len(tokens))
                packed += tokens
                positions += list(range(len(tokens)))
                requests[rid] = NS(prompt_token_ids=tokens, output_token_ids=[])
            self.model_runner.requests = requests
            self.model_runner.input_batch = NS(
                num_reqs=len(ids),
                req_ids=ids,
                num_computed_tokens_cpu=[0] * len(ids),
                req_id_to_index={rid: i for i, rid in enumerate(ids)},
            )
            self.model_runner.query_start_loc = NS(cpu=qsl)
            self.model_runner.input_ids = NS(cpu=packed)
            self.model_runner._positions_np_buf = positions
            task = NS(
                name="prefill", coroutine=self.runtime.current_coroutine, done=False
            )
            self._prefill_task = NS(tasks={"prefill": task}, active=None)
            self._prefill_cohort_id = output.muxpd_prefill_cohort_id
        if output.run_prefill:
            self.run_p()
        return "original-execute-result"

    def run_p(self):
        self._interleaver = self._prefill_task
        self._interleaver.active = self._interleaver.tasks["prefill"]
        self._execution_backend.active_lane = "prefill"
        n = self.model_runner.query_start_loc.cpu[-1]
        self.tensor = Tensor(np.arange(n * 4).reshape(n, 4) + 1)
        try:
            result = self.group._all_reduce_out_place(self.tensor)
            assert result is self.tensor
        finally:
            self._execution_backend.active_lane = None
            self._interleaver.active = None
            self._interleaver = None

    def _clear_prefill(self):
        if self.fail_clear:
            raise RuntimeError("original clear failure")
        self._prefill_task = self._prefill_cohort_id = None
        return "original-clear-result"


def fixture(mode="prefix"):
    runtime, group, sibling = Runtime(), Group("tp:0"), Group("tp:muxpd_prefill:0")
    worker = Worker(runtime, group)
    worker._prefill_groups = [sibling]
    router = NS(
        _backend=worker._execution_backend,
        _failed=False,
        calls={"caller": 0, "decode": 0, "prefill": 0},
        _installed=[],
    )
    original, prefill = (
        group.device_communicator.all_reduce,
        sibling.device_communicator.all_reduce,
    )

    def routed(*args, **kwargs):
        lane = router._backend.active_lane
        result = (prefill if lane == "prefill" else original)(*args, **kwargs)
        router.calls[lane or "caller"] += 1
        return result

    routed._muxpd_collective_router = router
    group.device_communicator.all_reduce = routed
    router._installed = [(group.device_communicator, "all_reduce", original, routed)]
    worker._collective_router = router
    layout = Layout(worker, mode, group, runtime, 4)
    setattr(worker, ATTRIBUTE, layout)
    layout.install()
    return worker, layout, runtime, group, sibling


def output(cohort=1, d=2, new=True, run=True):
    counts = {f"D{i}": 1 for i in range(d)}
    if new:
        counts.update({"P_A": 3, "P_B": 5})
    return NS(
        muxpd_execution="decoupled",
        muxpd_prefill_request_ids={"P_A", "P_B"} if new else set(),
        muxpd_pending_prefill_ids={"P_A", "P_B"},
        muxpd_cancel_prefill_request_ids=set(),
        muxpd_prefill_cohort_id=cohort,
        num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()),
        scheduled_spec_decode_tokens={},
        scheduled_cached_reqs=NS(req_ids=[], resumed_req_ids=set()),
        scheduled_new_reqs=[
            NS(req_id=rid, num_computed_tokens=0, prompt_token_ids=[1] * count)
            for rid, count in counts.items()
            if rid.startswith("P_")
        ],
        preempted_req_ids=set(),
        run_prefill=run,
    )


def finish(worker):
    worker._prefill_task.tasks["prefill"].done = True
    assert worker._clear_prefill() == "original-clear-result"
    worker.idle_runners()


@pytest.mark.parametrize("d", [0, 1, 5])
def test_actual_payload_exactly_once_original_alias_and_d0_no_allocation(d):
    worker, layout, runtime, _, sibling = fixture()
    if not d:
        runtime.transform = lambda *args: pytest.fail("D0 must not allocate or copy")
    assert worker.execute_model(output(d=d)) == "original-execute-result"
    assert len(sibling.device_communicator.inputs) == 1
    payload = sibling.device_communicator.inputs[0]
    assert payload.shape == (8 + d, 4)
    assert not np.any(payload[:d])
    assert np.array_equal(payload[d:], np.arange(32).reshape(8, 4) + 1)
    assert np.array_equal(worker.tensor.array, 4 * (np.arange(32).reshape(8, 4) + 1))
    record = layout.active
    if d:
        assert record["calls"][0]["existing_collective_calls"] == 1
        assert record["calls"][0]["return_original_input"]
    else:
        assert not record["calls"] and record["zero_d_bypass_calls"] == 1
    finish(worker)
    assert layout.cohorts[-1]["clear_count"] == 1


def test_frozen_actual_scheduled_decode_counts_survive_suspension():
    worker, layout, _, _, sibling = fixture()
    request = output(d=2)
    # A waiting/deferred/active but unscheduled D does not contribute.
    request.unscheduled_active_request_ids = {"D_deferred", "D_waiting"}
    worker.execute_model(request)
    worker.execute_model(output(new=False, d=5, run=False))
    worker.execute_model(output(new=False, d=0, run=True))
    assert layout.active["admission_d_counts"] == {"D0": 1, "D1": 1}
    assert [a.shape for a in sibling.device_communicator.inputs] == [(10, 4)] * 2
    finish(worker)
    assert layout.active is None


@pytest.mark.parametrize("lane", [None, "decode"])
def test_caller_and_decode_bypass_are_exact(lane):
    worker, layout, runtime, group, sibling = fixture()
    worker.execute_model(output(run=False))
    worker._execution_backend.active_lane = lane
    runtime.transform = lambda *args: pytest.fail("bypass transform")
    tensor = Tensor(np.ones((1, 4)))
    assert group._all_reduce_out_place(tensor) is tensor
    assert np.all(tensor.array == 4) and not sibling.device_communicator.inputs
    assert layout.context is None


@pytest.mark.parametrize(
    "fault", ["computed", "packed", "positions", "partial", "output"]
)
def test_actual_p_rows_must_be_complete_fresh_prompts(fault):
    worker, layout, _, _, sibling = fixture()
    worker.execute_model(output(run=False))
    runner = worker.model_runner
    if fault == "computed":
        runner.input_batch.num_computed_tokens_cpu[0] = 1
    elif fault == "packed":
        runner.input_ids.cpu[0] = 999
    elif fault == "positions":
        runner._positions_np_buf[0] = 4
    elif fault == "output":
        runner.requests["P_A"].output_token_ids.append(1)
    else:
        runner.requests["P_A"].prompt_token_ids.append(7)
    with pytest.raises(RuntimeError):
        worker.run_p()
    assert not sibling.device_communicator.inputs and layout.primary_failure
    with pytest.raises(RuntimeError, match="poisoned"):
        worker.execute_model(output(new=False, run=False))


@pytest.mark.parametrize(
    "fault", ["resumed", "cached", "preempted", "partial", "decode"]
)
def test_admission_rejects_invalid_scheduled_maps_before_ownership(fault):
    worker, layout, _, _, sibling = fixture()
    request = output()
    if fault == "resumed":
        request.scheduled_cached_reqs.resumed_req_ids = {"P_A"}
    elif fault == "cached":
        request.scheduled_cached_reqs.req_ids = ["P_A"]
    elif fault == "preempted":
        request.preempted_req_ids = {"P_A"}
    elif fault == "partial":
        request.scheduled_new_reqs[0].num_computed_tokens = 1
    else:
        request.num_scheduled_tokens["D0"] = 2
        request.total_num_scheduled_tokens += 1
    with pytest.raises(RuntimeError):
        worker.execute_model(request)
    assert worker._prefill_task is None and layout.active is None
    assert not sibling.device_communicator.inputs


def test_new_cohort_cannot_replace_suspended_owner():
    worker, layout, _, _, _ = fixture()
    worker.execute_model(output())
    with pytest.raises(RuntimeError, match="replace"):
        worker.execute_model(output(cohort=2))
    assert layout.active["cohort_id"] == 1


def test_unfinished_and_failed_clear_keep_ownership_then_allow_safe_retry():
    worker, layout, _, _, _ = fixture()
    worker.execute_model(output())
    with pytest.raises(RuntimeError, match="unfinished"):
        worker._clear_prefill()
    assert layout.active is not None
    worker._prefill_task.tasks["prefill"].done = True
    worker.fail_clear = True
    with pytest.raises(RuntimeError, match="original clear"):
        worker._clear_prefill()
    assert layout.active is not None
    worker.fail_clear = False
    finish(worker)
    assert layout.active is None and layout.primary_failure
    assert layout.uninstall()["all_originals_restored"]


def test_failed_execute_without_worker_ownership_rolls_back_and_poisons():
    worker, layout, _, _, _ = fixture()
    worker.fail_execute = True
    with pytest.raises(RuntimeError, match="original execute"):
        worker.execute_model(output())
    assert layout.active is None
    assert layout.cohorts[-1]["status"] == "admission_failed"
    assert layout.primary_failure
    with pytest.raises(RuntimeError, match="poisoned"):
        worker.execute_model(output(cohort=2))
    assert layout.uninstall()["all_originals_restored"]


def test_hundreds_of_cohorts_keep_bounded_history_and_no_lifetime_failure():
    worker, layout, _, _, sibling = fixture()
    for cohort in range(1, 201):
        worker.execute_model(output(cohort=cohort, d=cohort % 3))
        assert layout.active["cohort_id"] == cohort
        finish(worker)
        sibling.device_communicator.inputs.clear()
    assert len(layout.cohorts) == HISTORY_SIZE
    assert [row["cohort_id"] for row in layout.cohorts] == list(range(185, 201))
    snapshot = layout.snapshot()
    assert snapshot["stats"]["admitted"] == snapshot["stats"]["closed"] == 200
    assert not snapshot["history_is_complete"] and not snapshot["primary_failure"]
    assert (
        snapshot["active"] is None and len(snapshot["recent_cohorts"]) == HISTORY_SIZE
    )
    with pytest.raises(RuntimeError, match="monotonically"):
        worker.execute_model(output(cohort=1))


def test_foreign_slot_never_overwritten_and_restore_retry_keeps_original_slots():
    worker, layout, _, group, _ = fixture()
    original = group._all_reduce_out_place

    def foreign(value):
        return value

    group._all_reduce_out_place = foreign
    with pytest.raises(RuntimeError, match="externally"):
        layout.uninstall()
    assert layout.restored_count == 0 and group._all_reduce_out_place is foreign
    group._all_reduce_out_place = original
    assert layout.uninstall()["all_originals_restored"]
    assert layout.restored_count == 3
    assert "execute_model" not in vars(worker)
    assert "_all_reduce_out_place" not in vars(group)
    assert layout.uninstall()["restored_count"] == 3


def test_foreign_observer_rejected_before_forward():
    worker, layout, _, group, sibling = fixture()
    inner = group._all_reduce_out_place
    group._all_reduce_out_place = lambda value: inner(value)
    with pytest.raises(RuntimeError, match="ownership"):
        worker.execute_model(output())
    assert not sibling.device_communicator.inputs and layout.primary_failure


def test_partial_install_rolls_back_acquired_slots(monkeypatch):
    worker, prior, runtime, group, _ = fixture()
    prior.uninstall()
    layout = Layout(worker, "prefix", group, runtime, 4)
    original_patch = layout._patch

    def fail_last(owner, name, factory):
        if name == "_all_reduce_out_place":
            raise RuntimeError("install failure")
        return original_patch(owner, name, factory)

    monkeypatch.setattr(layout, "_patch", fail_last)
    with pytest.raises(RuntimeError, match="install failure"):
        layout.install()
    assert "execute_model" not in vars(worker) and "_clear_prefill" not in vars(worker)
    assert layout.primary_failure and layout._restored()
    assert layout.uninstall()["all_originals_restored"]


def test_transform_failure_retains_payload_and_does_not_retry_collective():
    worker, layout, runtime, _, sibling = fixture()
    original = sibling.device_communicator.all_reduce
    # Route captures its callable, so replace the captured method's behavior
    # by a Tensor copy-back failure after the single existing reduction.
    original_copy = Tensor.copy_
    copies = []

    def copy(tensor, other):
        copies.append(tensor)
        if len(copies) == 2:
            raise RuntimeError("copyback failure")
        return original_copy(tensor, other)

    from unittest.mock import patch

    with patch.object(Tensor, "copy_", copy):
        with pytest.raises(RuntimeError, match="copyback failure"):
            worker.execute_model(output())
    assert len(sibling.device_communicator.inputs) == 1
    assert len(runtime.failed_buffers) == 3
    assert layout.active is not None and layout.primary_failure
    with pytest.raises(RuntimeError, match="poisoned"):
        worker.run_p()
    assert len(sibling.device_communicator.inputs) == 1
    assert original.__self__ is sibling.device_communicator
    finish(worker)


def test_shutdown_sync_failure_keeps_hooks_and_buffers_for_retry():
    worker, layout, runtime, group, _ = fixture()
    runtime.failed_buffers = (object(),)
    fail = [True]
    events = []

    def sync():
        events.append("sync")
        if fail[0]:
            raise RuntimeError("sync failure")

    worker._streams = {"prefill": NS(synchronize=sync)}
    with pytest.raises(RuntimeError, match="sync failure"):
        shutdown_on_worker(worker)
    assert layout.snapshot()["all_hooks_intact"] and runtime.failed_buffers
    fail[0] = False
    assert shutdown_on_worker(worker)["all_originals_restored"]
    assert not runtime.failed_buffers and "_all_reduce_out_place" not in vars(group)


def test_startup_collective_after_install_is_detected_before_first_admission():
    worker, layout, _, group, _ = fixture()
    group._all_reduce_out_place(Tensor(np.ones((1, 4))))
    with pytest.raises(RuntimeError, match="after layout installation"):
        worker.execute_model(output())
    assert layout.active is None and layout.primary_failure


def test_failed_partial_install_rollback_retains_owned_slot_for_cleanup_retry(
    monkeypatch,
):
    worker, prior, runtime, group, _ = fixture()
    prior.uninstall()
    layout = Layout(worker, "prefix", group, runtime, 4)
    setattr(worker, ATTRIBUTE, layout)
    patch = layout._patch
    failing = [True]

    def delattr_with_failure(owner, name):
        if name == "_clear_prefill" and failing[0]:
            raise RuntimeError("restore failure")
        object.__delattr__(owner, name)

    def patch_with_failure(owner, name, factory):
        if name == "_all_reduce_out_place":
            raise RuntimeError("primary install failure")
        patch(owner, name, factory)

    monkeypatch.setattr(Worker, "__delattr__", delattr_with_failure, raising=False)
    monkeypatch.setattr(layout, "_patch", patch_with_failure)
    with pytest.raises(RuntimeError, match="primary install failure"):
        layout.install()
    assert "execute_model" not in vars(worker)
    assert "_clear_prefill" in vars(worker)
    assert layout.primary_failure.endswith("primary install failure")
    assert any("restore failure" in error for error in layout.errors)
    failing[0] = False
    assert layout.uninstall()["all_originals_restored"]
    assert layout.restored_count == 2


def test_mid_uninstall_failure_restores_remaining_owned_slots_on_retry(monkeypatch):
    worker, layout, _, group, _ = fixture()
    failing = [True]

    def delattr_with_failure(owner, name):
        if name == "_clear_prefill" and failing[0]:
            raise RuntimeError("restore failure")
        object.__delattr__(owner, name)

    monkeypatch.setattr(Worker, "__delattr__", delattr_with_failure, raising=False)
    with pytest.raises(RuntimeError, match="restore failure"):
        layout.uninstall()
    assert "_all_reduce_out_place" not in vars(group)
    assert "execute_model" in vars(worker) and "_clear_prefill" in vars(worker)
    failing[0] = False
    assert layout.uninstall()["all_originals_restored"]
    assert layout.restored_count == 3


def test_terminal_receipt_binds_rank_module_and_restoration_identity():
    worker, layout, _, _, _ = fixture()
    before = layout.snapshot()
    row = layout.uninstall()
    assert row["schema"] == "muxpd-prefill-collective-prefix-uninstall-v1"
    assert row["rank"] == worker.rank and row["mode"] == "prefix"
    assert row["module_sha256"] == before["module_sha256"]
    assert row["installed_hook_count"] == row["restored_count"] == 3
    assert row["uninstalled"] and row["all_originals_restored"]
    assert row["active_cohort"] is None and not row["in_collective"]
    assert row["primary_failure"] is None
