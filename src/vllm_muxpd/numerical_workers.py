"""Post-warmup observer extension; production worker selectors stay unchanged."""

from vllm_muxpd.numerical_diagnostics import (
    attach_observer,
    compiler_constants_snapshot,
    observe_prefill_group_return,
)

NATIVE_WORKER = "vllm_ascend.worker.worker.NPUWorker"
MUXPD_WORKER = "vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker"


def install_group_observer(worker):
    executor = getattr(worker, "_prefill_group_executor", None)
    if executor is None:
        return False
    original = executor.checkpoint

    def observed_checkpoint(index, *, final):
        observe_prefill_group_return(worker, index, final=final)
        return original(index, final=final)

    executor.checkpoint = observed_checkpoint
    return True


def install_observers(worker):
    selector = type(worker).__module__ + "." + type(worker).__qualname__
    if selector not in (NATIVE_WORKER, MUXPD_WORKER):
        raise ValueError(f"Unqualified numerical worker selector: {selector}")
    if getattr(worker, "_muxpd_numerics_install_state", None) is not None:
        raise RuntimeError("Numerical observer installation already attempted")
    runners = [(worker.model_runner, "native")]
    if selector == MUXPD_WORKER:
        runners = [(worker.model_runner, "prefill"), (worker.decode_runner, "decode")]
    if (
        any(
            getattr(runner, "execute_model_state", None) is not None
            for runner, _ in runners
        )
        or getattr(worker, "_awaiting_sample", False)
        or getattr(worker, "_prefill_task", None) is not None
    ):
        raise RuntimeError("Numerical observers must be installed before serving")
    worker._muxpd_numerics_install_state = "installing"
    try:
        for runner, lane in runners:
            attach_observer(worker, runner, lane)
        group_observed = install_group_observer(worker)
        compiler_constants_snapshot(worker)
        worker._muxpd_numerics_install_state = "installed"
    except BaseException:
        worker._muxpd_numerics_install_state = "failed"
        raise
    return {
        "installed": True,
        "rank": worker.rank,
        "worker_selector": selector,
        "observer_lanes": [lane for _, lane in runners],
        "group_checkpoint_observed": group_observed,
        "boundary": "collective RPC after LLM construction, before requests",
    }


class NumericalObserverExtension:
    """Only unique RPC names: native extension registration rejects collisions."""

    def muxpd_install_numerical_observers(self):
        from vllm_ascend.ascend_config import get_ascend_config

        if get_ascend_config().enable_reduce_sample:
            raise ValueError("Full-vocabulary diagnostic requires reduce_sample off")
        return install_observers(self)
