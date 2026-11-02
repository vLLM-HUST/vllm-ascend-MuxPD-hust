"""Complete a synchronous MuxPD transaction in one worker RPC."""

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_muxpd.decoupled_worker import DecoupledMuxPDNPUWorker


class AtomicDecoupledMuxPDNPUWorker(DecoupledMuxPDNPUWorker):
    """Return samples and lifecycle receipts directly from ``execute_model``.

    MuxPD rejects structured outputs, so no scheduler-produced grammar is
    needed between execution and sampling. The synchronous EngineCore already
    accepts a complete ModelRunnerOutput here; returning it avoids its second
    blocking RPC for every iteration. All lane ordering, sampling, cancellation
    acknowledgements and output validation remain in the existing worker.

    This opt-in worker consumes sampling in ``execute_model``. A subsequent
    explicit ``sample_tokens`` call is a duplicate and is rejected by the base
    worker, just as a second sampling call was before.
    """

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        super().execute_model(scheduler_output)
        return self.sample_tokens(None)
