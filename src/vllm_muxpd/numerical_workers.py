"""Worker selectors for explicit numerical diagnostics, never normal activation."""

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.worker import NPUWorker

from vllm_muxpd.decoupled_worker import DecoupledMuxPDNPUWorker
from vllm_muxpd.numerical_diagnostics import (
    attach_observer,
    compiler_constants_snapshot,
)


class NumericalWarmupMixin:
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        compiler_constants_snapshot(self)
        return result


class NumericalNativeNPUWorker(NumericalWarmupMixin, NPUWorker):
    def load_model(self):
        super().load_model()
        if get_ascend_config().enable_reduce_sample:
            raise ValueError("Full-vocabulary diagnostic requires reduce_sample off")
        attach_observer(self, self.model_runner, "native")


class NumericalMuxPDNPUWorker(NumericalWarmupMixin, DecoupledMuxPDNPUWorker):
    def load_model(self):
        super().load_model()
        if get_ascend_config().enable_reduce_sample:
            raise ValueError("Full-vocabulary diagnostic requires reduce_sample off")
        attach_observer(self, self.model_runner, "prefill")
        attach_observer(self, self.decode_runner, "decode")
