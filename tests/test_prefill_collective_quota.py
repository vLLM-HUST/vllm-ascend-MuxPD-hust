"""Exercise prefix quota admission before worker communication installation."""

import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
from test_prefill_collective_config import config, policy

from vllm_muxpd.config import MuxPDConfig, validate_engine_config
from vllm_muxpd.prefill_collective_layout import install_on_worker


@pytest.mark.parametrize("quota", [4, 40])
def test_prefix_accepts_declared_fixed_quota(quota):
    assert MuxPDConfig.from_mapping({**policy(), "layers_per_slice": quota})
    cfg = config()
    cfg.scheduler_config.batch_admission_policy_config["layers_per_slice"] = quota
    validate_engine_config(cfg)


@pytest.mark.parametrize("quota", [1, 8, 39, 41, 80])
def test_prefix_rejects_unqualified_fixed_quota(quota):
    with pytest.raises(ValueError, match="prefix layout"):
        MuxPDConfig.from_mapping({**policy(), "layers_per_slice": quota})


class CommunicationBoundary(Exception):
    """The actual installer has accepted the worker, before device operations."""


def stub_worker(monkeypatch, configured, actual):
    class Worker:
        pass

    class Runner:
        pass

    class SharedRunner:
        pass

    decoupled = ModuleType("vllm_muxpd.decoupled_worker")
    decoupled.DecoupledMuxPDNPUWorker = Worker
    base = ModuleType("vllm_muxpd.worker")
    base.NPUModelRunner, base._SharedCacheRunner = Runner, SharedRunner
    distributed = ModuleType("vllm.distributed")

    def group_boundary():
        raise CommunicationBoundary

    distributed.get_tp_group = distributed.get_ep_group = group_boundary
    monkeypatch.setitem(sys.modules, decoupled.__name__, decoupled)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    worker = Worker()
    worker.vllm_config = config()
    worker.vllm_config.scheduler_config.batch_admission_policy_config[
        "layers_per_slice"
    ] = configured
    worker.model_runner, worker.decode_runner = Runner(), SharedRunner()
    worker.execution_mode = "decoupled"
    worker._stream_mode = "overlap"
    worker._overlap_communication = "isolated"
    worker._decode_graph_enabled = True
    worker._prefill_groups_enabled = False
    worker._layers_per_slice = actual
    worker._decode_graph_controller = NS(
        snapshot=lambda: {
            "captured": True,
            "failed": False,
            "closed": False,
            "compile_full_prefill": False,
        }
    )
    return worker


@pytest.mark.parametrize("quota", [4, 40])
def test_installer_accepts_worker_matching_declared_quota(monkeypatch, quota):
    worker = stub_worker(monkeypatch, quota, quota)
    with pytest.raises(CommunicationBoundary):
        install_on_worker(worker)


@pytest.mark.parametrize("configured,actual", [(4, 40), (40, 4), (40, 8)])
def test_installer_rejects_worker_policy_quota_drift(monkeypatch, configured, actual):
    worker = stub_worker(monkeypatch, configured, actual)
    with pytest.raises(RuntimeError, match="raw isolated"):
        install_on_worker(worker)
