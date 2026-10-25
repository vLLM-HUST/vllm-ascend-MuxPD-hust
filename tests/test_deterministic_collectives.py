"""Opt-in environment/profile contracts without importing accelerator code."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_muxpd import deployment
from vllm_muxpd.config import MuxPDConfig
from vllm_muxpd.determinism import (
    POLICY_KEY,
    launch_environment,
    requested_environment,
    startup_readback,
    validate_profile_environment,
)


@pytest.mark.parametrize("flag", [0, 1, "true", None, [], {}])
def test_non_boolean_requirements_are_rejected(flag):
    with pytest.raises(ValueError, match="boolean"):
        MuxPDConfig(require_hccl_deterministic=flag)
    with pytest.raises(ValueError, match="boolean"):
        deployment.configuration(deterministic_collectives=flag)
    with pytest.raises(ValueError, match="boolean"):
        validate_profile_environment({POLICY_KEY: flag})


def test_requirement_cannot_be_ignored_by_a_serial_native_worker():
    with pytest.raises(ValueError, match="requires execution='decoupled'"):
        MuxPDConfig(require_hccl_deterministic=True)
    assert (
        MuxPDConfig(
            execution="decoupled", require_hccl_deterministic=True
        ).require_hccl_deterministic
        is True
    )


def test_default_configuration_and_environment_are_unchanged(monkeypatch):
    monkeypatch.setenv("HCCL_DETERMINISTIC", "strict")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    before = dict(os.environ)
    assert (
        POLICY_KEY
        not in deployment.configuration()["launch_options"][
            "batch_admission_policy_config"
        ]
    )
    assert requested_environment(False) == {}
    with launch_environment(False):
        assert dict(os.environ) == before
    assert validate_profile_environment({}) is False
    assert dict(os.environ) == before
    assert (
        "HCCL_DETERMINISTIC" not in deployment.manifest()["activation"]["environment"]
    )


@pytest.mark.parametrize("original", [None, "true"])
def test_explicit_launch_sets_true_and_restores_environment(monkeypatch, original):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("HCCL_DETERMINISTIC", raising=False)
    if original is not None:
        monkeypatch.setenv("HCCL_DETERMINISTIC", original)
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="child failure"):
        with launch_environment(True):
            assert os.environ["HCCL_DETERMINISTIC"] == "true"
            assert validate_profile_environment({POLICY_KEY: True}) is True
            raise RuntimeError("child failure")
    assert dict(os.environ) == before


@pytest.mark.parametrize("conflict", ["false", "strict", "1", "TRUE", ""])
def test_explicit_conflicts_are_not_silently_overwritten(monkeypatch, conflict):
    monkeypatch.setenv("HCCL_DETERMINISTIC", conflict)
    with pytest.raises(ValueError, match="conflicts"):
        with launch_environment(True):
            pytest.fail("must not launch")
    assert os.environ["HCCL_DETERMINISTIC"] == conflict


def test_batch_invariance_is_a_separate_profile(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.delenv("HCCL_DETERMINISTIC", raising=False)
    with pytest.raises(ValueError, match="separate profile"):
        with launch_environment(True):
            pytest.fail("must not launch")
    assert "HCCL_DETERMINISTIC" not in os.environ


def test_plan_binds_requirement_and_displays_requested_environment_without_mutation(
    monkeypatch, capsys
):
    monkeypatch.delenv("HCCL_DETERMINISTIC", raising=False)
    assert (
        deployment.main(["plan", "--model", "/model", "--deterministic-collectives"])
        == 0
    )
    value = json.loads(capsys.readouterr().out)
    assert value["requested_environment"] == {"HCCL_DETERMINISTIC": "true"}
    assert (
        value["configuration"]["launch_options"]["batch_admission_policy_config"][
            POLICY_KEY
        ]
        is True
    )
    assert "experimental" in value["deterministic_profile_status"]
    assert "HCCL_DETERMINISTIC" not in os.environ


def test_actual_worker_method_checks_requirement_before_native_init(monkeypatch):
    # Compile the actual entry method against a CPU sentinel base. Its failure
    # path must not reach the native initializer or any accelerator names.
    tree = ast.parse((Path(deployment.__file__).parent / "worker.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MuxPDNPUWorker"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "init_device"
    )
    events = []

    class NativeReached(Exception):
        pass

    class Base:
        def init_device(self):
            events.append("native-init")
            raise NativeReached

    generated = ast.Module(
        body=[
            ast.ClassDef(
                name="Probe",
                bases=[ast.Name(id="Base", ctx=ast.Load())],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = {
        "Base": Base,
        "validate_profile_environment": validate_profile_environment,
        "managed_runtime_preflight": lambda: events.append("preflight"),
    }
    exec(
        compile(ast.fix_missing_locations(generated), "actual-worker-entry", "exec"),
        namespace,
    )
    worker = namespace["Probe"]()
    policy = {POLICY_KEY: True}
    worker.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(batch_admission_policy_config=policy)
    )
    monkeypatch.delenv("HCCL_DETERMINISTIC", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    with pytest.raises(ValueError, match="before worker distributed"):
        worker.init_device()
    assert events == ["preflight"]
    events.clear()
    policy[POLICY_KEY] = "true"
    with pytest.raises(ValueError, match="boolean"):
        worker.init_device()
    assert events == ["preflight"]
    events.clear()
    policy[POLICY_KEY] = True
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    with pytest.raises(NativeReached):
        worker.init_device()
    assert events == ["preflight", "native-init"]


@pytest.mark.parametrize("rank", range(4))
def test_small_per_rank_readback_is_scoped_and_has_no_global_side_effects(
    monkeypatch, rank
):
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setenv("VLLM_ECPA_LAUNCH_ID", "launch-fixture")
    before = dict(os.environ)
    result = startup_readback(
        {POLICY_KEY: True, "overlap_communication": "isolated"},
        rank=rank,
        worker="qualified.worker",
        torch_deterministic=False,
        torch_warn_only=False,
        prefill_groups=[{"name": "tp:muxpd_prefill", "ranks": [0, 1, 2, 3]}],
    )
    assert result["rank"] == rank and result["experimental"] is True
    assert result["manager_launch_id"] == "launch-fixture"
    assert result["observed_environment"]["HCCL_DETERMINISTIC"] == "true"
    assert "not proof" in result["scope"]
    assert len(json.dumps(result)) < 2000
    assert dict(os.environ) == before


def test_actual_manager_projects_requirement_and_preserves_explicit_environment(
    monkeypatch,
):
    cli = pytest.importorskip("vllm_hust_ext.cli")
    from vllm_hust_ext import config
    from vllm_hust_ext.core import LifecycleState
    from vllm_hust_ext.manifest import load_manifest
    from vllm_hust_ext.providers.vllm import VllmProvider

    manifest_path = (
        Path(deployment.__file__).parent / "manifests/vllm-hust-extension-v0.3.json"
    )
    manifest = load_manifest(manifest_path)
    settings = deployment.configuration(deterministic_collectives=True)
    provider_plan = VllmProvider().plan(manifest, settings, enabled=True)
    projected = provider_plan.generated_config
    assert (
        projected["vllm_json_options"]["--batch-admission-policy-config"][POLICY_KEY]
        is True
    )
    assert projected["user_config"] == settings
    # Current Manager has no conditional environment declaration. Avoid a static
    # default that overwrites inherited caller settings on ordinary launches.
    assert "HCCL_DETERMINISTIC" not in projected["environment"]
    with pytest.raises(ValueError, match="must be declared by the manifest"):
        VllmProvider().plan(
            manifest,
            {**settings, "environment": {"HCCL_DETERMINISTIC": "true"}},
            enabled=True,
        )
    saved = config.UserConfig(
        {
            deployment.EXTENSION_ID: config.ExtensionConfig(
                enabled=True, configuration=settings
            )
        }
    )
    bundle = SimpleNamespace(
        bundle_id=deployment.EXTENSION_ID,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    monkeypatch.setattr(config, "load_config", lambda: saved)
    monkeypatch.setattr(cli, "load_config", lambda: saved)
    monkeypatch.setattr(cli, "discover_bundles", lambda *_: (bundle,))
    monkeypatch.setattr(
        cli,
        "status_for",
        lambda *_: SimpleNamespace(
            states=(LifecycleState.COMPATIBLE,), evidence=("CPU fixture",)
        ),
    )
    monkeypatch.setattr(cli, "plan_for", lambda *_: provider_plan)
    monkeypatch.setattr(
        cli,
        "launch_environment",
        lambda *a, **k: {"VLLM_ECPA_LAUNCH_ID": "fixture-no-write"},
    )
    monkeypatch.setattr(
        deployment, "doctor", lambda: {"software_profile_matches": True}
    )
    monkeypatch.setattr(deployment.metadata, "version", lambda _: "0.3.1.dev1")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("HCCL_DETERMINISTIC", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    seen = []

    def capture(command, *, env, **kwargs):
        policy = json.loads(
            command[command.index("--batch-admission-policy-config") + 1]
        )
        assert policy[POLICY_KEY] is True
        assert env["HCCL_DETERMINISTIC"] == "true"
        seen.append(command)
        return 0

    monkeypatch.setattr(cli, "supervise", capture)
    args = SimpleNamespace(
        model="/model",
        tensor_parallel_size=4,
        host="127.0.0.1",
        port=8123,
        served_model_name="qwen",
        trace_path=None,
        dry_run=False,
        experimental_prefill_groups=False,
        deterministic_collectives=True,
    )
    assert deployment.serve(args) == 0
    assert len(seen) == 1 and "HCCL_DETERMINISTIC" not in os.environ
    args.deterministic_collectives = False
    with pytest.raises(ValueError, match="saved Manager configuration differs"):
        deployment.serve(args)
    assert len(seen) == 1
