"""Installed deployment recipes must stay inert and preserve paired activation."""

import argparse
import json
import subprocess
import sys
from importlib import metadata
from types import ModuleType, SimpleNamespace

import pytest

from vllm_muxpd import deployment


def test_metadata_and_recipes_do_not_import_native_code():
    source = """
import builtins, json, sys
original_import = builtins.__import__
def deny_native(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'vllm_ascend'}:
        raise AssertionError('native import during static inspection: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = deny_native
from vllm_muxpd import deployment
assert len(deployment.manifest()['components']) == 3
policy = deployment.configuration()['launch_options']['batch_admission_policy_config']
assert policy['execution'] == 'decoupled'
assert '--scheduler-cls' not in deployment.host_command('/model with spaces')
result = deployment.doctor()
assert result['scope'].startswith('software preflight only')
print(json.dumps({'inert': True}))
"""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=35
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"inert": True}


def test_bundle_binds_exact_paired_modules_without_auto_activation():
    bundle = deployment.manifest()
    assert bundle["extension_version"] == deployment.__version__
    assert bundle["activation"]["entry_points"] == []
    assert {c["component_id"] for c in bundle["components"]} == {
        "scheduler",
        "admission",
        "worker",
    }
    claims = {item["resource"]: item for item in bundle["resource_claims"]}
    for role in ("scheduler", "worker"):
        claim = claims[f"vllm.process-carrier.{role}"]
        assert claim["mode"] == "exclusive"
        assert claim["scope"] == "vllm-process-tree"
    targets = {
        f"{item['module']}:{item['object']}" for item in bundle["implementation"]
    }
    assert targets == {c["implementation_ref"] for c in bundle["components"]}


def test_launch_keeps_full_prompt_and_separate_graph_profile():
    command = deployment.host_command("/a model", tensor_parallel_size=4, port=8123)
    assert command[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert command[command.index("--model") + 1] == "/a model"
    assert command[command.index("--port") + 1] == "8123"
    assert "--no-enable-chunked-prefill" in command
    assert "--no-async-scheduling" in command
    assert command[command.index("--mamba-cache-mode") + 1] == "none"
    graph = json.loads(command[command.index("--compilation-config") + 1])
    assert graph["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert graph["cudagraph_capture_sizes"] == [1, 2, 4, 8]
    assert not {"--scheduler-cls", "--worker-cls", "--batch-admission-policy"} & set(
        command
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"model": "m", "port": 0},
        {"model": "m", "tensor_parallel_size": 8},
    ],
)
def test_launch_rejects_unqualified_or_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        deployment.host_command(**kwargs)


def test_configuration_does_not_overwrite_existing_saved_file(tmp_path):
    output = tmp_path / "configuration.json"
    assert deployment.main(["configuration", "--output", str(output)]) == 0
    original = output.read_bytes()
    assert json.loads(original) == deployment.configuration()
    with pytest.raises(SystemExit) as error:
        deployment.main(["configuration", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == original


def test_trace_configuration_rejects_unknown_format_fields():
    with pytest.raises(ValueError):
        deployment.configuration("/tmp/{unexpected}.jsonl")
    assert (
        deployment.configuration("/tmp/{pid}.jsonl")["launch_options"][
            "batch_admission_policy_config"
        ]["trace_path"]
        == "/tmp/{pid}.jsonl"
    )


def test_source_checkout_rejects_wrong_commit_and_tracked_modifications(
    tmp_path, monkeypatch
):
    source = tmp_path / "native"
    source.mkdir()
    package = source / "vllm"
    package.mkdir()
    origin = package / "__init__.py"
    origin.write_text("raise AssertionError('qualification must not execute native')\n")
    subprocess.run(["git", "init", str(source)], capture_output=True, check=True)
    (source / "runtime.py").write_text("original = True\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "fixture",
        ],
        capture_output=True,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    fake = SimpleNamespace(
        version="0.1.dev1",
        read_text=lambda name: json.dumps(
            {"url": source.as_uri(), "dir_info": {"editable": True}}
        ),
    )
    monkeypatch.setattr(metadata, "distribution", lambda _: fake)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.syspath_prepend(str(source))
    assert deployment._source_checkout("vllm", commit)["commit"] == commit
    assert "vllm" not in sys.modules
    shadow = tmp_path / "shadow"
    (shadow / "vllm").mkdir(parents=True)
    (shadow / "vllm/__init__.py").write_text("raise AssertionError('no imports')\n")
    with monkeypatch.context() as shadow_patch:
        shadow_patch.syspath_prepend(str(shadow))
        with pytest.raises(ValueError, match="import origin differs"):
            deployment._source_checkout("vllm", commit)
        assert "vllm" not in sys.modules
    with pytest.raises(ValueError, match="not qualified"):
        deployment._source_checkout("vllm", "0" * 40)
    (source / "runtime.py").write_text("original = False\n")
    with pytest.raises(ValueError, match="tracked sources differ"):
        deployment._source_checkout("vllm", commit)


@pytest.fixture
def runtime_binding(monkeypatch):
    for name in deployment._MANAGER_BINDING_ENV:
        monkeypatch.delenv(name, raising=False)

    def enable():
        for name in deployment._MANAGER_BINDING_ENV:
            monkeypatch.setenv(name, "manager-binding")

    return enable


def test_unmanaged_legacy_skips_qualification(runtime_binding, monkeypatch):
    monkeypatch.setattr(
        deployment, "doctor", lambda: pytest.fail("legacy must not run doctor")
    )
    assert deployment.managed_runtime_preflight() is None


def test_qualification_and_effective_evidence_use_the_same_binding_keys():
    from vllm_muxpd import evidence

    assert deployment._MANAGER_BINDING_ENV == evidence._BINDING_ENV


def test_partial_binding_fails_before_qualification(runtime_binding, monkeypatch):
    monkeypatch.setenv("VLLM_ECPA_LAUNCH_ID", "launch")
    monkeypatch.setattr(deployment, "doctor", lambda: pytest.fail("partial binding"))
    with pytest.raises(RuntimeError, match="incomplete Manager binding"):
        deployment.managed_runtime_preflight()


def test_managed_qualification_is_rechecked_and_fails_closed(
    runtime_binding, monkeypatch
):
    runtime_binding()
    result = {"software_profile_matches": True, "errors": []}
    calls = []

    def doctor():
        calls.append(1)
        return result

    monkeypatch.setattr(deployment, "doctor", doctor)
    assert deployment.managed_runtime_preflight() is result
    result.update(
        software_profile_matches=False, errors=["current source is unqualified"]
    )
    with pytest.raises(RuntimeError, match="current source is unqualified"):
        deployment.managed_runtime_preflight()
    assert len(calls) == 2


@pytest.fixture
def manager(monkeypatch):
    package, config = ModuleType("vllm_hust_ext"), ModuleType("vllm_hust_ext.config")
    calls = []
    saved = SimpleNamespace(enabled=True, configuration=deployment.configuration())
    package.cli = SimpleNamespace(main=lambda argv: calls.append(argv) or 0)
    config.load_config = lambda: SimpleNamespace(extension=lambda _: saved)
    monkeypatch.setitem(sys.modules, "vllm_hust_ext", package)
    monkeypatch.setitem(sys.modules, "vllm_hust_ext.config", config)
    monkeypatch.setattr(metadata, "version", lambda _: "0.3.1.dev1")
    monkeypatch.setattr(
        deployment, "doctor", lambda: {"software_profile_matches": True}
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1,2,3,6")
    return saved, calls


def arguments():
    return argparse.Namespace(
        model="/model",
        tensor_parallel_size=4,
        host="127.0.0.1",
        port=8123,
        served_model_name="qwen",
        trace_path=None,
        dry_run=True,
        experimental_prefill_groups=False,
    )


def test_serve_uses_manager_with_explicit_enabled_matching_configuration(manager):
    _, calls = manager
    assert deployment.serve(arguments()) == 0
    assert calls[0][:3] == ["run", "--dry-run", "--"]
    assert "--scheduler-cls" not in calls[0]


def test_group_profile_requires_explicit_matching_activation(manager):
    saved, calls = manager
    args = arguments()
    args.experimental_prefill_groups = True
    with pytest.raises(ValueError, match="saved Manager configuration differs"):
        deployment.serve(args)
    assert calls == []
    saved.configuration = deployment.configuration(experimental_prefill_groups=True)
    assert deployment.serve(args) == 0
    command = calls[0]
    additional = json.loads(command[command.index("--additional-config") + 1])
    assert additional["ascend_compilation_config"] == {
        "enable_npugraph_ex": False,
        "enable_static_kernel": False,
    }
    assert (
        saved.configuration["launch_options"]["batch_admission_policy_config"][
            "compile_prefill_groups"
        ]
        is True
    )
    args.experimental_prefill_groups = False
    with pytest.raises(ValueError, match="saved Manager configuration differs"):
        deployment.serve(args)
    assert len(calls) == 1


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_group_profile_rejects_non_boolean_opt_in(flag):
    with pytest.raises(ValueError, match="must be a boolean"):
        deployment.configuration(experimental_prefill_groups=flag)
    with pytest.raises(ValueError, match="must be a boolean"):
        deployment.host_command("/model", experimental_prefill_groups=flag)


def test_raw_recipe_and_explicit_group_cli_do_not_start_models(capsys):
    raw = deployment.configuration()
    assert (
        "compile_prefill_groups"
        not in raw["launch_options"]["batch_admission_policy_config"]
    )
    command = deployment.host_command("/model")
    assert command[command.index("--additional-config") + 1] == (
        '{"enable_cpu_binding":false}'
    )
    assert (
        deployment.main(["plan", "--model", "/model", "--experimental-prefill-groups"])
        == 0
    )
    planned = json.loads(capsys.readouterr().out)
    assert planned["configuration"] == deployment.configuration(
        experimental_prefill_groups=True
    )
    assert planned["host_command"] == deployment.host_command(
        "/model", experimental_prefill_groups=True
    )


@pytest.mark.parametrize(
    "failure", ["disabled", "configuration", "stack", "devices", "manager"]
)
def test_serve_cannot_silently_run_native_or_another_profile(
    manager, monkeypatch, failure
):
    saved, calls = manager
    if failure == "disabled":
        saved.enabled = False
    elif failure == "configuration":
        saved.configuration = {"launch_options": {}}
    elif failure == "stack":
        monkeypatch.setattr(
            deployment,
            "doctor",
            lambda: {"software_profile_matches": False, "errors": ["source differs"]},
        )
    elif failure == "devices":
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES")
    else:
        monkeypatch.setattr(metadata, "version", lambda _: "0.3.0")
    with pytest.raises(ValueError):
        deployment.serve(arguments())
    assert calls == []
