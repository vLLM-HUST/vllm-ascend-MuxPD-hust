"""Installed, opt-in launch recipe for the qualified streaming profile.

Metadata/configuration commands import neither vLLM nor accelerator libraries.
The Manager owns activation and child processes; this module never patches vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from importlib import metadata, resources
from importlib.util import find_spec
from pathlib import Path

from . import __version__

EXTENSION_ID = "org.vllm-hust.muxpd"
CORE_COMMIT = "d0f22d2bda562156e4dbf433ce645e1769b4f804"
ASCEND_COMMIT = "03766ac696fde5ab1980d80ca0b8543d3580c989"
ASCEND_BUILD_EDIT_SHA = (
    "b089d52475c6cb36fffd733bb5be3fbe4158b6845074751268938e98a1d0042c"
)
_MANAGER_BINDING_ENV = (
    "VLLM_ECPA_EVIDENCE_SINK",
    "VLLM_ECPA_EVIDENCE_STRICT",
    "VLLM_ECPA_PLAN_ID",
    "VLLM_ECPA_LAUNCH_ID",
    "VLLM_HUST_EXT_EVIDENCE_PATH",
    "VLLM_HUST_EXT_MODULE_BINDING",
)


def manifest() -> dict:
    return json.loads(
        resources.files("vllm_muxpd.manifests")
        .joinpath("vllm-hust-extension-v0.3.json")
        .read_text(encoding="utf-8")
    )


def configuration(
    trace_path: str | None = None, *, experimental_prefill_groups: bool = False
) -> dict:
    if type(experimental_prefill_groups) is not bool:
        raise ValueError("experimental_prefill_groups must be a boolean")
    # Keep the same raw-P q4 algorithm as the existing streaming entry point.
    policy = {
        "execution": "decoupled",
        "stream_mode": "overlap",
        "layers_per_slice": 4,
        "adaptive_layers": False,
        "prefill_first": False,
        "decode_graph": True,
        "compile_full_prefill": False,
        "prefill_layer_token_budget": 0,
        "decode_budget_guard": False,
        "overlap_communication": "isolated",
    }
    if experimental_prefill_groups:
        policy["compile_prefill_groups"] = True
    if trace_path is not None or experimental_prefill_groups:
        from .config import MuxPDConfig

        MuxPDConfig(**policy, trace_path=trace_path)
        if trace_path is not None:
            policy["trace_path"] = trace_path
    return {"launch_options": {"batch_admission_policy_config": policy}}


def host_command(
    model: str,
    *,
    tensor_parallel_size: int = 4,
    host: str = "127.0.0.1",
    port: int = 8002,
    served_model_name: str = "Qwen3.5-35B-A3B",
    experimental_prefill_groups: bool = False,
) -> list[str]:
    if type(experimental_prefill_groups) is not bool:
        raise ValueError("experimental_prefill_groups must be a boolean")
    if not model.strip() or not host.strip() or not served_model_name.strip():
        raise ValueError("model, host and served model name must be nonempty")
    if type(tensor_parallel_size) is not int or tensor_parallel_size not in (2, 4):
        raise ValueError("this deployment profile supports TP2 or TP4")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    compile_config = {
        "mode": 3,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "max_cudagraph_capture_size": 8,
    }
    additional_config = {"enable_cpu_binding": False}
    if experimental_prefill_groups:
        additional_config["ascend_compilation_config"] = {
            "enable_npugraph_ex": False,
            "enable_static_kernel": False,
        }
    # Class/policy selectors are deliberately absent: the Manager must project
    # the installed Bundle's three components together, with conflict checking.
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--data-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--api-server-count",
        "1",
        "--enable-expert-parallel",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "16384",
        "--max-num-batched-tokens",
        "16384",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.85",
        "--language-model-only",
        "--no-enable-chunked-prefill",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--mamba-cache-mode",
        "none",
        "--no-enforce-eager",
        "--no-numa-bind",
        "--additional-config",
        json.dumps(additional_config, separators=(",", ":")),
        "--compilation-config",
        json.dumps(compile_config, separators=(",", ":")),
    ]


def _source_checkout(distribution_name: str, expected_commit: str) -> dict:
    """The first release qualifies the existing editable native source stack.

    A matching public version or API 1.1 string alone is insufficient. Other
    native wheel/source stacks need a separately qualified profile.
    """
    from urllib.parse import unquote, urlparse

    distribution = metadata.distribution(distribution_name)
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    parsed = urlparse(direct.get("url", ""))
    if (
        direct.get("dir_info", {}).get("editable") is not True
        or parsed.scheme != "file"
        or parsed.netloc not in ("", "localhost")
    ):
        raise ValueError(f"{distribution_name} is not the qualified source checkout")
    source = Path(unquote(parsed.path)).resolve()
    # Top-level find_spec does not execute the package. Metadata can describe
    # one editable checkout while PYTHONPATH actually selects another tree.
    module = distribution_name.replace("-", "_")
    expected_origin = (source / module / "__init__.py").resolve(strict=True)
    spec = find_spec(module)
    if (
        spec is None
        or spec.origin is None
        or Path(spec.origin).resolve(strict=True) != expected_origin
    ):
        raise ValueError(
            f"{distribution_name} import origin differs from qualified checkout: "
            f"{getattr(spec, 'origin', None)!r}"
        )

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args],
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        ).strip()

    head = git("rev-parse", "HEAD")
    changes = git("status", "--porcelain", "--untracked-files=no")
    if head != expected_commit:
        raise ValueError(f"{distribution_name} source {head} is not qualified")
    if changes:
        build = source / "csrc/build_aclnn.sh"
        allowed = (
            distribution_name == "vllm-ascend" and changes == "M csrc/build_aclnn.sh"
        )
        if (
            not allowed
            or hashlib.sha256(build.read_bytes()).hexdigest() != ASCEND_BUILD_EDIT_SHA
        ):
            raise ValueError(
                f"{distribution_name} tracked sources differ from qualification"
            )
    return {
        "version": distribution.version,
        "source": str(source),
        "commit": head,
        "import_origin": str(expected_origin),
    }


def doctor() -> dict:
    from packaging.specifiers import SpecifierSet

    result = {
        "extension_id": EXTENSION_ID,
        "extension_version": __version__,
        "software_profile_matches": False,
        "sources": {},
        "errors": [],
        "scope": "software preflight only; no device initialization or inference",
    }
    for name, commit in (("vllm", CORE_COMMIT), ("vllm-ascend", ASCEND_COMMIT)):
        try:
            result["sources"][name] = _source_checkout(name, commit)
        except (
            OSError,
            ImportError,
            ValueError,
            subprocess.SubprocessError,
            metadata.PackageNotFoundError,
        ) as error:
            result["errors"].append(f"{name}: {error}")
    for name, expected in (("torch", "2.10.0+cpu"), ("torch-npu", "2.10.0.post4")):
        try:
            actual = metadata.version(name)
            result["sources"][name] = {"version": actual}
            if actual != expected:
                result["errors"].append(f"{name}: expected {expected}, found {actual}")
        except metadata.PackageNotFoundError:
            result["errors"].append(f"{name}: not installed")
    try:
        greenlet_version = metadata.version("greenlet")
        result["sources"]["greenlet"] = {"version": greenlet_version}
        if greenlet_version not in SpecifierSet(">=3.1,<4"):
            result["errors"].append("greenlet: expected >=3.1,<4")
    except metadata.PackageNotFoundError:
        result["errors"].append("greenlet: install the layered or ecpa extra")
    result["software_profile_matches"] = not result["errors"]
    return result


def managed_runtime_preflight() -> dict | None:
    """Gate every Manager class-carrier path before native resource mutation.

    The installed CLI also checks qualification, but generic Manager ``run``
    can select these classes directly. Recheck in each actual carrier process;
    never cache a parent's result across spawn or a changed import path. Direct
    legacy launches keep their existing interface checks and opt-in behavior.
    This gate does not authenticate a launch or emit effective-work evidence.
    """
    present = [name in os.environ for name in _MANAGER_BINDING_ENV]
    if not any(present):
        return None
    if not all(os.environ.get(name) for name in _MANAGER_BINDING_ENV):
        raise RuntimeError("incomplete Manager binding before MuxPD initialization")
    result = doctor()
    if not result["software_profile_matches"]:
        raise RuntimeError(
            "unqualified managed MuxPD native stack: " + "; ".join(result["errors"])
        )
    return result


def serve(args) -> int:
    from packaging.specifiers import SpecifierSet

    if metadata.version("vllm-hust-ext") not in SpecifierSet(">=0.3.1.dev1,<0.4"):
        raise ValueError(
            "MuxPD requires Manager scheduler/module carrier support >=0.3.1.dev1,<0.4"
        )
    from vllm_hust_ext import cli as manager
    from vllm_hust_ext.config import load_config

    saved = load_config().extension(EXTENSION_ID)
    if not saved.enabled:
        raise ValueError(f"explicitly configure and enable {EXTENSION_ID} first")
    if saved.configuration != configuration(
        args.trace_path, experimental_prefill_groups=args.experimental_prefill_groups
    ):
        raise ValueError(
            "saved Manager configuration differs from this streaming profile"
        )
    check = doctor()
    if not check["software_profile_matches"]:
        raise ValueError("unqualified native stack: " + "; ".join(check["errors"]))
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip():
        raise ValueError(
            "select this container's devices with ASCEND_RT_VISIBLE_DEVICES"
        )
    command = host_command(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        host=args.host,
        port=args.port,
        served_model_name=args.served_model_name,
        experimental_prefill_groups=args.experimental_prefill_groups,
    )
    return manager.main(
        ["run", *(["--dry-run"] if args.dry_run else []), "--", *command]
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("manifest", help="show static installed Bundle metadata")
    commands.add_parser(
        "doctor", help="check the qualified software profile without NPU imports"
    )
    config = commands.add_parser(
        "configuration", help="write a Manager configuration; does not enable"
    )
    config.add_argument("--output", type=Path)
    config.add_argument("--trace-path")
    config.add_argument(
        "--experimental-prefill-groups",
        action="store_true",
        help="select unqualified four-layer FX prefill groups; does not enable",
    )
    for name in ("plan", "serve"):
        child = commands.add_parser(name)
        child.add_argument("--model", required=True)
        child.add_argument(
            "--tensor-parallel-size", type=int, choices=(2, 4), default=4
        )
        child.add_argument("--host", default="127.0.0.1")
        child.add_argument("--port", type=int, default=8002)
        child.add_argument("--served-model-name", default="Qwen3.5-35B-A3B")
        child.add_argument("--trace-path")
        child.add_argument(
            "--experimental-prefill-groups",
            action="store_true",
            help="select unqualified grouped P with the required fusion backend",
        )
        if name == "serve":
            child.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "serve":
            return serve(args)
        if args.action == "manifest":
            value = manifest()
        elif args.action == "doctor":
            value = doctor()
        elif args.action == "configuration":
            value = configuration(
                args.trace_path,
                experimental_prefill_groups=args.experimental_prefill_groups,
            )
        else:
            value = {
                "extension_id": EXTENSION_ID,
                "configuration": configuration(
                    args.trace_path,
                    experimental_prefill_groups=args.experimental_prefill_groups,
                ),
                "host_command": host_command(
                    args.model,
                    tensor_parallel_size=args.tensor_parallel_size,
                    host=args.host,
                    port=args.port,
                    served_model_name=args.served_model_name,
                    experimental_prefill_groups=args.experimental_prefill_groups,
                ),
                "scope": "recipe only; Manager supplies all three component selectors",
            }
        output = json.dumps(value, indent=2) + "\n"
        if args.action == "configuration" and args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(output)
        else:
            print(output, end="")
        return int(args.action == "doctor" and not value["software_profile_matches"])
    except (OSError, ValueError, metadata.PackageNotFoundError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
