"""Capture non-secret final environment after all owned TP8 services stop."""
import datetime
import importlib.metadata
import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
path = OUT / "environment-final.json"
assert not path.exists(), "Refusing to overwrite final environment"
npu = subprocess.check_output(["npu-smi", "info"], text=True)
versions = {}
for package in ("numpy", "torch", "torch-npu", "vllm", "vllm-ascend"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
value = {
    "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "npu_smi_info": npu,
    "idle_npu_count": npu.count("No running processes found in NPU"),
    "software_versions": versions,
    "implementation_commit": "258aef6ce52b7fd77cfe9a7f70d1ea29a13f7d41",
    "working_head_at_capture": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "cpu_stat": Path("/sys/fs/cgroup/cpu.stat").read_text(),
    "note": "No initial user service existed; experiment-owned services stopped; no service auto-restored.",
}
path.write_text(json.dumps(value, indent=2) + "\n")
assert value["idle_npu_count"] == 8, "Not all 8 NPUs are idle"
print(json.dumps({"idle_npu_count": value["idle_npu_count"], "software_versions": versions}))
