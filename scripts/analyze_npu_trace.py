#!/usr/bin/env python3
"""Measure device-task interval overlap without importing torch or using an NPU.

Examples::

    python scripts/analyze_npu_trace.py trace_view.json --output overlap.json
    python scripts/analyze_npu_trace.py kernel_details.csv

An optional --lane-map JSON uses the exact device/stream keys printed by this
script: {"device:0": {"prefill": ["physical:5"], "decode": ["physical:6"]}}.
Only provide this after verifying the worker's physical stream IDs against the
trace. Opaque NPU stream handles and PyTorch stream IDs are not trace IDs.

Format references: installed torch_npu profiler _csv_headers.py,
_kernel_view_parser.py, _event_bean.py; CANN task_time_viewer.py and
trace_view_header_constant.py. Chrome trace ts/dur and CSV (us) fields are us.
Default torch_npu kernel_details.csv omits stream IDs: that format alone cannot
establish cross-stream overlap. Export all kernel headers or use trace_view.json.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

AI_CORE_TYPES = {"AI_CORE", "AI_VECTOR_CORE", "AI_VECTOR", "MIX_AIC", "MIX_AIV"}
HCCL_TYPES = {"HCCL", "HCCL_AI_CPU", "COMMUNICATION"}
DEVICE_PROCESSES = {"Ascend Hardware", "Task Scheduler"}
COMMUNICATION_NAME = re.compile(
    r"hccl|hcom|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all", re.I
)


@dataclass(frozen=True)
class Kernel:
    device: str
    stream: str | None
    start: int  # integer nanoseconds retain precision for epoch timestamps
    end: int
    kind: str
    name: str


def _get(row: dict, *names: str) -> Any:
    normalized = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in row.items()}
    for name in names:
        value = normalized.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if value is not None and str(value).strip():
            return value
    return None


def _ns(value: Any) -> int:
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"Invalid timestamp/duration: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"Nonfinite timestamp/duration: {value!r}")
    return int((number * 1000).to_integral_value())


def _stream(row: dict, tid: Any = None) -> str | None:
    physical = _get(row, "Physic Stream Id", "Physical Stream Id")
    if physical is not None:
        return f"physical:{_identifier(physical)}"
    generic = _get(row, "Stream Id", "stream")
    if generic is not None:
        return f"stream:{_identifier(generic)}"
    return f"logical:{_identifier(tid)}" if tid is not None else None


def _identifier(value: Any) -> str:
    text = str(value).strip()
    try:
        number = Decimal(text)
        if number.is_finite() and number == number.to_integral_value():
            return str(int(number))
    except InvalidOperation:
        pass
    return text


def _kind(task_type: Any, name: str) -> str | None:
    task_type = str(task_type or "").strip().upper().replace(" ", "_")
    # HCCL may itself execute AI-core subkernels. Exclude recognizable
    # communication kernels before classifying ordinary model computation.
    if task_type in HCCL_TYPES or COMMUNICATION_NAME.search(name):
        return "hccl" if task_type else None
    if task_type in AI_CORE_TYPES:
        return "ai_core"
    if task_type == "AI_CPU":
        return "ai_cpu"
    return None


def _open(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8-sig")
        if path.suffix == ".gz"
        else (path.open(encoding="utf-8-sig", newline=""))
    )


def read_trace(path: Path) -> tuple[list[Kernel], dict]:
    with _open(path) as handle:
        data = json.load(handle, parse_float=Decimal)
    events = data.get("traceEvents") if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError("Expected a trace event list or a traceEvents object")
    processes, labels = {}, {}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "M":
            continue
        value = event.get("args", {}).get("name", "")
        if event.get("name") == "process_name":
            processes[str(event.get("pid"))] = str(value)
        elif event.get("name") == "process_labels":
            labels[str(event.get("pid"))] = str(
                _get(event.get("args", {}), "labels", "name") or ""
            )
    kernels = []
    ignored = Counter()
    communication_ranges = 0
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        pid = str(event.get("pid"))
        process = processes.get(pid, "")
        args = event.get("args") or {}
        if process == "Communication":
            # This track mixes entire HCCL operations and nested transport
            # ranges. Counting it as kernels would double-count and overclaim.
            communication_ranges += 1
            continue
        if process not in DEVICE_PROCESSES:
            ignored["not_a_recognized_device_task_process"] += 1
            continue
        name = str(event.get("name", ""))
        kind = _kind(_get(args, "Task Type"), name)
        if kind is None:
            ignored["unknown_or_noncompute_task_type"] += 1
            continue
        try:
            start, duration = _ns(event["ts"]), _ns(event["dur"])
        except (ValueError, KeyError):
            ignored["invalid_timing"] += 1
            continue
        if duration <= 0:
            ignored["nonpositive_duration"] += 1
            continue
        device = _get(args, "Device Id", "device")
        if device is None:
            match = re.fullmatch(r"NPU\s+(\d+)", labels.get(pid, ""))
            device = match.group(1) if match else None
        domain = (
            f"device:{_identifier(device)}"
            if device is not None
            else f"trace-pid:{pid}"
        )
        kernels.append(
            Kernel(
                domain,
                _stream(args, event.get("tid")),
                start,
                start + duration,
                kind,
                name,
            )
        )
    return kernels, {
        "format": "chrome_trace",
        "ignored_complete_events": dict(ignored),
        "excluded_hccl_operation_or_transport_ranges": communication_ranges,
        "classification": (
            "device process metadata + explicit task type; never CPU ranges"
        ),
    }


def read_csv(path: Path) -> tuple[list[Kernel], dict]:
    kernels, ignored = [], Counter()
    with _open(path) as handle:
        rows = csv.DictReader(handle)
        headers = rows.fieldnames or []
        if not any(h.strip() in ("Accelerator Core", "Task Type") for h in headers):
            raise ValueError("CSV is missing Accelerator Core / Task Type")
        for row in rows:
            name = str(_get(row, "Name", "Op Name") or "")
            kind = _kind(_get(row, "Accelerator Core", "Task Type"), name)
            if kind is None:
                ignored["unknown_or_noncompute_task_type"] += 1
                continue
            try:
                start = _ns(_get(row, "Start Time(us)", "Task Start Time(us)"))
                duration = _ns(_get(row, "Duration(us)", "Task Duration(us)"))
            except ValueError:
                ignored["invalid_timing"] += 1
                continue
            if duration <= 0:
                ignored["nonpositive_duration"] += 1
                continue
            device = _get(row, "Device_id", "Device Id")
            if device is None:
                ignored["missing_device_id"] += 1
                continue
            kernels.append(
                Kernel(
                    f"device:{_identifier(device)}",
                    _stream(row),
                    start,
                    start + duration,
                    kind,
                    name,
                )
            )
    return kernels, {
        "format": "kernel_csv",
        "headers": headers,
        "ignored_rows": dict(ignored),
        "classification": "kernel export's Accelerator Core / Task Type column",
    }


def interval_metrics(intervals: list[tuple[int, int, str]]) -> dict:
    """Sweep half-open intervals; multiple overlapping rows on one stream count once."""
    events = defaultdict(Counter)
    for start, end, stream in intervals:
        if end <= start:
            raise ValueError("Intervals must have positive duration")
        events[start][stream] += 1
        events[end][stream] -= 1
    active = Counter()
    union = overlap = maximum = 0
    previous = None
    for timestamp in sorted(events):
        count = len(active)
        if previous is not None and count:
            elapsed = timestamp - previous
            union += elapsed
            if count >= 2:
                overlap += elapsed
            maximum = max(maximum, count)
        active.update(events[timestamp])
        active = +active
        previous = timestamp
    return {
        "active_union_us": union / 1000,
        "cross_stream_overlap_us": overlap / 1000,
        "overlap_fraction_of_active_union": overlap / union if union else None,
        "max_simultaneously_active_streams": maximum,
    }


def _metrics(kernels: list[Kernel]) -> dict:
    known = [k for k in kernels if k.stream is not None]
    result = interval_metrics([(k.start, k.end, k.stream) for k in known])
    if not known:
        result = {key: None for key in result}
    result.update(
        {
            "kernel_count": len(kernels),
            "kernel_count_with_stream": len(known),
            "kernel_count_missing_stream": len(kernels) - len(known),
            "streams": sorted({k.stream for k in known}),
        }
    )
    return result


def summarize(
    path: Path,
    lane_map: dict | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> dict:
    suffixes = path.suffixes
    kernels, details = read_csv(path) if ".csv" in suffixes else read_trace(path)
    if start_ns is not None and end_ns is not None and start_ns >= end_ns:
        raise ValueError("Measurement window start must precede end")
    if start_ns is not None or end_ns is not None:
        clipped = []
        for kernel in kernels:
            start = (
                max(kernel.start, start_ns) if start_ns is not None else kernel.start
            )
            end = min(kernel.end, end_ns) if end_ns is not None else kernel.end
            if start < end:
                clipped.append(
                    Kernel(
                        kernel.device,
                        kernel.stream,
                        start,
                        end,
                        kernel.kind,
                        kernel.name,
                    )
                )
        kernels = clipped
    devices = {}
    grouped = defaultdict(list)
    for kernel in kernels:
        grouped[kernel.device].append(kernel)
    for device, rows in sorted(grouped.items()):
        compute = [k for k in rows if k.kind == "ai_core"]
        streams = {}
        for stream in sorted({k.stream for k in rows if k.stream is not None}):
            stream_rows = [k for k in rows if k.stream == stream]
            streams[stream] = {
                "kernel_count_by_kind": dict(Counter(k.kind for k in stream_rows)),
                "sample_kernel_names": list(dict.fromkeys(k.name for k in stream_rows))[
                    :8
                ],
            }
        report = {
            "ai_core_compute": _metrics(compute),
            "physical_stream_ai_core_compute": _metrics(
                [k for k in compute if k.stream and k.stream.startswith("physical:")]
            ),
            "hccl_device_tasks": _metrics([k for k in rows if k.kind == "hccl"]),
            "all_classified_device_tasks": _metrics(rows),
            "streams": streams,
        }
        if lane_map and device in lane_map:
            mapping = lane_map[device]
            if not isinstance(mapping, dict) or set(mapping) != {"prefill", "decode"}:
                raise ValueError("Each lane mapping must contain prefill and decode")
            if any(not isinstance(v, list) or not v for v in mapping.values()):
                raise ValueError("Lane mappings must contain nonempty stream-ID lists")
            p, d = set(mapping["prefill"]), set(mapping["decode"])
            if p & d or (p | d) - streams.keys():
                raise ValueError(f"Overlapping or unknown lane stream IDs for {device}")
            mapped = [k for k in compute if k.stream in p | d]
            pair = interval_metrics(
                [
                    (k.start, k.end, "prefill" if k.stream in p else "decode")
                    for k in mapped
                ]
            )
            # Here the two counters are lanes, each a union of its stream IDs.
            report["mapped_prefill_decode_compute"] = {
                **pair,
                "mapping": mapping,
                "attribution": (
                    "user-supplied stream mapping; verify against worker logs"
                ),
            }
        devices[device] = report
    if lane_map and set(lane_map) - devices.keys():
        raise ValueError("Lane mapping names a device absent from this trace")
    warnings = []
    if not kernels:
        warnings.append(
            "No recognized device kernels; this file cannot establish overlap."
        )
    if any(k.stream is None for k in kernels):
        warnings.append(
            "Missing stream IDs: overlap metrics use only identified kernels and "
            "are null if none exist. Default kernel_details.csv omits stream IDs."
        )
    if any(k.stream and not k.stream.startswith("physical:") for k in kernels):
        warnings.append(
            "Some stream IDs are generic/logical, not physical IDs; verify exporter "
            "mapping before attributing these streams to worker lanes."
        )
    return {
        "source": str(path),
        "format_details": details,
        "devices": devices,
        "measurement_window_ns": {"start": start_ns, "end": end_ns},
        "cross_stream_ai_core_overlap_observed": any(
            (r["physical_stream_ai_core_compute"]["cross_stream_overlap_us"] or 0) > 0
            for r in devices.values()
        )
        if any(
            r["physical_stream_ai_core_compute"]["kernel_count_with_stream"]
            for r in devices.values()
        )
        else None,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--lane-map", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-us", help="Absolute trace timestamp; clip intervals")
    parser.add_argument("--end-us", help="Absolute trace timestamp; clip intervals")
    args = parser.parse_args()
    try:
        mapping = json.loads(args.lane_map.read_text()) if args.lane_map else None
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError("Lane mapping must be a JSON object")
        report = {
            "schema_version": 1,
            "meaning": (
                "Device timestamp interval overlap across distinct streams on one "
                "device. This is not proof of simultaneous AI-core occupancy, "
                "speedup, or P/D attribution without verified lane mappings."
                " Missing/unclassified events mean absence is not proof of no overlap."
            ),
            "method": (
                "Half-open intervals, union per stream, integer nanosecond sweep; "
                "AI-core/vector/mixed compute separated from recognized HCCL. "
                "No cross-device aggregation or CPU-range evidence. Sources are "
                "analyzed separately to avoid double-counting trace/CSV exports."
            ),
            "sources": [
                summarize(
                    path,
                    mapping,
                    _ns(args.start_us) if args.start_us is not None else None,
                    _ns(args.end_us) if args.end_us is not None else None,
                )
                for path in args.traces
            ],
        }
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Trace analysis failed: {exc}\n")
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
