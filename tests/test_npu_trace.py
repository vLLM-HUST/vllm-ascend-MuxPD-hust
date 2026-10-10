"""Small synthetic device traces; no profiler, torch, or NPU is required."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_npu_trace.py"
SPEC = importlib.util.spec_from_file_location("muxpd_trace_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def metadata(pid=7, device=0, process="Ascend Hardware"):
    return [
        {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": process}},
        {
            "ph": "M",
            "pid": pid,
            "name": "process_labels",
            "args": {"labels": f"NPU {device}"},
        },
    ]


def event(start, duration, stream, *, name="MatMul", task="AI_CORE", pid=7):
    return {
        "ph": "X",
        "pid": pid,
        "tid": stream + 100,
        "ts": start,
        "dur": duration,
        "name": name,
        "args": {"Task Type": task, "Physic Stream Id": stream},
    }


def write_trace(tmp_path, events):
    path = tmp_path / "trace_view.json"
    path.write_text(json.dumps(events))
    return path


def test_half_open_intervals_count_streams_once_and_preserve_idle_gaps():
    result = analysis.interval_metrics(
        [
            (0, 10000, "p"),
            (2000, 8000, "p"),  # nested/duplicate same stream
            (5000, 15000, "d"),
            (15000, 20000, "p"),  # merely touching d
            (25000, 30000, "p"),
        ]
    )
    assert result["active_union_us"] == 25
    assert result["cross_stream_overlap_us"] == 5
    assert result["overlap_fraction_of_active_union"] == 0.2
    assert result["max_simultaneously_active_streams"] == 2


def test_only_device_compute_counts_and_hccl_ranges_are_separate(tmp_path):
    events = (
        metadata()
        + metadata(9, process="CANN")
        + metadata(10, process="Communication")
        + [
            event(0, 10, 5),
            event(5, 10, 6),
            event(2, 30, 7, name="HcomAllReduce", task="AI_CORE"),
            event(0, 100, 5, pid=9),  # CPU launch, even with spoofed Task Type
            event(0, 100, 5, pid=10, name="HcomAllReduce", task="HCCL"),
        ]
    )
    report = analysis.summarize(write_trace(tmp_path, {"traceEvents": events}))
    device = report["devices"]["device:0"]
    assert device["ai_core_compute"]["cross_stream_overlap_us"] == 5
    assert device["ai_core_compute"]["active_union_us"] == 15
    assert device["hccl_device_tasks"]["kernel_count"] == 1
    assert device["streams"]["physical:5"]["kernel_count_by_kind"] == {"ai_core": 1}
    assert report["format_details"]["excluded_hccl_operation_or_transport_ranges"] == 1


def test_identical_times_on_different_devices_do_not_prove_overlap(tmp_path):
    events = (
        metadata()
        + metadata(8, device=1)
        + [
            event(0, 10, 5),
            event(0, 10, 6, pid=8),
        ]
    )
    report = analysis.summarize(write_trace(tmp_path, events))
    assert report["cross_stream_ai_core_overlap_observed"] is False
    assert len(report["devices"]) == 2


def test_default_kernel_csv_without_streams_is_inconclusive(tmp_path):
    path = tmp_path / "kernel_details.csv"
    path.write_text(
        "Device_id,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
        "0,MatMul,MatMul,AI_CORE,100,10\n"
        "0,Add,Add,AI_VECTOR_CORE,105,10\n"
    )
    report = analysis.summarize(path)
    compute = report["devices"]["device:0"]["ai_core_compute"]
    assert compute["kernel_count_missing_stream"] == 2
    assert compute["cross_stream_overlap_us"] is None
    assert report["cross_stream_ai_core_overlap_observed"] is None
    assert report["warnings"]


def test_epoch_timestamp_precision_stream_aliases_and_invalid_rows(tmp_path):
    path = tmp_path / "kernel_details.csv"
    path.write_text(
        "Device_id,Name,Accelerator Core,Start Time(us),Duration(us),Physic Stream Id\n"
        "0,A,AI_CORE,1700000000000000.001,0.003,5\n"
        "0,B,AI_CORE,1700000000000000.002,0.003,6\n"
        "0,A,AI_CORE,1700000000000000.001,0.003,5.0\n"
        "0,C,AI_CORE,1700000000000000.001,nan,7\n"
        "0,D,AI_CORE,1700000000000000.001,-1,7\n"
    )
    report = analysis.summarize(path)
    compute = report["devices"]["device:0"]["ai_core_compute"]
    assert compute["cross_stream_overlap_us"] == 0.002
    assert compute["active_union_us"] == 0.004
    assert compute["max_simultaneously_active_streams"] == 2
    assert compute["streams"] == ["physical:5", "physical:6"]
    assert report["format_details"]["ignored_rows"] == {
        "invalid_timing": 1,
        "nonpositive_duration": 1,
    }


def test_verified_lane_map_clips_and_excludes_unmapped_streams(tmp_path):
    path = write_trace(
        tmp_path,
        metadata()
        + [
            event(0, 10, 5),
            event(5, 10, 6),
            event(0, 30, 7),
        ],
    )
    mapping = {"device:0": {"prefill": ["physical:5"], "decode": ["physical:6"]}}
    report = analysis.summarize(path, mapping, start_ns=6000, end_ns=12000)
    mapped = report["devices"]["device:0"]["mapped_prefill_decode_compute"]
    assert mapped["cross_stream_overlap_us"] == 4
    assert mapped["active_union_us"] == 6
    mapping["device:0"]["decode"] = ["physical:99"]
    with pytest.raises(ValueError, match="unknown"):
        analysis.summarize(path, mapping)


def test_host_only_trace_never_establishes_hardware_overlap(tmp_path):
    report = analysis.summarize(
        write_trace(
            tmp_path,
            metadata(process="CANN")
            + [
                event(0, 100, 5),
                event(0, 100, 6),
            ],
        )
    )
    assert report["devices"] == {}
    assert report["cross_stream_ai_core_overlap_observed"] is None
    assert report["warnings"]


def test_logical_tracks_do_not_establish_physical_stream_overlap(tmp_path):
    first, second = event(0, 10, 5), event(5, 10, 6)
    first["args"].pop("Physic Stream Id")
    second["args"].pop("Physic Stream Id")
    report = analysis.summarize(write_trace(tmp_path, metadata() + [first, second]))
    candidate = report["devices"]["device:0"]["ai_core_compute"]
    assert candidate["cross_stream_overlap_us"] == 5
    assert report["cross_stream_ai_core_overlap_observed"] is None
    assert report["warnings"]
