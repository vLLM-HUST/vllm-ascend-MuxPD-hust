"""Device-free coverage of event reads and their completion preconditions."""

import pytest

from vllm_muxpd.slice_timing import PrefillSliceTimer


def make_timer(*, complete=True, start_layer=0, compiled=False):
    calls = []

    class Event:
        def record(self, stream):
            calls.append(("record", stream))

        def query(self):
            calls.append(("query",))
            return complete

        def elapsed_time(self, end):
            calls.append(("elapsed",))
            return 4.5

        def synchronize(self):
            pytest.fail("slice timing must not add a synchronization")

    ticks = iter((10.0, 10.003))
    timer = PrefillSliceTimer(
        Event(), Event(), "prefill",
        start_layer=start_layer, prefill_tokens=12288, compiled=compiled,
        clock=lambda: next(ticks),
    )
    timer.start()
    timer.stop()
    return timer, calls


def test_slice_timer_keeps_submit_elapsed_and_completed_device_span_separate():
    timer, calls = make_timer(start_layer=8)
    receipt = timer.receipt(12, completion_barrier=True)
    assert receipt == {
        "version": 1, "layer_delta": 4, "first_slice": False,
        "prefill_tokens": 12288, "mode": "raw_layers",
        "host_submit_ms": pytest.approx(3.0),
        "device_elapsed_ms": 4.5, "device_complete": True,
    }
    assert calls == [
        ("record", "prefill"), ("record", "prefill"),
        ("query",), ("elapsed",),
    ]


@pytest.mark.parametrize("event_complete", [False, True])
def test_no_sampler_barrier_never_reads_device_elapsed(event_complete):
    timer, calls = make_timer(complete=event_complete)
    receipt = timer.receipt(1, completion_barrier=False)
    assert receipt["first_slice"]
    assert receipt["device_elapsed_ms"] is None
    assert not receipt["device_complete"]
    assert calls == [("record", "prefill"), ("record", "prefill")]


def test_incomplete_event_remains_host_only_even_after_sampler_barrier():
    timer, calls = make_timer(complete=False)
    receipt = timer.receipt(8, completion_barrier=True)
    assert receipt["device_elapsed_ms"] is None
    assert not receipt["device_complete"]
    assert calls[-1] == ("query",)
    assert ("elapsed",) not in calls


def test_compiled_full_forward_is_labeled_separately_from_raw_slice():
    timer, _ = make_timer(compiled=True)
    receipt = timer.receipt(40, completion_barrier=True)
    assert receipt["mode"] == "compiled_full_forward"
    assert receipt["layer_delta"] == 40
    assert receipt["first_slice"]


def test_timer_cannot_be_reused_or_publish_an_unfinished_interval():
    timer, _ = make_timer()
    with pytest.raises(RuntimeError, match="cannot be reused"):
        timer.start()
    with pytest.raises(RuntimeError, match="no active interval"):
        timer.stop()
    timer._host_submit_ms = None
    with pytest.raises(RuntimeError, match="has not stopped"):
        timer.receipt(1, completion_barrier=True)
