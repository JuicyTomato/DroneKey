import pytest

from dronepose.common.timing import end_to_end_fps, latency_summary


def test_latency_summary_drops_warmup():
    s = latency_summary([100.0] * 50 + [float(i) for i in range(1, 11)])
    assert s["n"] == 10
    assert s["median_ms"] == 5.5
    assert s["p90_ms"] == pytest.approx(9.1)


def test_latency_summary_needs_more_than_warmup():
    with pytest.raises(ValueError):
        latency_summary([1.0] * 50)


def test_end_to_end_fps():
    assert end_to_end_fps([10, 15]) == 40.0
