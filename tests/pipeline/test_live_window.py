import numpy as np
import pytest

from dronepose.eval.units import GRAVITY_FROM_CAMERA as G, METERS_PER_UNIT
from dronepose.pipeline.forecast_now import live_window


def _h(cls, valid=True, t=(0, 0, 10.0)):
    return {"R": np.eye(3).tolist() if valid else None, "t": list(t) if valid else None, "cls": cls}


def test_mixed_classes_rescale_per_frame():
    hist = [_h("02")] * 10 + [_h("03")] * 5
    P, R, V, ref = live_window(hist)
    assert V.all() and ref == "03"
    assert np.allclose(P[0], G @ [0, 0, 10.0] * METERS_PER_UNIT["02"])
    assert np.allclose(P[-1], G @ [0, 0, 10.0] * METERS_PER_UNIT["03"])


def test_frames_predicted_08_or_without_class_are_lost():
    hist = [_h("02")] * 13 + [_h("08"), _h(None, valid=False)]
    P, R, V, ref = live_window(hist)
    assert V.sum() == 13 and not V[-2] and not V[-1]
    assert ref == "02"


def test_no_metric_frame_gives_no_reference():
    assert live_window([_h("08")] * 15)[3] is None


def test_custom_gravity_frame_is_used():
    a = np.deg2rad(50)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    G2 = Rz @ G
    t = (0.3, -0.2, 8.0)
    hist = [_h("02", t=t)] * 15
    P, R, V, ref = live_window(hist, G2)
    assert np.allclose(P[0], G2 @ np.array(t) * METERS_PER_UNIT["02"])
    assert np.allclose(R[0], G2)
    P0 = live_window(hist)[0]
    assert np.allclose(P0[0], G @ np.array(t) * METERS_PER_UNIT["02"])
    assert not np.allclose(P[0], P0[0])
