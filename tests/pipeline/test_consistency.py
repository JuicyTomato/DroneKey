import numpy as np
import pytest

from dronepose.keypoints.detections import Detection
from dronepose.pipeline.consistency import compare_boxes, compare_kpts, compare_probs
from dronepose.pipeline.perception import Perception


def _p(fid, box=None, kpts=None, probs=None):
    k = np.full((4, 3), np.nan) if kpts is None else np.asarray(kpts, float)
    return Perception(fid, 1, None if box is None else list(box), 0.9 if box else 0.0, k, probs, None, {})


def test_compare_boxes():
    live = {"a": _p("a", (0, 0, 10, 10)), "b": _p("b", (5, 0, 15, 10)), "c": _p("c", (0, 0, 10, 10)), "d": _p("d")}
    ref = {"a": Detection((0, 0, 10, 10), 0.9, "01"), "b": Detection((0, 0, 10, 10), 0.9, "01"),
           "c": Detection(None, 0.0, None), "d": Detection(None, 0.0, None)}
    r = compare_boxes(live, ref)
    assert r["same_presence"] == pytest.approx(0.75)        # c differs
    assert r["iou_ge_099"] == pytest.approx(0.5)            # among a, b (both with a box): a only
    assert r["n"] == 4


def test_compare_kpts():
    k = np.array([[0, 0, 1], [10, 0, 1], [10, 10, 1], [0, 10, 1]], float)
    live = {"a": _p("a", (0, 0, 1, 1), k + [0.5, 0, 0]), "b": _p("b")}
    ref = {"a": k, "b": np.full((4, 3), np.nan)}
    r = compare_kpts(live, ref)
    assert r["median_px"] == pytest.approx(0.5)
    assert r["n"] == 1


def test_compare_probs():
    a, b = {"01": 0.9, "02": 0.1}, {"01": 0.2, "02": 0.8}
    live = {f: _p(f, (0, 0, 1, 1), probs=a) for f in "wxyz"}
    ref = {"w": a, "x": a, "y": a, "z": b}
    r = compare_probs(live, ref)
    assert r["same_argmax"] == pytest.approx(0.75)
    assert r["max_abs"] == pytest.approx(0.7)
