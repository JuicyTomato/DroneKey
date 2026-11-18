import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dronepose.data import mav6d
from dronepose.data.drone_models import load_drone_models
from dronepose.eval.units import GRAVITY_FROM_CAMERA
from dronepose.pipeline.perception import MS_KEYS, Perception, PerceptionModels, undistort_perception
from dronepose.pipeline.sources import Frame, SceneInfo

K = np.array([[2666.6667, 0, 960], [0, 2666.6667, 540], [0, 0, 1.0]])
NAMES = {i: d for i, d in enumerate(["01", "02", "03", "04", "05", "06", "08"])}


def _frame():
    img = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    return Frame("14_02_07_000_0001", 1, img, K, None)


def _info():
    return SceneInfo(GRAVITY_FROM_CAMERA, None, load_drone_models(__import__("pathlib").Path("configs/drone_models.json")))


def _det(box=(100, 50, 140, 80), conf=0.9):
    def detect(img):
        assert img.shape == (360, 640, 3)
        return (np.array([box], float), np.array([conf]), np.array([1.0]), np.zeros((1, 4, 2)), None)
    return detect


def _kp(crop, box):
    # plausible square of keypoints around the box centre (frame pixels)
    cx, cy = box.x0 + box.side / 2, box.y0 + box.side / 2
    return np.array([[cx + 20, cy, 1], [cx, cy + 20, 1], [cx - 20, cy, 1], [cx, cy - 20, 1]], float)


def _cls(crop):
    return np.array([0.05, 0.7, 0.05, 0.05, 0.05, 0.05, 0.05]), NAMES


def _boom(*a):
    raise AssertionError("must not be called")


def test_perceive_scales_box_to_full_frame():
    p = PerceptionModels(_det(), _kp, _cls).perceive(_frame(), _info())
    assert p.box == pytest.approx([300, 150, 420, 240])
    assert p.box_conf == pytest.approx(0.9)
    assert np.isfinite(p.kpts).all()
    assert p.probs["02"] == pytest.approx(0.7)
    assert set(p.reproj) == set(NAMES.values())


def test_perceive_no_detection():
    empty = lambda img: (np.zeros((0, 4)), np.zeros(0), np.zeros(0), None, None)
    p = PerceptionModels(empty, _boom, _boom).perceive(_frame(), _info())
    assert p.box is None and p.probs is None and p.reproj is None
    assert np.isnan(p.kpts[:, :2]).all()


def test_perceive_low_conf_is_no_detection():
    p = PerceptionModels(_det(conf=0.2), _boom, _boom).perceive(_frame(), _info())
    assert p.box is None


def test_same_crop_for_keypoints_and_classifier():
    seen = []
    def kp(crop, box):
        seen.append(crop.copy())
        return _kp(crop, box)
    def cls(crop):
        seen.append(crop.copy())
        return _cls(crop)
    PerceptionModels(_det(), kp, cls).perceive(_frame(), _info())
    assert len(seen) == 2 and np.array_equal(seen[0], seen[1])


def test_fixed_class_skips_class_stages():
    info = _info()
    info.fixed_class = "05"
    p = PerceptionModels(_det(), _kp, _boom).perceive(_frame(), info)
    assert p.probs is None and p.reproj is None
    assert p.ms["classify"] == 0 and p.ms["geometry"] == 0


def test_row_roundtrip():
    p = PerceptionModels(_det(), _kp, _cls).perceive(_frame(), _info())
    q = Perception.from_row(p.to_row())
    assert q.frame_id == p.frame_id and q.box == p.box and q.probs == p.probs and q.reproj == p.reproj
    assert np.array_equal(q.kpts, p.kpts)
    empty = lambda img: (np.zeros((0, 4)), np.zeros(0), np.zeros(0), None, None)
    n = PerceptionModels(empty, _boom, _boom).perceive(_frame(), _info())
    m = Perception.from_row(n.to_row())
    assert np.isnan(m.kpts[:, :2]).all() and m.box is None


def test_ms_keys():
    p = PerceptionModels(_det(), _kp, _cls).perceive(_frame(), _info())
    assert set(p.ms) == set(MS_KEYS) and all(v >= 0 for v in p.ms.values())


def test_box_scale_follows_frame_width():
    img = np.zeros((720, 1280, 3), np.uint8)
    fr = Frame("x_0001", 1, img, K, None)
    det = lambda im: (np.array([[100, 50, 140, 80]], float), np.array([0.9]), np.array([1.0]), np.zeros((1, 4, 2)), None)
    p = PerceptionModels(det, _kp, _cls).perceive(fr, _info())
    assert p.box == pytest.approx([200, 100, 280, 160])


def test_top_conf_from_config():
    assert PerceptionModels(_det(conf=0.15), _kp, _cls, top_conf=0.1).perceive(_frame(), _info()).box is not None
    assert PerceptionModels(_det(conf=0.15), _boom, _boom).perceive(_frame(), _info()).box is None


def test_undistort_perception_matches_e2():
    spec = importlib.util.spec_from_file_location("eval_e2", Path("scripts/real/eval_e2.py"))
    e2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2)
    k = np.array([[900, 500, 0.9], [1000, 480, 0.8], [np.nan, np.nan, 0.0], [905, 610, 0.7]], float)
    p = Perception("f", 1, [0, 0, 1, 1], 0.9, k, None, None, {})
    out = undistort_perception(p, mav6d.K, mav6d.DIST)
    assert out is not p and np.array_equal(p.kpts, k, equal_nan=True)
    np.testing.assert_allclose(out.kpts, e2.undistort_kpts(k, mav6d.K, mav6d.DIST), equal_nan=True)
    assert np.isnan(out.kpts[2, :2]).all()
    assert np.array_equal(out.kpts[:, 2], k[:, 2])
    assert not np.allclose(out.kpts[0, :2], k[0, :2])
    assert undistort_perception(p, mav6d.K, None) is p
