import numpy as np

from dronepose.data import mav6d
from dronepose.pipeline.draw import OUT_H, OUT_W, band_lines, draw_frame
from dronepose.pipeline.perception import Perception
from dronepose.pipeline.sources import Frame
from dronepose.viz.overlay import draw_overlay

K = np.array([[2666.6667, 0, 960], [0, 2666.6667, 540], [0, 0, 1.0]])


def test_draw_frame_shape_and_no_mutation():
    img = np.full((1080, 1920, 3), 80, np.uint8)
    fr = Frame("14_02_07_000_0001", 1, img, K, {"kpts2d": [[900, 500], [1000, 500], [1000, 600], [900, 600]]})
    k = np.array([[900, 500, 1], [1000, 500, 1], [1000, 600, 1], [900, 600, 1]], float)
    p = Perception(fr.frame_id, 1, [850, 450, 1050, 650], 0.9, k, None, None, {"detect": 5.0})
    res = {"R": np.eye(3).tolist(), "t": [0, 0, 10.0], "drone_model": "02"}
    fut = np.array([[0, 0, 10.0], [0.1, 0, 10.0], [0.2, 0, 10.0]])
    before = img.copy()
    out = draw_frame(fr, p, res, fut, "streaming", 42.0)
    assert np.array_equal(img, before)
    assert out.shape == (OUT_H, OUT_W, 3)
    none = draw_frame(fr, Perception(fr.frame_id, 1, None, 0.0, np.full((4, 3), np.nan), None, None, {}),
                      {"R": None, "t": None, "drone_model": None}, None, "offline", None)
    assert none.shape == (OUT_H, OUT_W, 3)


def test_overlay_with_distortion_moves_axes():
    img = np.full((1080, 1920, 3), 80, np.uint8)
    k = np.array([[900, 500, 1], [1000, 500, 1], [1000, 600, 1], [900, 600, 1]], float)
    R, t = np.eye(3), np.array([0.5, 0.3, 3.0])
    fut = np.array([[0.5, 0.3, 3.0], [0.8, 0.3, 3.0], [1.1, 0.4, 3.0]])
    base = draw_overlay(img, mav6d.K, np.zeros((0, 2)), k, R, t, fut)
    assert np.array_equal(base, draw_overlay(img, mav6d.K, np.zeros((0, 2)), k, R, t, fut, dist=None))
    dist = draw_overlay(img, mav6d.K, np.zeros((0, 2)), k, R, t, fut, dist=mav6d.DIST)
    assert not np.array_equal(base, dist)


def test_band_lines_forecast_text():
    img = np.zeros((1080, 1920, 3), np.uint8)
    fr = Frame("f", 1, img, K, {"drone": "05"})
    res = {"R": np.eye(3).tolist(), "t": [0, 0, 3.0], "drone_model": "05"}
    fut = np.zeros((3, 3))
    old = band_lines(fr, res, fut, "streaming", 42.0)
    assert "previsione 1 s: si" in old[1] and "42 ms/frame" in old[1]
    assert "previsione 1 s: no" in band_lines(fr, res, None, "streaming", None)[1]
    new = band_lines(fr, res, None, "streaming", 42.0, show_forecast=False)
    assert all("previsione" not in s for s in new) and "42 ms/frame" in new[1]
    assert draw_frame(fr, Perception("f", 1, None, 0.0, np.full((4, 3), np.nan), None, None, {}), res, None,
                      "streaming", 42.0, show_forecast=False).shape == (OUT_H, OUT_W, 3)
