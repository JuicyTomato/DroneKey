import numpy as np

from dronepose.pipeline.draw import OUT_H, OUT_W, draw_frame
from dronepose.pipeline.perception import Perception
from dronepose.pipeline.sources import Frame

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
