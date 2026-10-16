from pathlib import Path

import numpy as np

from dronepose.data.drone_models import load_drone_models
from dronepose.forecast.config import load_config, metric_scales
from dronepose.forecast.dataset import build_trajectories
from dronepose.forecast.video import TopDownView, compose_frame, draw_camera_overlay, to_camera
from dronepose.forecast.windows import make_windows

CFG = load_config()
SCALES = metric_scales(load_drone_models(Path("configs/drone_models.json")), CFG.dji_diagonal_mm)
K = np.array([[2666.6667, 0, 960], [0, 2666.6667, 540], [0, 0, 1.0]])


def _records(drone, T=60):
    return [{"scene": "14", "drone": drone, "background": "01", "degree": 0, "frame_idx": i, "split": "val",
             "t_gt": [0.1 * np.sin(i / 10), 0.2 - 0.003 * i, 2.5 + 0.01 * i], "R_gt": np.eye(3).tolist()}
            for i in range(1, T + 1)]


def test_to_camera_inverts_dataset_and_windows():
    for drone in ("01", "08"):                     # metric and non-metric
        recs = _records(drone)
        tr = build_trajectories(recs, CFG, SCALES)[0]
        w = make_windows([tr], 15, 30, 30)
        i = 5
        back = to_camera(w.fut_p[i], w.last_p[i], tr.s_c)
        expected = np.array([r["t_gt"] for r in recs])[w.start[i] + 15: w.start[i] + 45]
        np.testing.assert_allclose(back, expected, atol=1e-12)


def test_topdown_maps_bounds_inside_panel():
    view = TopDownView(np.array([[-1.0, 2.0, 0], [3.0, 5.0, 1.0]]), size=400, margin=20)
    px = view.to_px(np.array([[-1.0, 2.0, 0], [3.0, 5.0, 0], [1.0, 3.5, 0]]))
    assert ((px >= 20) & (px <= 380)).all()
    assert px[0, 1] > px[1, 1]                     # larger forward (y) is higher up in the panel


def test_compose_frame_shape():
    img = np.zeros((1080, 1920, 3), np.uint8)
    cam = draw_camera_overlay(img, K, {"vero": np.array([[0, 0, 3.0], [0.1, 0, 3.0]])}, {"vero": (0, 255, 0)})
    panel = np.zeros((720, 720, 3), np.uint8)
    out = compose_frame(cam, panel, height=720)
    assert out.shape == (720, 1280 + 720, 3)


def test_zoom_inset_handles_image_borders():
    from dronepose.forecast.video import zoom_inset
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[0:10, 0:10] = 255
    out = zoom_inset(img, np.array([3.0, 3.0]), crop=200, size=400)
    assert out.shape == (400, 400, 3)
    # crop centred on (3, 3): the white 10x10 corner lands at rows/cols 97..107 of the crop, i.e. ~194..214 after x2
    assert out[198:210, 198:210].min() == 255
    assert out[:180, :180].max() == 0 and out[230:, 230:].max() == 0   # outside the image: black padding


def test_scale_bar_is_round_and_readable():
    from dronepose.forecast.video import nice_length
    assert nice_length(0.6) == 0.5 and nice_length(1.9) == 1 and nice_length(0.25) == 0.2 and nice_length(7) == 5
