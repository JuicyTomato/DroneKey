import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.data.canonical import to_canonical
from dronepose.data.drone_models import build_drone_models
from dronepose.data.sixdronesyn import iter_frames
from dronepose.keypoints.predictors import GTPredictor
from dronepose.pipeline.skeleton import run_sequence, summarize
from dronepose.viz.overlay import draw_overlay
from tests.fixtures.fake_dataset import FakeSequence, write_fake_dataset


def _linear_sequence(tmp_path, n=40):
    seq = FakeSequence("14", "01", "03", 90, n, Rotation.from_euler("xyz", [0.3, 0.1, 0.5]).as_matrix(),
                       t0=np.array([-0.3, 0.0, 3.0]), velocity=np.array([0.01, 0.002, 0.0]))
    root = tmp_path / "ds"
    write_fake_dataset(root, [seq])
    frames = list(iter_frames(root))
    models = build_drone_models(frames)
    recs = sorted((to_canonical(f, models) for f in frames), key=lambda r: r["frame_idx"])
    return recs, models


def _blank(_record):
    return np.zeros((1080, 1920, 3), np.uint8)


def test_gt_keypoints_give_exact_pose_and_exact_cv_forecast(tmp_path):
    recs, models = _linear_sequence(tmp_path)
    res = run_sequence(recs, _blank, GTPredictor(), models, obs=15, horizon=10)
    assert all(r["rot_err_deg"] < 1e-3 and r["t_err"] < 1e-5 for r in res)
    with_forecast = [r for r in res if r["ade"] is not None]
    assert len(with_forecast) == 40 - 15 - 10 + 1
    assert max(r["fde"] for r in with_forecast) < 1e-5
    s = summarize(res)
    assert s["n_frames"] == 40 and s["n_pose"] == 40 and s["flip_rate"] == 0.0


class _DropFrame20(GTPredictor):
    def predict(self, image, record):
        out = super().predict(image, record)
        if record["frame_idx"] == 20:
            out[:] = np.nan
            out[:, 2] = 0.0
        return out


def test_missed_detection_is_skipped_not_fatal(tmp_path):
    recs, models = _linear_sequence(tmp_path)
    res = run_sequence(recs, _blank, _DropFrame20(), models, obs=15, horizon=10)
    r20 = next(r for r in res if r["frame_idx"] == 20)
    assert r20["R"] is None and r20["t"] is None and r20["rot_err_deg"] is None
    later = [r for r in res if r["ade"] is not None]
    assert max(r["fde"] for r in later) < 1e-5          # gap handled through explicit timestamps
    assert summarize(res)["n_pose"] == 39


def test_noisy_keypoints_still_reasonable(tmp_path):
    recs, models = _linear_sequence(tmp_path)
    res = run_sequence(recs, _blank, GTPredictor(sigma_px=1.0, seed=3), models)
    assert summarize(res)["median_rot_err_deg"] < 10


def test_overlay_draws_something(tmp_path):
    recs, models = _linear_sequence(tmp_path, n=2)
    r = recs[0]
    img = _blank(r)
    out = draw_overlay(img, np.array(r["K"]), np.array(r["kpts2d"]), np.array(r["kpts2d"]),
                       np.array(r["R_gt"]), np.array(r["t_gt"]), np.array([r["t_gt"], recs[1]["t_gt"]]))
    assert out.shape == img.shape and out.sum() > 0 and img.sum() == 0   # input not modified
