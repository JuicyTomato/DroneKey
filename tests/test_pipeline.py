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


def test_run_sequence_rejects_records_from_several_sequences(tmp_path):
    import pytest
    root = tmp_path / "ds"
    r = Rotation.from_euler("xyz", [0.3, 0.1, 0.5]).as_matrix()
    write_fake_dataset(root, [
        FakeSequence("14", "01", "03", 90, 20, r, t0=np.array([0.0, 0.0, 3.0])),
        FakeSequence("14", "01", "04", 90, 20, r, t0=np.array([0.2, 0.0, 3.0])),
    ])
    frames = list(iter_frames(root))
    models = build_drone_models(frames)
    recs = [to_canonical(f, models) for f in frames]
    with pytest.raises(ValueError, match="more than one sequence"):
        run_sequence(recs, _blank, GTPredictor(), models)


import json

from dronepose.pose.disambiguation import TemporalParams


def test_strategy_fields_and_default_unchanged(tmp_path):
    recs, models = _linear_sequence(tmp_path)
    res = run_sequence(recs, _blank, GTPredictor(), models, obs=15, horizon=10)
    r = res[0]
    assert r["solution_index"] == 0 and r["rejected"] is False and r["reject_reason"] is None
    assert r["oracle_rot_err_deg"] < 1e-3
    json.dumps(res, allow_nan=False)                     # no Infinity/NaN in the output


def test_temporal_never_flips_more_than_reproj(tmp_path):
    recs, models = _linear_sequence(tmp_path)
    flips = {}
    for strategy in ("reproj", "temporal"):
        res = run_sequence(recs, _blank, GTPredictor(sigma_px=6.0, seed=1), models, strategy=strategy)
        flips[strategy] = summarize(res)["flip_rate"]
    assert flips["temporal"] <= flips["reproj"]


def test_reject_strategy_reports_coverage(tmp_path):
    recs, models = _linear_sequence(tmp_path)
    res = run_sequence(recs, _blank, GTPredictor(sigma_px=6.0, seed=1), models, strategy="temporal_reject",
                       params=TemporalParams(max_jump_deg=1.0))
    s = summarize(res)
    assert s["n_rejected"] + s["n_pose"] == s["n_frames"]
    assert 0.0 < s["coverage"] <= 1.0
    assert all(r["R"] is None for r in res if r["rejected"])


def test_missed_detection_has_null_ratio(tmp_path):
    recs, models = _linear_sequence(tmp_path)
    res = run_sequence(recs, _blank, _DropFrame20(), models, obs=15, horizon=10, strategy="temporal")
    r20 = next(r for r in res if r["frame_idx"] == 20)
    assert r20["reject_reason"] == "no_solution" and r20["reproj_ratio"] is None


def test_global_strategy_through_run_sequence(tmp_path):
    import pytest

    recs, models = _linear_sequence(tmp_path)
    kw = dict(obs=15, horizon=10)
    flips = {}
    for strategy in ("reproj", "global"):
        res = run_sequence(recs, _blank, GTPredictor(sigma_px=6.0, seed=1), models, strategy=strategy, **kw)
        json.dumps(res, allow_nan=False)                         # no Infinity/NaN in the output
        flips[strategy] = summarize(res)["flip_rate"]
    assert flips["global"] <= flips["reproj"]
    assert all(r["solution_index"] is None for r in res if r["rejected"])
    with pytest.raises(TypeError):
        run_sequence(recs, _blank, GTPredictor(), models, strategy="global", params=TemporalParams())


def test_global_strategy_gravity_gate_reaches_the_pipeline(tmp_path):
    recs, models = _linear_sequence(tmp_path)
    up = np.array(recs[0]["R_gt"])[:, 2]                         # the drone's own "up" at frame 0, so it is upright
    res = run_sequence(recs, _blank, GTPredictor(sigma_px=6.0, seed=1), models, strategy="global", up=up)
    tilts = [np.degrees(np.arccos(np.clip(np.array(r["R"])[:, 2] @ up, -1, 1))) for r in res if r["R"] is not None]
    assert tilts and max(tilts) <= 70.0
