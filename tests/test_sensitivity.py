import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.data.canonical import to_canonical
from dronepose.data.drone_models import build_drone_models
from dronepose.data.sixdronesyn import iter_frames
from dronepose.pose.sensitivity import noisy_keypoints, run_noise, run_scale
from tests.fixtures.fake_dataset import FakeSequence, write_fake_dataset


@pytest.fixture
def seq(tmp_path):
    R = Rotation.from_euler("xyz", [0.4, -0.2, 0.9]).as_matrix()
    s = FakeSequence("09", "01", "02", 0, 30, R, np.array([0.0, 0.0, 1.5]), np.array([0.0, 0.0, 0.1]))  # 1.5 -> 4.4
    write_fake_dataset(tmp_path / "ds", [s])
    frames = list(iter_frames(tmp_path / "ds"))
    models = build_drone_models(frames)
    return [to_canonical(f, models) for f in frames], models


def test_noise_is_reproducible_and_zero_sigma_is_exact(seq):
    recs, _ = seq
    a, b = noisy_keypoints(recs, 2.0, seed=4), noisy_keypoints(recs, 2.0, seed=4)
    fid = recs[0]["frame_id"]
    np.testing.assert_array_equal(a[fid], b[fid])
    np.testing.assert_allclose(noisy_keypoints(recs, 0.0, 0)[fid][:, :2], recs[0]["kpts2d"])


def test_error_grows_with_noise_and_rows_have_distance_bins(seq):
    recs, models = seq
    rows = run_noise(recs, models, sigmas=(0.5, 10.0), seeds=(0,), strategies=("reproj",))
    allrows = {r["sigma"]: r for r in rows if r["bin"] == "all"}
    assert allrows[10.0]["median_rot_err_deg"] > allrows[0.5]["median_rot_err_deg"]
    assert {r["bin"] for r in rows} >= {"all", "<2", "2-3", "3-4", ">=4"}


def test_wrong_scale_moves_depth_proportionally_not_rotation(seq):
    recs, models = seq
    rows = {r["scale"]: r for r in run_scale(recs, models, scales=(0.8, 1.0, 1.2))}
    for s, r in rows.items():
        assert r["median_depth_ratio"] == pytest.approx(s, rel=1e-6)
        assert r["median_rot_err_deg"] < 1e-3
