import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.common.geometry import project
from dronepose.eval.metrics import ade, fde, rotation_error_deg, translation_error
from dronepose.pose.pnp import estimate_pose, solve_ippe
from tests.fixtures.fake_dataset import FAKE_K, FLU_MODELS

MODEL = FLU_MODELS["01"] - FLU_MODELS["01"].mean(0)


def _image(R, t):
    return project(FAKE_K, (R @ MODEL.T).T + t)


@pytest.mark.parametrize("seed", range(10))
def test_exact_keypoints_give_exact_pose(seed):
    R = Rotation.random(random_state=seed).as_matrix()
    # keep the drone in front of the camera and not edge-on
    if abs(R[2, 2]) < 0.2:
        R = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    t = np.array([0.2, -0.1, 3.0])
    sol = estimate_pose(MODEL, _image(R, t), FAKE_K)
    assert rotation_error_deg(sol.R, R) < 1e-3
    assert translation_error(sol.t, t) < 1e-5
    assert sol.reproj_px < 1e-3


def test_two_ippe_solutions_sorted_by_reprojection():
    R = Rotation.from_euler("xyz", [20, 10, 5], degrees=True).as_matrix()
    img = _image(R, np.array([0.0, 0.0, 3.0])) + np.random.default_rng(0).normal(0, 1.0, (4, 2))
    sols = solve_ippe(MODEL, img, FAKE_K)
    assert len(sols) == 2
    assert sols[0].reproj_px <= sols[1].reproj_px


def test_nan_input_raises():
    img = _image(np.eye(3), np.array([0, 0, 3.0]))
    img[1] = np.nan
    with pytest.raises(ValueError):
        estimate_pose(MODEL, img, FAKE_K)


def test_trajectory_metrics():
    gt = np.zeros((3, 3))
    pred = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
    assert ade(pred, gt) == pytest.approx(2.0)
    assert fde(pred, gt) == pytest.approx(3.0)
