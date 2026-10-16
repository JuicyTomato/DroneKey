import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.data.drone_models import load_drone_models
from dronepose.data.sixdronesyn import iter_annotation_files
from dronepose.forecast.config import load_config, metric_scales
from dronepose.forecast.frames import GRAVITY_FROM_CAMERA, to_gravity_positions, to_gravity_rotations, up_axis


def test_gravity_matrix_is_rotation_and_maps_axes():
    C = GRAVITY_FROM_CAMERA
    np.testing.assert_allclose(C @ C.T, np.eye(3))
    assert np.isclose(np.linalg.det(C), 1)
    np.testing.assert_allclose(C @ [0, -1, 0], [0, 0, 1])   # camera up -> G up
    np.testing.assert_allclose(C @ [0, 0, 1], [0, 1, 0])    # camera forward -> G forward


def test_batched_conversions():
    t = np.random.default_rng(0).normal(size=(5, 3))
    np.testing.assert_allclose(to_gravity_positions(t), t @ GRAVITY_FROM_CAMERA.T)
    R = Rotation.random(5, random_state=0).as_matrix()
    np.testing.assert_allclose(to_gravity_rotations(R), GRAVITY_FROM_CAMERA @ R)
    np.testing.assert_allclose(up_axis(to_gravity_rotations(R)), (GRAVITY_FROM_CAMERA @ R)[..., :, 2])


def test_up_axis_invariant_to_body_yaw_symmetries():
    R = Rotation.from_euler("zyx", [30, 20, -15], degrees=True).as_matrix()
    for k in range(4):
        Rk = R @ Rotation.from_euler("z", 90 * k, degrees=True).as_matrix()
        np.testing.assert_allclose(up_axis(Rk), up_axis(R), atol=1e-12)


def test_metric_scales_match_memory_table():
    s = metric_scales(load_drone_models(Path("configs/drone_models.json")), load_config().dji_diagonal_mm)
    expected = {"01": 0.7463, "02": 0.8053, "03": 0.8525, "04": 0.9334, "05": 0.9253, "06": 0.8521}
    for d, v in expected.items():
        assert abs(s[d] - v) < 1e-3
    assert s["08"] is None


def test_config_values():
    c = load_config()
    assert (c.fps, c.obs_len, c.pred_len, c.horizons_frames) == (30, 15, 30, (3, 10, 15, 30))
    assert c.motion_type["03"] == "linear" and c.motion_type["14"] == "nonlinear"
    assert c.cv_k_grid == (2, 3, 5, 10) and c.ca_k_grid == (3, 5, 10)
    assert len(c.kf_cv_q_grid) == len(c.kf_ca_q_grid) == 8
    assert (c.g, c.max_tilt_deg, c.bootstrap_b, c.bootstrap_seed) == (9.81, 60, 10000, 0)


@pytest.mark.dataset
def test_raw_camera_is_level(dataset_root):
    # Camera level in every file: 3DKey(camera).y == -0.582 * 3DKey(world).z, i.e. up = -y_cam.
    for path in iter_annotation_files(dataset_root):
        for fid, e in json.loads(path.read_text()).items():
            if fid == "camera_parameters":
                continue
            for k in ("key1", "key2", "key3", "key4"):
                assert abs(e["3DKey(camera)"][k][1] + 0.582 * e["3DKey(world)"][k][2]) < 1e-9, (path.name, fid)
