import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.common.geometry import (BLENDER_BODY_TO_FLU, euler_xyz, geodesic_deg, kabsch, project,
                                       quat_to_rotmat, rot_z, rotmat_to_6d, rotmat_to_euler_zyx,
                                       rotmat_to_quat, sixd_to_rotmat)

RNG = np.random.default_rng(0)
RANDOM_R = Rotation.random(20, random_state=1).as_matrix()


def test_euler_xyz_matches_blender_convention():
    e = np.array([0.3, -1.1, 2.0])
    expected = Rotation.from_euler("xyz", e).as_matrix()       # scipy 'xyz' estrinseco = Rz @ Ry @ Rx
    np.testing.assert_allclose(euler_xyz(e), expected, atol=1e-12)


def test_rot_z_90():
    np.testing.assert_allclose(rot_z(90) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_kabsch_recovers_rotation_and_translation():
    src = RNG.normal(size=(4, 3))
    for R in RANDOM_R:
        t = RNG.normal(size=3)
        R_est, t_est = kabsch(src, src @ R.T + t)
        np.testing.assert_allclose(R_est, R, atol=1e-9)
        np.testing.assert_allclose(t_est, t, atol=1e-9)


def test_kabsch_never_returns_reflection():
    src = RNG.normal(size=(5, 3))
    dst = src * np.array([1, 1, -1])                            # specchio: il migliore proprio resta una rotazione
    R, _ = kabsch(src, dst)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_project_principal_point():
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    np.testing.assert_allclose(project(K, np.array([[0, 0, 5.0], [1, 0, 2.0]])), [[960, 540], [1460, 540]])


def test_geodesic():
    assert geodesic_deg(np.eye(3), rot_z(30)) == pytest.approx(30)
    assert geodesic_deg(rot_z(10), rot_z(10)) == pytest.approx(0, abs=1e-6)


def test_quat_and_6d_roundtrip():
    for R in RANDOM_R:
        np.testing.assert_allclose(quat_to_rotmat(rotmat_to_quat(R)), R, atol=1e-9)
        np.testing.assert_allclose(sixd_to_rotmat(rotmat_to_6d(R)), R, atol=1e-9)
    assert rotmat_to_quat(np.eye(3))[0] == pytest.approx(1.0)  # ordine (w, x, y, z)


def test_euler_zyx_order_is_yaw_pitch_roll():
    R = Rotation.from_euler("ZYX", [40, 10, -5], degrees=True).as_matrix()
    np.testing.assert_allclose(rotmat_to_euler_zyx(R), [40, 10, -5], atol=1e-9)


def test_blender_to_flu_is_proper_rotation_and_maps_front():
    assert np.isclose(np.linalg.det(BLENDER_BODY_TO_FLU), 1.0)
    np.testing.assert_allclose(BLENDER_BODY_TO_FLU @ [0, -1, 0], [1, 0, 0])   # davanti Blender (-y) -> +x FLU
    np.testing.assert_allclose(BLENDER_BODY_TO_FLU @ [-1, 0, 0], [0, -1, 0])  # destra Blender (-x) -> -y FLU
