import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.forecast.dataset import Trajectory
from dronepose.forecast.realism import coverage_2d, descriptors


def traj(p, R):
    t = np.arange(len(p)) / 30
    return Trajectory(scene="gen", drone="03", split="train", motion_type="normale", copies=[], s_c=1.0, times=t,
                      p_G=p, R_G=R)


def test_descriptors_on_known_motion():
    t = np.arange(120) / 30
    w = 2.0 / 4.0
    p = np.stack([4 * np.cos(w * t), 4 * np.sin(w * t), 0 * t], -1)
    R = Rotation.from_euler("z", w * t[:, None]).as_matrix()            # yaw at the turn rate
    d = descriptors([traj(p, R)])
    np.testing.assert_allclose(d["speed"], 2.0, rtol=1e-3)
    np.testing.assert_allclose(d["accel"], 1.0, rtol=1e-2)
    np.testing.assert_allclose(d["jerk"], 2.0 * w ** 2, rtol=2e-2)       # uniform circle: |j| = v w^2, not 0
    np.testing.assert_allclose(d["tilt_deg"], 0.0, atol=1e-5)          # arccos near 1: ~1e-6 deg of round-off
    np.testing.assert_allclose(d["omega_deg_s"], np.degrees(w), rtol=1e-6)
    assert len(d["curve_s"]) == 1 and d["curve_s"][0] > 3.5                # one long turn


def test_heading_motion_angle():
    t = np.arange(60) / 30
    p = np.stack([0 * t, 1.5 * t, 0 * t], -1)                           # moving along G y
    R = np.repeat(np.eye(3)[None], 60, 0)                               # nose along G x
    d = descriptors([traj(p, R)])
    np.testing.assert_allclose(d["heading_motion_deg"], 90.0, atol=1e-6)
    assert len(d["hover_s"]) == 0


def test_hover_durations():
    p = np.zeros((90, 3))
    p[45:, 0] = np.arange(45) / 30 * 1.0                                # 1.5 s still, then 1 m/s
    d = descriptors([traj(p, np.repeat(np.eye(3)[None], 90, 0))])
    assert len(d["hover_s"]) == 1 and abs(d["hover_s"][0] - 1.5) < 0.15


def test_coverage_2d():
    x = np.array([0.1, 0.2, 1.1, 2.5, 3.9, 3.8])
    y = np.array([0.1, 0.3, 1.2, 2.2, 3.5, 3.1])
    edges = np.linspace(0, 4, 5)
    assert coverage_2d(x, y, (edges, edges)) == 4 / 16


def test_non_metric_trajectories_skipped():
    t = np.arange(60) / 30
    a = traj(np.stack([t, t, t], -1), np.repeat(np.eye(3)[None], 60, 0))
    b = traj(np.stack([t, t, t], -1), np.repeat(np.eye(3)[None], 60, 0))
    b.s_c = None
    assert len(descriptors([a, b])["speed"]) == len(descriptors([a])["speed"])
