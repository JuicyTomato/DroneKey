import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.forecast.dataset import Trajectory
from dronepose.forecast.windows import make_windows


def tr(T, offset=0.0, drone="01", s_c=1.0, scene="03"):
    t = np.arange(T) / 30
    p = np.stack([offset + t, 2 * t, 0 * t], axis=1)
    R = Rotation.from_euler("z", 10 * t[:, None], degrees=True).as_matrix()
    return Trajectory(scene=scene, drone=drone, split="val", motion_type="linear", copies=[("01", 0)], s_c=s_c,
                      times=t, p_G=p, R_G=R)


def test_window_count():
    assert len(make_windows([tr(120), tr(360)], 15, 30, 30)) == 76 + 316


def test_short_trajectory_gives_no_windows():
    w = make_windows([tr(44)], 15, 30, 30)
    assert len(w) == 0 and w.obs_p.shape == (0, 15, 3) and w.fut_p.shape == (0, 30, 3)


def test_windows_never_cross_trajectories():
    w = make_windows([tr(60, offset=0.0), tr(60, offset=100.0)], 15, 30, 30)
    absolute = np.concatenate([w.obs_p, w.fut_p], axis=1) + w.last_p[:, None]
    for i in range(len(w)):
        x = absolute[i, :, 0]
        assert (x < 50).all() if w.traj[i] == 0 else (x > 50).all()
    assert list(np.bincount(w.traj)) == [16, 16]


def test_relative_to_last_observed():
    w = make_windows([tr(60)], 15, 30, 30)
    np.testing.assert_allclose(w.obs_p[:, -1], 0)
    np.testing.assert_allclose(w.last_p[:, 0], w.start / 30 + 14 / 30)
    np.testing.assert_allclose(w.fut_p[:, 0], [[1 / 30, 2 / 30, 0]] * len(w))


def test_times():
    w = make_windows([tr(60)], 15, 30, 30)
    np.testing.assert_allclose(w.obs_t[-1], 0)
    np.testing.assert_allclose(w.obs_t[0], -14 / 30)
    np.testing.assert_allclose(w.fut_t[[2, 29]], [0.1, 1.0])


def test_rotations_and_metadata_follow_windows():
    trajs = [tr(60), tr(50, drone="08", s_c=None, scene="14")]
    w = make_windows(trajs, 15, 30, 30)
    i = int(np.flatnonzero(w.traj == 1)[2])
    np.testing.assert_allclose(w.obs_R[i], trajs[1].R_G[2:17])
    np.testing.assert_allclose(w.fut_u[i], trajs[1].u[17:47])
    assert not w.metric[i] and w.drone[i] == "08" and w.scene[i] == "14" and w.motion[i] == "linear"
    assert w.metric[0] and w.traj_keys == ["03_01", "14_08"]


def test_irregular_times_raise():
    bad = tr(60)
    bad.times[10] += 0.001
    with pytest.raises(ValueError, match="time step"):
        make_windows([bad], 15, 30, 30)
