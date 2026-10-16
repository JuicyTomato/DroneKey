import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.forecast.baselines import cv
from dronepose.forecast.orientation import constant_angular_velocity, hold_orientation
from dronepose.forecast.physics import physics_predict, tilt_acceleration

N, H = 15, 30
OBS_T = (np.arange(N) - (N - 1)) / 30
FUT_T = np.arange(1, H + 1) / 30
G = 9.81


def test_physics_exact_when_tilt_drives_acceleration():
    tilt = np.radians(20)
    u = np.array([np.sin(tilt), 0, np.cos(tilt)])
    a = np.array([G * np.tan(tilt), 0, 0])
    p0, v0 = np.array([1.0, 2.0, 3.0]), np.array([0.5, -0.2, 0.1])
    # the drone has accelerated since long before the window: the fitted velocity must be the one at the last frame
    traj = lambda t: p0 + v0 * t[:, None] + 0.5 * a * t[:, None] ** 2
    obs, fut = traj(OBS_T)[None], traj(FUT_T)[None]
    obs_u = np.tile(u, (1, N, 1))
    pred = physics_predict(OBS_T, obs, obs_u, FUT_T, k=2, g=G, max_tilt_deg=60)
    # k=2 finite difference is the velocity at the mid point, v0 - a*dt/2: check against that exact expectation
    v_fit = v0 - a / 60
    expected = p0 + v_fit * FUT_T[:, None] + 0.5 * a * FUT_T[:, None] ** 2
    np.testing.assert_allclose(pred[0], expected, atol=1e-9)
    assert np.abs(pred[0] - fut[0]).max() < np.abs(cv(OBS_T, obs, FUT_T, 2)[0] - fut[0]).max()


def test_level_drone_gives_cv():
    obs = np.random.default_rng(0).normal(size=(4, N, 3))
    up = np.tile([0.0, 0.0, 1.0], (4, N, 1))
    np.testing.assert_allclose(physics_predict(OBS_T, obs, up, FUT_T, k=5, g=G, max_tilt_deg=60),
                               cv(OBS_T, obs, FUT_T, 5), atol=1e-12)


def test_physics_inverted_drone_is_finite_and_saturated():
    a = tilt_acceleration(np.array([[0.6, 0, -0.8], [0, 0, -1.0], [0, 0.8, 0.6]]), G, 60)
    np.testing.assert_allclose(a[0], [G * np.tan(np.radians(60)), 0, 0])
    np.testing.assert_allclose(a[1], [0, 0, 0])
    np.testing.assert_allclose(a[2], [0, G * 0.8 / 0.6, 0])      # tilt 53.1 deg < 60: not saturated
    assert np.isfinite(a).all()


def test_hold_orientation_repeats_last():
    R = Rotation.random(4 * N, random_state=1).as_matrix().reshape(4, N, 3, 3)
    out = hold_orientation(R, H)
    assert out.shape == (4, H, 3, 3)
    np.testing.assert_allclose(out, np.repeat(R[:, -1:], H, axis=1))


def test_constant_angular_velocity_exact_on_uniform_rotation():
    axis = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
    R0 = Rotation.random(random_state=2)
    at = lambda t: (R0 * Rotation.from_rotvec(np.radians(30) * t[:, None] * axis)).as_matrix()   # 30 deg/s, body frame
    out = constant_angular_velocity(OBS_T, at(OBS_T)[None], FUT_T)
    np.testing.assert_allclose(out[0], at(FUT_T), atol=1e-9)
