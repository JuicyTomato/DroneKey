import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.forecast.controller import (CtrlGains, angle_mode, attitude_rate_cmd, heading_of, p_mode,
                                           thrust_from_accel)
from dronepose.forecast.dynamics import G, QuadParams, QuadState, simulate

DT = 1 / 300


def state(B=1, R=None, v=None):
    R = np.repeat(np.eye(3)[None], B, 0) if R is None else np.broadcast_to(R, (B, 3, 3)).copy()
    return QuadState(p=np.zeros((B, 3)), v=np.zeros((B, 3)) if v is None else np.broadcast_to(v, (B, 3)).copy(),
                     R=R, omega=np.zeros((B, 3)), c=np.full(B, G))


def params(B=1, k_d=0.6):
    f = lambda x: np.full(B, float(x))
    return QuadParams(k_d=f(k_d), tau_w=f(0.04), tau_c=f(0.03), k_q=f(0), d_xy=np.zeros((B, 2)), c_max=f(2 * G),
                      delta=f(0), level=np.zeros(B, int))


def gains(B=1, tau_att=0.1, max_tilt=35.0, max_rate=300.0, k_v=2.0):
    f = lambda x: np.full(B, float(x))
    return CtrlGains(k_att=f(1 / tau_att), k_v=f(k_v), k_vz=f(2.0), max_tilt=f(np.radians(max_tilt)),
                     max_rate=f(np.radians(max_rate)))


def tilt_deg(R):
    return np.degrees(np.arccos(np.clip(R[..., 2, 2], -1, 1)))


def fly(mode, sticks, seconds, B=1, k_d=0.6, out_every=3, **kw):
    prm, ctrl = params(B, k_d), gains(B, **kw)
    fn = angle_mode if mode == "angle" else p_mode
    states, _ = simulate(state(B), lambda s, k: fn(s, sticks, prm, ctrl), prm, int(round(seconds / DT)), DT, out_every)
    return states


def course_rate(states, dt):
    v = np.stack([s.v[0] for s in states])
    psi = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    return np.gradient(psi, dt)


def test_singular_thrust_keeps_previous_axis():
    v = np.array([[1.0, -2.0, 0.5]])
    k_d = np.array([0.6])
    prev = np.array([[0.1, 0.2, 0.97]]) / np.linalg.norm([0.1, 0.2, 0.97])
    c, b3 = thrust_from_accel(-G * np.array([[0, 0, 1.0]]) - k_d[:, None] * v, v, k_d, prev)
    assert np.isfinite(c).all() and np.isfinite(b3).all()
    np.testing.assert_allclose(c, 0.3 * G)
    np.testing.assert_allclose(b3, prev)


def test_thrust_from_accel_flatness():
    v, k_d = np.array([[2.0, 0.0, 0.0]]), np.array([0.5])
    a = np.array([[1.0, 0.5, 0.2]])
    c, b3 = thrust_from_accel(a, v, k_d, np.array([[0, 0, 1.0]]))
    np.testing.assert_allclose(c[:, None] * b3, a + G * np.array([0, 0, 1.0]) + 0.5 * v, atol=1e-12)


def test_angle_mode_converges_to_target_tilt():
    sticks = {"tilt_xy": np.radians([[15.0, 0.0]]), "yaw_rate": np.zeros(1), "vz": np.zeros(1)}
    states = fly("angle", sticks, 1.5)
    t = np.arange(len(states)) * 3 * DT
    tilts = tilt_deg(np.stack([s.R[0] for s in states]))
    after = t >= 0.5
    assert np.abs(tilts[after] - 15).max() < 0.5
    assert abs(states[-1].v[0, 2]) < 0.05
    assert states[-1].v[0, 0] > 0     # forward tilt accelerates along the nose (body x = G x at R = I)


def test_angle_mode_level_flight_goes_straight():
    th = np.radians(20)
    sticks = {"tilt_xy": np.array([[th, 0.0]]), "yaw_rate": np.zeros(1), "vz": np.zeros(1)}
    states = fly("angle", sticks, 10.0, out_every=30)
    rate = course_rate(states, 30 * DT)
    assert np.abs(rate[-10:]).max() < 1e-3
    v_term = G * np.tan(th) / 0.6
    assert abs(np.linalg.norm(states[-1].v[0, :2]) - v_term) / v_term < 0.02


def test_angle_mode_with_yaw_rate_turns():
    sticks = {"tilt_xy": np.radians([[15.0, 0.0]]), "yaw_rate": np.radians([20.0]), "vz": np.zeros(1)}
    states = fly("angle", sticks, 10.0, out_every=30)
    rate = np.degrees(course_rate(states, 30 * DT))
    np.testing.assert_allclose(rate[-10:], 20.0, atol=0.5)


def test_p_mode_converges_to_target_velocity():
    sticks = {"v_xy": np.array([[3.0, 0.0]]), "yaw_rate": np.zeros(1), "vz": np.zeros(1)}
    states = fly("pmode", sticks, 3.0)
    np.testing.assert_allclose(states[-1].v[0], [3.0, 0.0, 0.0], atol=0.05)
    sticks["yaw_rate"] = np.radians([30.0])
    sticks["vz"] = np.array([1.0])
    states = fly("pmode", sticks, 6.0, out_every=30)
    rate = np.degrees(course_rate(states, 30 * DT))
    np.testing.assert_allclose(rate[-10:], 30.0, atol=1.0)                  # arc in the plane
    np.testing.assert_allclose(states[-1].v[0, 2], 1.0, atol=0.05)          # climbing: helix
    np.testing.assert_allclose(np.linalg.norm(states[-1].v[0, :2]), 3.0, atol=0.1)


def test_lee_rate_is_saturated():
    R = Rotation.from_euler("x", 90, degrees=True).as_matrix()[None]
    w = attitude_rate_cmd(R, np.array([[0, 0, 1.0]]), np.zeros(1), np.array([10.0]), np.radians([200.0]))
    np.testing.assert_allclose(np.linalg.norm(w, axis=1), np.radians(200.0))


def test_lee_rate_zero_at_target():
    R = Rotation.from_euler("zyx", [40, 10, -5], degrees=True).as_matrix()[None]
    w = attitude_rate_cmd(R, R[:, :, 2], heading_of(R), np.array([10.0]), np.radians([200.0]))
    np.testing.assert_allclose(w, 0, atol=1e-12)


def test_heading_of():
    R = Rotation.from_euler("z", [[0], [90], [-135]], degrees=True).as_matrix()
    np.testing.assert_allclose(np.degrees(heading_of(R)), [0, 90, -135], atol=1e-9)


def test_yaw_independent_of_velocity():
    sticks = {"v_xy": np.array([[0.0, 3.0]]), "yaw_rate": np.zeros(1), "vz": np.zeros(1)}
    states = fly("pmode", sticks, 3.0)
    np.testing.assert_allclose(states[-1].v[0], [0.0, 3.0, 0.0], atol=0.05)   # left of the nose (body y = G y)
    heads = np.degrees(heading_of(np.stack([s.R[0] for s in states])))
    assert np.abs(heads).max() < 0.5


def test_p_mode_hard_descent_keeps_thrust_up():
    # commanded fall (final review, focus 1): climbing at 10 m/s with the stick at -6 m/s makes a_des + g e3 point
    # down; the thrust must stay up at the 0.3 g floor, never flip the drone to push it down
    B = 2
    s = state(B, v=np.array([[0.0, 0.0, 10.0], [5.0, 0.0, 8.0]]))
    sticks = {"v_xy": np.array([[0.0, 0.0], [5.0, 0.0]]), "yaw_rate": np.zeros(B), "vz": np.full(B, -6.0)}
    w, c = p_mode(s, sticks, params(B), gains(B))
    assert np.isfinite(w).all() and np.isfinite(c).all()
    np.testing.assert_allclose(c[0], 0.3 * G)
    assert np.abs(w[0]).max() < 1e-9                                         # level and on target: no rotation
    prm, ctrl = params(B), gains(B, max_tilt=35.0)
    states, _ = simulate(s, lambda st, k: p_mode(st, sticks, prm, ctrl), prm, 300, DT, 3)
    assert max(tilt_deg(x.R).max() for x in states) < 35.0 + 5.0
