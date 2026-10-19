import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.forecast.dynamics import G, QuadParams, QuadState, acceleration, orthonormalize, simulate, step

DT = 1 / 600


def state(B=1, R=None, v=None, omega=None, c=G):
    R = np.repeat(np.eye(3)[None], B, 0) if R is None else np.broadcast_to(R, (B, 3, 3)).copy()
    return QuadState(p=np.zeros((B, 3)), v=np.zeros((B, 3)) if v is None else np.broadcast_to(v, (B, 3)).copy(),
                     R=R, omega=np.zeros((B, 3)) if omega is None else np.broadcast_to(omega, (B, 3)).copy(),
                     c=np.full(B, float(c)))


def params(B=1, level=0, k_d=0.0, tau_w=0.04, tau_c=0.03, k_q=0.0, d_xy=(0.0, 0.0), c_max=2 * G, delta=0.0):
    f = lambda x: np.full(B, float(x))
    return QuadParams(k_d=f(k_d), tau_w=f(tau_w), tau_c=f(tau_c), k_q=f(k_q), d_xy=np.tile(d_xy, (B, 1)).astype(float),
                      c_max=f(c_max), delta=f(delta), level=np.full(B, level, int))


def hold(omega_cmd, c_cmd):
    return lambda s, k: (np.broadcast_to(omega_cmd, s.omega.shape), np.broadcast_to(c_cmd, s.c.shape))


def run(s, prm, omega_cmd, c_cmd, seconds, dt=DT, out_every=1):
    return simulate(s, hold(np.asarray(omega_cmd, float), float(c_cmd)), prm, int(round(seconds / dt)), dt, out_every)


def test_hover_stays_still():
    states, _ = run(state(), params(), [0, 0, 0], G, 1.0, out_every=600)
    assert np.abs(states[-1].p).max() < 1e-12


def test_free_fall():
    states, _ = run(state(c=0.0), params(), [0, 0, 0], 0.0, 1.0, out_every=60)
    for i, s in enumerate(states):
        t = i * 60 * DT
        assert abs(s.p[0, 2] + G * t ** 2 / 2) < 1e-9


def test_constant_rate_rotation():
    w = np.radians([120.0, -40.0, 60.0])
    R0 = Rotation.from_euler("zyx", [30, 10, -5], degrees=True).as_matrix()
    states, _ = run(state(R=R0, omega=w), params(), w, G, 1.0, out_every=60)
    n0 = R0 @ w
    for i, s in enumerate(states):
        t = i * 60 * DT
        np.testing.assert_allclose(s.R[0], R0 @ Rotation.from_rotvec(w * t).as_matrix(), atol=1e-9)
        np.testing.assert_allclose(s.R[0] @ s.omega[0], n0, atol=1e-9)


def test_constant_tilt_accelerates_g_tan():
    th = np.radians(20)
    R = Rotation.from_euler("y", th).as_matrix()    # body x forward tilted down: thrust leans toward +x
    a = acceleration(state(R=R, c=G / np.cos(th)), params())
    np.testing.assert_allclose(a[0], [G * np.tan(th), 0, 0], atol=1e-12)
    states, _ = run(state(R=R, c=G / np.cos(th)), params(), [0, 0, 0], G / np.cos(th), 1.0, out_every=600)
    np.testing.assert_allclose(states[-1].v[0], [G * np.tan(th), 0, 0], atol=1e-9)


def test_terminal_velocity_with_drag():
    th = np.radians(20)
    R = Rotation.from_euler("y", th).as_matrix()
    c = G / np.cos(th)
    states, _ = run(state(R=R, c=c), params(k_d=0.5), [0, 0, 0], c, 20.0, dt=1 / 120, out_every=2400)
    v = states[-1].v[0]
    assert abs(np.linalg.norm(v[:2]) - G * np.tan(th) / 0.5) < 1e-3
    assert abs(v[2]) < 1e-9


def test_rotation_stays_orthonormal():
    w = np.radians([300.0, 0.0, 0.0])
    states, _ = run(state(omega=w), params(), w, G, 10.0, out_every=6000)
    R = states[-1].R[0]
    assert np.abs(R.T @ R - np.eye(3)).max() < 1e-9
    assert abs(np.linalg.det(R) - 1) < 1e-9


def test_orthonormalize_projects_to_so3():
    R = Rotation.from_euler("zyx", [40, -20, 70], degrees=True).as_matrix()[None] + 1e-4
    Q = orthonormalize(R)[0]
    assert np.abs(Q.T @ Q - np.eye(3)).max() < 1e-12 and abs(np.linalg.det(Q) - 1) < 1e-12
    assert np.abs(Q - R[0]).max() < 1e-3


def test_level_masks():
    R = Rotation.from_euler("zyx", [30, 15, -10], degrees=True).as_matrix()
    v = np.array([3.0, -2.0, 1.0])
    s = state(B=3, R=R, v=v, c=2.2 * G)
    prm = params(B=3, k_d=0.5, k_q=0.04, d_xy=(0.2, 0.1), c_max=2.0 * G, delta=0.05)
    prm.level[:] = [0, 1, 2]
    a = acceleration(s, prm)
    e3 = np.array([0, 0, 1.0])
    base = -G * e3 - 0.5 * v
    rotor = R @ np.diag([0.2, 0.1, 0.0]) @ R.T @ v
    np.testing.assert_allclose(a[0], 2.2 * G * R[:, 2] + base, atol=1e-12)
    lv1 = 2.0 * G * R[:, 2] + base - 0.04 * np.linalg.norm(v) * v - rotor
    np.testing.assert_allclose(a[1], lv1, atol=1e-12)
    np.testing.assert_allclose(a[2], lv1 + 0.05 * 2.0 * G * R[:, 2], atol=1e-12)


def test_actuator_first_order():
    w = np.radians([200.0, 0.0, 0.0])
    states, _ = run(state(), params(tau_w=0.04), w, G, 0.04, out_every=24)
    np.testing.assert_allclose(states[-1].omega[0, 0] / w[0], 1 - np.exp(-1), atol=1e-6)


def test_simulate_outputs_and_commands():
    states, cmds = run(state(B=2), params(B=2), [0, 0, 0], G, 1.0, out_every=20)
    assert len(states) == len(cmds) == 31
    assert cmds[0][0].shape == (2, 3) and cmds[0][1].shape == (2,)


def test_step_is_vectorised():
    s = state(B=4, omega=np.radians([10.0, 20.0, 30.0]))
    out = step(s, np.zeros((4, 3)), np.full(4, G), params(B=4), DT)
    assert out.p.shape == (4, 3) and out.R.shape == (4, 3, 3) and out.c.shape == (4,)


def test_rotating_thrust_axis_is_second_order():
    # thrust axis spinning about x at 300 deg/s: halving dt must cut the velocity error ~4x (R inside the RK4 stages)
    w = np.radians([300.0, 0.0, 0.0])
    ref, _ = run(state(omega=w), params(), w, G, 0.5, dt=1 / 4800, out_every=2400)
    errs = []
    for dt in (1 / 300, 1 / 600):
        s, _ = run(state(omega=w), params(), w, G, 0.5, dt=dt, out_every=int(round(0.5 / dt)))
        errs.append(np.abs(s[-1].v - ref[-1].v).max())
    assert errs[1] < errs[0] / 3
    assert errs[1] < 1e-4
