import copy

import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.forecast.acro import acro_oracle, acro_predict, estimate_state
from dronepose.forecast.baselines import cv
from dronepose.forecast.config import load_generator_config
from dronepose.forecast.dynamics import G, QuadParams, QuadState, orthonormalize, simulate
from dronepose.forecast.generator import generate
from dronepose.forecast.windows import freeze_windows

CFG = load_generator_config()
FAST = copy.deepcopy(CFG)
FAST["generator"]["warmup_s"] = [1.0, 1.4]
FI = 29
N, H = 15, 30


def params(B, k_d=0.6, tau=0.04):
    f = lambda x: np.full(B, float(x))
    return QuadParams(k_d=f(k_d), tau_w=f(tau), tau_c=f(tau), k_q=f(0), d_xy=np.zeros((B, 2)), c_max=f(3 * G),
                      delta=f(0), level=np.zeros(B, int))


def const_flight(R0, omega, c, k_d=0.6, v0=(0.0, 0.0, 0.0), seconds=2.0):
    """Level-0 flight with constant body rate and thrust (actuators already at the command): 30 FPS samples."""
    B = len(R0)
    s = QuadState(p=np.zeros((B, 3)), v=np.tile(v0, (B, 1)).astype(float), R=R0.copy(), omega=omega.copy(),
                  c=np.asarray(c, float).copy())
    states, _ = simulate(s, lambda st, k: (omega, np.asarray(c, float)), params(B, k_d), int(seconds * 600), 1 / 600, 20)
    p = np.stack([x.p for x in states], 1)
    R = np.stack([x.R for x in states], 1)
    return p, R


def windows_from(p, R, end):
    obs_t = (np.arange(N) - (N - 1)) / 30
    fut_t = np.arange(1, H + 1) / 30
    last = p[:, end:end + 1]
    return obs_t, fut_t, p[:, end - N + 1:end + 1] - last, R[:, end - N + 1:end + 1], p[:, end + 1:end + 1 + H] - last


def test_oracle_matches_generator_level0():
    out = generate(16, seed=31, level=0, cfg=FAST, setup_overrides={"mode": "acro"}, record_hf=False)
    I = out.internals
    B = len(out.trajs)
    m = [t.meta for t in out.trajs]
    state = QuadState(p=I["p"][:, FI].copy(), v=np.array([x["fz_v"] for x in m]), R=I["R"][:, FI].copy(),
                      omega=np.array([x["fz_omega"] for x in m]), c=np.array([x["fz_c"] for x in m]))
    prm = QuadParams(k_d=np.array([x["k_d"] for x in m]), tau_w=np.array([x["tau_w"] for x in m]),
                     tau_c=np.array([x["tau_c"] for x in m]), k_q=np.zeros(B), d_xy=np.zeros((B, 2)),
                     c_max=np.full(B, 3 * G), delta=np.zeros(B), level=np.zeros(B, int))
    pred = acro_oracle(state, prm, np.array([x["fz_omega_cmd"] for x in m]), np.array([x["fz_c_cmd"] for x in m]),
                       np.arange(1, H + 1) / 30)
    truth = I["p"][:, FI + 1:] - I["p"][:, FI:FI + 1]
    assert np.abs(pred - truth).max() < 1e-4


def test_lambda_reproduces_angle_mode():
    # angle mode with a fast actuator (tau_w = 5 ms): the horizontal rate decays as exp(-t / tau_att), so the
    # generalised acro with lambda = 1 / tau_att and hold_vz reproduces it
    from dronepose.forecast.controller import CtrlGains, angle_mode
    B, tau_att = 3, 0.1
    prm = params(B, k_d=0.6, tau=0.005)
    ctrl = CtrlGains(k_att=np.full(B, 1 / tau_att), k_v=np.full(B, 2.0), k_vz=np.full(B, 2.0),
                     max_tilt=np.full(B, np.radians(45)), max_rate=np.full(B, np.radians(300)))
    sticks = {"tilt_xy": np.radians([[5.0, 0.0], [10.0, -5.0], [0.0, 8.0]]), "yaw_rate": np.radians([0.0, 20.0, -10.0]),
              "vz": np.zeros(B)}
    late = {"tilt_xy": np.radians([[20.0, 5.0], [-5.0, 10.0], [12.0, 0.0]]), "yaw_rate": sticks["yaw_rate"], "vz": np.zeros(B)}
    s = QuadState(p=np.zeros((B, 3)), v=np.tile([2.0, 0.0, 0.0], (B, 1)), R=np.repeat(np.eye(3)[None], B, 0),
                  omega=np.zeros((B, 3)), c=np.full(B, G))
    switch = 600                                    # new tilt target at 1 s; frozen 0.1 s later, mid-transient
    states, _ = simulate(s, lambda st, k: angle_mode(st, sticks if k < switch else late, prm, ctrl), prm, 2400,
                         1 / 600, 20)
    p, R = np.stack([x.p for x in states], 1), np.stack([x.R for x in states], 1)
    end = switch // 20 + 3
    obs_t, fut_t, obs_p, obs_R, fut_p = windows_from(p, R, end)
    pred = acro_predict(obs_t, obs_p, obs_R, fut_t, 3, 1 / tau_att, "hold_vz", 0.6)
    pure = acro_predict(obs_t, obs_p, obs_R, fut_t, 3, 0.0, "constant", 0.6)
    err = np.linalg.norm(pred[:, -1] - fut_p[:, -1], axis=-1)
    # with the true state at the freeze the error is < 0.05 m (model + 5 ms actuator); estimating the state from
    # 3 frames 0.1 s into the transient adds ~0.02 m (degree-2 velocity fit)
    assert err.max() < 0.1
    assert (err < 0.5 * np.linalg.norm(pure[:, -1] - fut_p[:, -1], axis=-1)).all()


def test_lambda_helps_on_generated_angle_mode():
    # generated angle mode (actuators 20-60 ms): lambda = 1 / tau_att still beats pure acro on average
    out = generate(16, seed=32, level=0, cfg=FAST, setup_overrides={"mode": "angle", "stick_noise": 0.0, "s": 0.3})
    w = freeze_windows(out.trajs, N, H, 30)
    lam = np.array([1 / t.meta["tau_att"] for t in out.trajs])
    kd = np.array([t.meta["k_d"] for t in out.trajs])
    gen = acro_predict(w.obs_t, w.obs_p, w.obs_R.astype(float), w.fut_t, 3, lam, "hold_vz", kd)
    pure = acro_predict(w.obs_t, w.obs_p, w.obs_R.astype(float), w.fut_t, 3, 0.0, "constant", kd)
    e = lambda x: np.linalg.norm(x[:, -1] - w.fut_p[:, -1], axis=-1).mean()
    assert e(gen) < e(pure)


def test_beats_cv_on_constant_rate_curve():
    tilt = np.radians(25)
    R0 = Rotation.from_euler("zx", [[10, 25], [80, -25], [-40, 25]], degrees=True).as_matrix()
    yaw = np.radians(40.0)
    omega = np.stack([R.T @ np.array([0, 0, yaw]) for R in R0])     # body rate whose world axis is the vertical
    c = np.full(3, G / np.cos(tilt))
    p, R = const_flight(R0, omega, c, k_d=0.5, v0=(1.0, 2.0, 0.0), seconds=3.0)
    obs_t, fut_t, obs_p, obs_R, fut_p = windows_from(p, R, 50)
    acro = acro_predict(obs_t, obs_p, obs_R, fut_t, 5, 0.0, "constant", "estimate")
    e_acro = np.linalg.norm(acro[:, -1] - fut_p[:, -1], axis=-1)
    e_cv = np.linalg.norm(cv(obs_t, obs_p, fut_t, 2)[:, -1] - fut_p[:, -1], axis=-1)
    assert (e_acro < 0.5 * e_cv).all()


def test_estimate_state_exact_on_noise_free():
    R0 = Rotation.from_euler("zy", [[30, 20], [-60, 15]], degrees=True).as_matrix()
    omega = np.array([[0.0, 0.0, 0.3], [0.0, 0.0, -0.5]])            # yaw about the body axis: b3 fixed in G
    c = np.array([G / np.cos(np.radians(20)), G / np.cos(np.radians(15))])
    p, R = const_flight(R0, omega, c, k_d=0.6, seconds=2.0)
    obs_t, _, obs_p, obs_R, _ = windows_from(p, R, 40)
    est = estimate_state(obs_t, obs_p, obs_R, 5, "estimate")
    np.testing.assert_allclose(est["omega"], omega, rtol=0.01, atol=1e-6)
    np.testing.assert_allclose(est["c"], c, rtol=0.01)
    np.testing.assert_allclose(est["k_d"], 0.6, rtol=0.01)
    fixed = estimate_state(obs_t, obs_p, obs_R, 5, 0.6)
    np.testing.assert_allclose(fixed["k_d"], 0.6)
    np.testing.assert_allclose(fixed["c"], c, rtol=0.01)


def test_inverted_drone_finite():
    R0 = Rotation.from_euler("x", [[170.0], [100.0]], degrees=True).as_matrix()
    p, R = const_flight(R0, np.array([[0.5, 0.0, 0.0], [0.0, 0.3, 0.0]]), np.full(2, G), seconds=1.5)
    obs_t, fut_t, obs_p, obs_R, _ = windows_from(p, R, 20)
    for lam, thrust in ((0.0, "constant"), (5.0, "hold_vz")):
        assert np.isfinite(acro_predict(obs_t, obs_p, obs_R, fut_t, 5, lam, thrust, "estimate")).all()


def test_per_window_lambda_and_thrust():
    R0 = Rotation.from_euler("zy", [[30, 20], [-60, 15]], degrees=True).as_matrix()
    p, R = const_flight(R0, np.array([[0.2, 0.1, 0.3], [0.1, -0.2, 0.0]]), np.full(2, G * 1.05), seconds=2.0)
    obs_t, fut_t, obs_p, obs_R, _ = windows_from(p, R, 40)
    both = acro_predict(obs_t, obs_p, obs_R, fut_t, 5, np.array([0.0, 8.0]), np.array(["constant", "hold_vz"]), 0.6)
    a = acro_predict(obs_t, obs_p, obs_R, fut_t, 5, 0.0, "constant", 0.6)
    b = acro_predict(obs_t, obs_p, obs_R, fut_t, 5, 8.0, "hold_vz", 0.6)
    np.testing.assert_allclose(both[0], a[0])
    np.testing.assert_allclose(both[1], b[1])
