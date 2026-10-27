"""Generalised acro physics baseline (spec §5.4): rigid body with the commands held, estimated from the observations.

Estimate (last k frames): position, velocity, acceleration from a degree-2 fit; rate from the consecutive relative
rotations, least squares in G under the prediction model itself (vertical part constant, horizontal part
omega_h(0) e^(-lambda tau): with lambda = 0 the mean of the rotation vectors); specific thrust c and linear drag k_d from
a + g e3 = c b3 - k_d v at the centre of the window (where the degree-2 fit is most accurate), k_d in [0, 2], or a
fixed k_d. Future: the rate in G splits into a vertical part, held, and a horizontal part decaying as e^(-lambda t);
thrust constant (acro) or holding the vertical speed (angle / P-mode). lambda = 0 + constant thrust = pure acro,
the frozen-sticks model of requirement 3. Integrated with the level-0 dynamics (actuators settled: omega = command).
"""
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.forecast.baselines import _fit
from dronepose.forecast.dynamics import E3, G, QuadParams, QuadState, simulate, step

DT = 1 / 600
KD_RANGE = (0.0, 2.0)
C_RANGE = (0.0, 3 * G)


def estimate_state(obs_t, obs_p, obs_R, k: int, kd_mode, lam=0.0) -> dict:
    obs_t = np.asarray(obs_t, float)
    obs_R = np.asarray(obs_R, float)
    W = len(obs_R)
    coef = _fit(obs_t, obs_p, k, 2)                                    # (W,3,3): p0, v0, a0/2 at the last frame
    tau = obs_t[-k:] - obs_t[-1]
    Rk = obs_R[:, -k:]
    rel = np.einsum("wkji,wkjl->wkil", Rk[:, :-1], Rk[:, 1:])          # R_i^T R_{i+1}: body-frame increments
    rv = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(W, k - 1, 3)
    y_G = np.einsum("wkij,wkj->wki", Rk[:, :-1], rv / np.diff(tau)[None, :, None])   # R_i (increment) = rate in G
    lam = np.broadcast_to(np.asarray(lam, float), (W,))
    decay = np.exp(-lam[:, None] * (tau[1:] + tau[:-1])[None] / 2)                   # model at the mid-times (tau <= 0)
    w_h = np.einsum("wk,wki->wi", decay, y_G * (1 - E3)) / np.sum(decay ** 2, 1)[:, None]
    w_G = y_G[..., 2].mean(1)[:, None] * E3 + w_h
    omega = np.einsum("wji,wj->wi", obs_R[:, -1], w_G)                                  # body rate at the last frame
    # thrust and drag at the window centre
    tc = tau.mean()
    v_c = coef[:, 1] + 2 * coef[:, 2] * tc
    a_c = 2 * coef[:, 2]
    mid = (k - 1) / 2
    b3 = Rk[:, int(np.floor(mid)), :, 2] + Rk[:, int(np.ceil(mid)), :, 2]
    b3 /= np.linalg.norm(b3, axis=1, keepdims=True)
    y = a_c + G * E3
    if isinstance(kd_mode, str):
        if kd_mode != "estimate":
            raise ValueError(f"kd_mode {kd_mode!r}: 'estimate' or a number")
        bb, bv, vv = np.ones(W), np.sum(b3 * v_c, 1), np.sum(v_c * v_c, 1)
        by, vy = np.sum(b3 * y, 1), np.sum(v_c * y, 1)
        det = bb * vv - bv * bv
        ok = det > 1e-9 * np.maximum(vv, 1e-12)
        kd = np.where(ok, (bv * by - bb * vy) / np.where(ok, det, 1.0), 0.0)   # unidentifiable (v ∥ b3, hover): no drag
        kd = np.clip(kd, *KD_RANGE)
    else:
        kd = np.broadcast_to(np.asarray(kd_mode, float), (W,)).copy()
    c = np.clip(np.sum(b3 * (y + kd[:, None] * v_c), 1), *C_RANGE)
    return {"p0": coef[:, 0], "v0": coef[:, 1], "a0": 2 * coef[:, 2], "R0": obs_R[:, -1], "omega": omega,
            "c": c, "k_d": kd}


def _level0(B: int, k_d: np.ndarray) -> QuadParams:
    one = np.ones(B)
    return QuadParams(k_d=k_d, tau_w=one, tau_c=one, k_q=0 * one, d_xy=np.zeros((B, 2)), c_max=C_RANGE[1] * one,
                      delta=0 * one, level=np.zeros(B, int))


def acro_predict(obs_t, obs_p, obs_R, fut_t, k: int, lam, thrust, kd_mode) -> np.ndarray:
    est = estimate_state(obs_t, obs_p, obs_R, k, kd_mode, lam)
    return acro_rollout(est, fut_t, lam, thrust, obs_t[-1])


def acro_rollout(est: dict, fut_t, lam, thrust, t_last: float) -> np.ndarray:
    """Integrate the estimated states (estimate_state) to the future times; t_last = time of the last observation."""
    W = len(est["c"])
    lam = np.broadcast_to(np.asarray(lam, float), (W,))
    hold = np.broadcast_to(np.asarray(thrust) == "hold_vz", (W,))
    if not np.isin(np.asarray(thrust), ("constant", "hold_vz")).all():
        raise ValueError(f"thrust {thrust!r}: 'constant' or 'hold_vz'")
    R0 = est["R0"]
    wG = np.einsum("wij,wj->wi", R0, est["omega"])
    w_vert = wG * E3
    w_h0 = wG - w_vert
    prm = _level0(W, est["k_d"])
    s = QuadState(p=est["p0"].copy(), v=est["v0"].copy(), R=R0.copy(), omega=est["omega"].copy(), c=est["c"].copy())
    out = np.empty((W, len(fut_t), 3))
    t, t_prev = 0.0, t_last
    for j, tf in enumerate(np.asarray(fut_t, float)):
        for _ in range(int(round((tf - t_prev) / DT))):
            w_b = np.einsum("wji,wj->wi", s.R, w_vert + w_h0 * np.exp(-lam * t)[:, None])
            c_hold = np.clip((G + est["k_d"] * s.v[:, 2]) / np.maximum(s.R[:, 2, 2], 0.2), *C_RANGE)
            c = np.where(hold, c_hold, est["c"])
            s = step(replace(s, omega=w_b, c=c), w_b, c, prm, DT)       # omega = command: rate held over the step
            t += DT
        t_prev = tf
        out[:, j] = s.p
    return out


def acro_oracle(state: QuadState, params: QuadParams, omega_cmd, c_cmd, fut_t) -> np.ndarray:
    """The level-0 dynamics with the true state, actuators and frozen commands; positions relative to state.p."""
    B = len(state.c)
    prm = replace(params, level=np.zeros(B, int))
    w, c = np.asarray(omega_cmd, float), np.asarray(c_cmd, float)
    fut_t = np.asarray(fut_t, float)
    every = int(round((fut_t[1] - fut_t[0]) / DT)) if len(fut_t) > 1 else int(round(fut_t[0] / DT))
    n = int(round(fut_t[-1] / DT))
    states, _ = simulate(state, lambda s, k: (w, c), prm, n, DT, every)
    idx = np.round(fut_t / DT / every).astype(int)
    return np.stack([states[i].p for i in idx], 1) - state.p[:, None]
