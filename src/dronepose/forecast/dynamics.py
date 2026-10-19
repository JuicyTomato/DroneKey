"""Quadrotor rigid-body dynamics for the trajectory generator (plan step 3.1b, spec §5.1), vectorised over samples.

State: position p and velocity v in G (x right, y forward, z up), attitude R (body FLU -> G), body angular
velocity omega and specific thrust c (m/s^2). First-order actuators drive omega and c toward the commands.

    p' = v,  v' = c R e3 - g e3 - k_d v  [- k_q |v| v - R D R^T v, c <= c_max  (level >= 1)]  [c -> (1+delta) c  (level >= 2)]
    R' = R [omega]x,  omega' = (omega_cmd - omega) / tau_w,  c' = (c_cmd - c) / tau_c

The mismatch level is per sample: level 0 is the model the acro baseline assumes, levels 1 and 2 add terms the
baseline does not know. RK4 on (p, v, omega, c); each stage sees the attitude rotated by the stage's integrated
rate, and R <- R exp([mean omega] dt) with the RK4 average of omega over the step (second order in R, not first).
"""
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

G = 9.81
E3 = np.array([0.0, 0.0, 1.0])


@dataclass
class QuadState:
    p: np.ndarray       # (B,3) G
    v: np.ndarray       # (B,3) G
    R: np.ndarray       # (B,3,3) body FLU -> G
    omega: np.ndarray   # (B,3) body
    c: np.ndarray       # (B,) specific thrust, m/s^2

    def copy(self) -> "QuadState":
        return QuadState(self.p.copy(), self.v.copy(), self.R.copy(), self.omega.copy(), self.c.copy())


@dataclass
class QuadParams:
    k_d: np.ndarray     # (B,) linear drag, 1/s
    tau_w: np.ndarray   # (B,) angular-rate actuator time constant, s
    tau_c: np.ndarray   # (B,) thrust actuator time constant, s
    k_q: np.ndarray     # (B,) quadratic drag, 1/m (level >= 1)
    d_xy: np.ndarray    # (B,2) body-frame rotor drag diag(d_x, d_y, 0), 1/s (level >= 1)
    c_max: np.ndarray   # (B,) thrust saturation, m/s^2 (level >= 1)
    delta: np.ndarray   # (B,) thrust gain error (level >= 2)
    level: np.ndarray   # (B,) int mismatch level


def acceleration(state: QuadState, params: QuadParams, c: np.ndarray | None = None,
                 v: np.ndarray | None = None, R: np.ndarray | None = None) -> np.ndarray:
    c = state.c if c is None else c
    v = state.v if v is None else v
    Rm = state.R if R is None else R
    lv1 = params.level >= 1
    c_eff = np.where(lv1, np.minimum(c, params.c_max), c) * (1 + np.where(params.level >= 2, params.delta, 0.0))
    b3 = Rm[:, :, 2]
    a = c_eff[:, None] * b3 - G * E3 - params.k_d[:, None] * v
    if not lv1.any():
        return a
    v_body = np.einsum("bji,bj->bi", Rm, v)                               # R^T v
    D = np.concatenate([params.d_xy, np.zeros((len(v), 1))], axis=1)
    extra = params.k_q[:, None] * np.linalg.norm(v, axis=1, keepdims=True) * v + np.einsum("bij,bj->bi", Rm, D * v_body)
    return a - np.where(lv1[:, None], extra, 0.0)


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a x b for (..., 3) arrays (np.cross is several times slower on small batches)."""
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: exp([w]x) for (B,3) rotation vectors."""
    th = np.linalg.norm(w, axis=1)
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    A = np.where(small, 1 - th ** 2 / 6, np.sin(ths) / ths)
    Bc = np.where(small, 0.5 - th ** 2 / 24, (1 - np.cos(ths)) / ths ** 2)
    K = hat(w)
    return np.eye(3)[None] + A[:, None, None] * K + Bc[:, None, None] * (K @ K)


def hat(w: np.ndarray) -> np.ndarray:
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = w[..., 2], -w[..., 1], w[..., 0]
    return K


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Q = U @ Vt
    neg = np.linalg.det(Q) < 0
    if neg.any():
        U[neg, :, -1] *= -1
        Q = U @ Vt
    return Q


def step(state: QuadState, omega_cmd: np.ndarray, c_cmd: np.ndarray, params: QuadParams, dt: float) -> QuadState:
    tw, tc = params.tau_w[:, None], params.tau_c

    R0 = state.R

    def deriv(v, w, c, R):
        return acceleration(state, params, c=c, v=v, R=R), (omega_cmd - w) / tw, (c_cmd - c) / tc

    v0, w0, c0 = state.v, state.omega, state.c
    a1, dw1, dc1 = deriv(v0, w0, c0, R0)
    v2, w2, c2 = v0 + a1 * dt / 2, w0 + dw1 * dt / 2, c0 + dc1 * dt / 2
    a2, dw2, dc2 = deriv(v2, w2, c2, R0 @ so3_exp((w0 + w2) * dt / 4))
    v3, w3, c3 = v0 + a2 * dt / 2, w0 + dw2 * dt / 2, c0 + dc2 * dt / 2
    a3, dw3, dc3 = deriv(v3, w3, c3, R0 @ so3_exp((w0 + w3) * dt / 4))
    v4, w4, c4 = v0 + a3 * dt, w0 + dw3 * dt, c0 + dc3 * dt
    a4, dw4, dc4 = deriv(v4, w4, c4, R0 @ so3_exp((w0 + 2 * w2 + 2 * w3 + w4) * dt / 6))
    p = state.p + dt / 6 * (v0 + 2 * v2 + 2 * v3 + v4)
    v = v0 + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
    w = w0 + dt / 6 * (dw1 + 2 * dw2 + 2 * dw3 + dw4)
    c = c0 + dt / 6 * (dc1 + 2 * dc2 + 2 * dc3 + dc4)
    return QuadState(p=p, v=v, R=R0 @ so3_exp((w0 + 2 * w2 + 2 * w3 + w4) * dt / 6), omega=w, c=c)


CmdFn = Callable[[QuadState, int], tuple[np.ndarray, np.ndarray]]


def simulate(state: QuadState, cmd_fn: CmdFn, params: QuadParams, n_steps: int, dt: float, out_every: int,
             reorth_every: int = 60) -> tuple[list[QuadState], list[tuple[np.ndarray, np.ndarray]]]:
    """Integrate n_steps; states (and the commands applied from them) at steps 0, out_every, 2 out_every, ..."""
    states, cmds = [], []
    s = state.copy()
    for k in range(n_steps + 1):
        cmd = cmd_fn(s, k) if k < n_steps or k % out_every == 0 else None
        if k % out_every == 0:
            states.append(s.copy())
            cmds.append((np.array(cmd[0], float), np.array(cmd[1], float)))
        if k == n_steps:
            break
        s = step(s, np.asarray(cmd[0], float), np.asarray(cmd[1], float), params, dt)
        if (k + 1) % reorth_every == 0:
            s = replace(s, R=orthonormalize(s.R))
    return states, cmds
