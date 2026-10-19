"""Flight controller of the simulated drone (spec §5.2): geometric attitude loop on SO(3) (Lee), angle mode, P-mode.

The same loops serve the pilot in the past and the frozen sticks in the future. Thrust axis from flatness,
c b3 = a + g e3 + k_d v; heading = body x projected on the horizontal plane, independent of the motion.
Sticks are in the heading frame: x forward (nose), y left.
"""
from dataclasses import dataclass

import numpy as np

from dronepose.forecast.dynamics import E3, G, QuadParams, QuadState, cross3

C_MIN, C_MAX = 0.3 * G, 2.5 * G


@dataclass
class CtrlGains:
    k_att: np.ndarray     # (B,) attitude gain = 1 / tau_att, 1/s
    k_v: np.ndarray       # (B,) horizontal velocity gain (P-mode), 1/s
    k_vz: np.ndarray      # (B,) vertical velocity gain, 1/s
    max_tilt: np.ndarray  # (B,) rad
    max_rate: np.ndarray  # (B,) rad/s, norm of the commanded body rate


def heading_of(R: np.ndarray) -> np.ndarray:
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def _heading_axes(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, s, z = np.cos(psi), np.sin(psi), np.zeros_like(psi)
    return np.stack([c, s, z], -1), np.stack([-s, c, z], -1)


def _unit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(x, axis=-1)
    return x / np.where(n > 0, n, 1.0)[..., None], n


def thrust_from_accel(a_des: np.ndarray, v: np.ndarray, k_d: np.ndarray, prev_b3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """c_cmd and the desired thrust axis from c b3 = a_des + g e3 + k_d v; near-zero thrust keeps the previous axis."""
    b3, n = _unit(a_des + G * E3 + k_d[:, None] * v)
    singular = n < C_MIN
    return np.where(singular, C_MIN, np.clip(n, C_MIN, C_MAX)), np.where(singular[:, None], prev_b3, b3)


def attitude_rate_cmd(R: np.ndarray, b3_des: np.ndarray, heading_des: np.ndarray, k_att: np.ndarray,
                      max_rate: np.ndarray, yaw_rate: np.ndarray | None = None) -> np.ndarray:
    """Body rate command -k_att e_R (+ yaw-rate feed-forward about the vertical), norm saturated at max_rate."""
    _, left = _heading_axes(heading_des)
    b1, n = _unit(cross3(left, b3_des))       # in the plane of the heading and the vertical, normal to b3
    b1 = np.where((n < 1e-6)[:, None], R[:, :, 0], b1)   # thrust axis along the heading's left: keep the body x
    b2 = cross3(b3_des, b1)
    R_des = np.stack([b1, b2, b3_des], axis=-1)
    M = np.einsum("bji,bjk->bik", R_des, R) - np.einsum("bji,bjk->bik", R, R_des)
    e_R = 0.5 * np.stack([M[:, 2, 1], M[:, 0, 2], M[:, 1, 0]], -1)
    w = -k_att[:, None] * e_R
    if yaw_rate is not None:
        w = w + yaw_rate[:, None] * R[:, 2, :]              # R^T e3 * yaw rate
    _, nw = _unit(w)
    return w * np.minimum(1.0, max_rate / np.where(nw > 0, nw, 1.0))[:, None]


def _hold_vz_thrust(state: QuadState, vz: np.ndarray, params: QuadParams, ctrl: CtrlGains) -> np.ndarray:
    az = ctrl.k_vz * (vz - state.v[:, 2])
    return np.clip((G + az + params.k_d * state.v[:, 2]) / np.maximum(state.R[:, 2, 2], 0.2), C_MIN, C_MAX)


def angle_mode(state: QuadState, sticks: dict, params: QuadParams, ctrl: CtrlGains) -> tuple[np.ndarray, np.ndarray]:
    """Sticks: tilt_xy (B,2) target tilt in the heading frame (rad), yaw_rate (B,), vz (B,) target climb rate."""
    psi = heading_of(state.R)
    h, l = _heading_axes(psi)
    t = sticks["tilt_xy"][:, :1] * h + sticks["tilt_xy"][:, 1:] * l
    t_dir, t_mag = _unit(t)
    t_mag = np.minimum(t_mag, ctrl.max_tilt)
    b3_des = np.sin(t_mag)[:, None] * t_dir + np.cos(t_mag)[:, None] * E3
    w = attitude_rate_cmd(state.R, b3_des, psi, ctrl.k_att, ctrl.max_rate, sticks["yaw_rate"])
    return w, _hold_vz_thrust(state, sticks["vz"], params, ctrl)


def p_mode(state: QuadState, sticks: dict, params: QuadParams, ctrl: CtrlGains) -> tuple[np.ndarray, np.ndarray]:
    """Sticks: v_xy (B,2) target horizontal velocity in the heading frame, yaw_rate (B,), vz (B,)."""
    psi = heading_of(state.R)
    h, l = _heading_axes(psi)
    v_t = sticks["v_xy"][:, :1] * h + sticks["v_xy"][:, 1:] * l
    a = ctrl.k_v[:, None] * (v_t - state.v) + sticks["yaw_rate"][:, None] * cross3(E3, v_t)   # target turns with the nose
    a[:, 2] = ctrl.k_vz * (sticks["vz"] - state.v[:, 2])
    f = a + G * E3 + params.k_d[:, None] * state.v
    fz = np.maximum(f[:, 2], C_MIN)             # a commanded fall: thrust floor, axis kept up (never pushes down)
    fh_dir, fh = _unit(f[:, :2])
    f[:, :2] = fh_dir * np.minimum(fh, fz * np.tan(ctrl.max_tilt))[:, None]
    f[:, 2] = fz
    c, b3_des = thrust_from_accel(f - G * E3 - params.k_d[:, None] * state.v, state.v, params.k_d, state.R[:, :, 2])
    return attitude_rate_cmd(state.R, b3_des, psi, ctrl.k_att, ctrl.max_rate, sticks["yaw_rate"]), c
