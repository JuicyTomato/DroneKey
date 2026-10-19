"""Turn-rate baselines CTRV / CTRA (spec §5.4): constant speed (or tangential acceleration) and constant turn rate
in the horizontal plane, CV on the vertical; vectorised over windows.

From the last k observations: chord velocities between consecutive frames give the course psi at the mid-times,
whose straight-line fit gives the turn rate; chord speeds are corrected for the curvature (a chord of an arc is
shorter than the arc by sinc(omega dt / 2)). The future starts at the last observed position.
"""
import numpy as np

from dronepose.forecast.baselines import fit_state

MIN_SPEED = 1e-6     # below this the course is undefined: no turn


def _line_fit(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares y = a + b t per window (W,k) on shared times t (k,); returns a, b."""
    A = np.stack([np.ones_like(t), t], 1)
    coef = np.linalg.pinv(A) @ y.T                                    # (2, W)
    return coef[0], coef[1]


def _turn_state(obs_t, obs_p, k: int):
    obs_t, obs_p = np.asarray(obs_t, float), np.asarray(obs_p, float)
    if k < 2 or k > len(obs_t):
        raise ValueError(f"k = {k} outside [2, {len(obs_t)}]")
    t, p = obs_t[-k:] - obs_t[-1], obs_p[:, -k:, :2]
    dt = np.diff(t)
    chord = np.diff(p, axis=1) / dt[None, :, None]                   # (W,k-1,2)
    tm = (t[1:] + t[:-1]) / 2
    speed_c = np.linalg.norm(chord, axis=-1)
    moving = speed_c.mean(1) > MIN_SPEED
    psi = np.unwrap(np.arctan2(chord[..., 1], chord[..., 0]), axis=1)
    if k >= 3:
        psi0, w = _line_fit(tm, psi)
    else:
        psi0, w = psi[:, -1], np.zeros(len(psi))
    w = np.where(moving, w, 0.0)
    corr = 1 / np.sinc(w[:, None] * dt[None] / (2 * np.pi))         # arc / chord
    speed = speed_c * corr
    return psi0, w, speed, tm


def _vertical(obs_t, obs_p, fut_t, k):
    z0, vz = fit_state(obs_t, np.asarray(obs_p, float)[..., 2:], k, 1)
    tau = np.asarray(fut_t, float) - obs_t[-1]
    return z0[:, None, 0] + vz[:, None, 0] * tau[None]


def ctrv(obs_t, obs_p, fut_t, k: int) -> np.ndarray:
    psi0, w, speed, _ = _turn_state(obs_t, obs_p, k)
    v = speed.mean(1)
    tau = (np.asarray(fut_t, float) - obs_t[-1])[None]               # (1,H)
    # exact arc: chord length v * tau * sinc(w tau / 2), direction psi0 + w tau / 2 (no division by w)
    L = v[:, None] * tau * np.sinc(w[:, None] * tau / (2 * np.pi))
    ang = psi0[:, None] + w[:, None] * tau / 2
    out = np.empty((len(v), tau.shape[1], 3))
    last = np.asarray(obs_p, float)[:, -1]
    out[..., 0] = last[:, None, 0] + L * np.cos(ang)
    out[..., 1] = last[:, None, 1] + L * np.sin(ang)
    out[..., 2] = _vertical(obs_t, obs_p, fut_t, k)
    return out


def ctra(obs_t, obs_p, fut_t, k: int, substeps: int = 16) -> np.ndarray:
    psi0, w, speed, tm = _turn_state(obs_t, obs_p, k)
    if speed.shape[1] >= 2:
        v0, acc = _line_fit(tm, speed)
    else:
        v0, acc = speed[:, -1], np.zeros(len(speed))
    tau = np.asarray(fut_t, float) - obs_t[-1]
    fine = np.linspace(0.0, tau[-1], substeps * len(tau) + 1)        # trapezoid on a fine grid
    sp = v0[:, None] + acc[:, None] * fine[None]
    ang = psi0[:, None] + w[:, None] * fine[None]
    vel = np.stack([sp * np.cos(ang), sp * np.sin(ang)], -1)
    dpos = np.concatenate([np.zeros((len(sp), 1, 2)),
                           np.cumsum((vel[:, 1:] + vel[:, :-1]) / 2 * np.diff(fine)[None, :, None], axis=1)], 1)
    idx = np.searchsorted(fine, tau - 1e-12)
    out = np.empty((len(sp), len(tau), 3))
    out[..., :2] = np.asarray(obs_p, float)[:, -1, None, :2] + dpos[:, idx]
    out[..., 2] = _vertical(obs_t, obs_p, fut_t, k)
    return out
