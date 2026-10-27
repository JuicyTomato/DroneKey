"""Lost observed frames (spec 2026-10-27 §5.2): every predictor fits on the valid frames only.

Windows are grouped by their pattern of valid frames; each group is predicted with the observed times and poses
restricted to its valid frames (all predictors work in time relative to their last observation, so a lost last frame
needs nothing else). Without `obs_valid`, or with every frame valid, the predictor is called once on the windows as
they are: results are bit-identical to the code before the mask. Fits that need more frames than are valid shrink k
to the valid count and fall back to CV through the valid frames when even that is too few.
"""
import dataclasses
from typing import Callable

import numpy as np

from dronepose.forecast.acro import acro_predict, acro_rollout, estimate_state
from dronepose.forecast.baselines import ca, constant_position, cv
from dronepose.forecast.ctr import ctra, ctrv
from dronepose.forecast.kalman import kalman_predict
from dronepose.forecast.kalman_los import kalman_los_predict
from dronepose.forecast.orientation import constant_angular_velocity, hold_orientation
from dronepose.forecast.physics import physics_predict
from dronepose.forecast.windows import Windows


def last_valid_index(obs_valid: np.ndarray) -> np.ndarray:
    """(W,) index of the last valid observed frame (0 when none is valid)."""
    v = np.asarray(obs_valid, bool)
    return v.shape[1] - 1 - np.argmax(v[:, ::-1], axis=1)


def restrict_frames(windows: Windows, frames: np.ndarray) -> Windows:
    """The windows with only the observed frames `frames` (a boolean mask over N); no mask left."""
    f = np.asarray(frames, bool)
    w = windows
    return dataclasses.replace(w, obs_t=w.obs_t[f], obs_p=w.obs_p[:, f], obs_R=w.obs_R[:, f], obs_u=w.obs_u[:, f],
                               obs_valid=None)


def last_k_valid(obs_valid: np.ndarray, k: int | None) -> np.ndarray:
    """(W,N) the last k valid frames of every window (all valid frames when k is None)."""
    v = np.asarray(obs_valid, bool)
    if k is None:
        return v
    rank = np.cumsum(v[:, ::-1], axis=1)[:, ::-1]          # 1 = last valid frame, 2 = the one before, ...
    return v & (rank <= k)


def _groups(windows: Windows, min_valid: int, last_k: int | None):
    v = np.asarray(windows.obs_valid, bool)
    n_valid = v.sum(1)
    if (n_valid < min_valid).any():
        raise ValueError(f"{n_valid.min()} valid observed frames, {min_valid} needed: drop_sparse the windows first")
    patterns, inv = np.unique(last_k_valid(v, last_k), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    return [(pat, np.flatnonzero(inv == j)) for j, pat in enumerate(patterns)]


def predict_gapped(windows: Windows, predict: Callable[[Windows], np.ndarray], min_valid: int = 2,
                   last_k: int | None = None) -> np.ndarray:
    """predict(windows) on the valid frames of every window, reassembled in window order. last_k: the predictor only
    uses the last k observations, so windows are grouped (and restricted) by their last k valid frames only."""
    v = windows.obs_valid
    if v is None or np.asarray(v, bool).all():
        return predict(windows)
    out = None
    for pat, sel in _groups(windows, min_valid, last_k):
        pred = predict(restrict_frames(windows.subset(sel), pat))
        if out is None:
            out = np.empty((len(windows),) + pred.shape[1:], pred.dtype)
        out[sel] = pred
    return out


def drop_sparse(windows: Windows, min_valid: int = 2) -> tuple[Windows, np.ndarray]:
    """Windows with at least `min_valid` valid observed frames, and the (W,) mask of the kept ones."""
    if windows.obs_valid is None:
        return windows, np.ones(len(windows), bool)
    keep = np.asarray(windows.obs_valid, bool).sum(1) >= min_valid
    return windows.subset(keep), keep


def _n(s: Windows) -> int:
    return len(s.obs_t)


def _or_cv(s: Windows, min_n: int, fn: Callable[[Windows], np.ndarray]) -> np.ndarray:
    return fn(s) if _n(s) >= min_n else cv(s.obs_t, s.obs_p, s.fut_t, _n(s))


# ----- the predictors of tuning.predict_all, gap-aware -------------------------------------------------------------

def const_g(w: Windows) -> np.ndarray:
    return predict_gapped(w, lambda s: constant_position(s.obs_t, s.obs_p, s.fut_t), min_valid=1)


def cv_g(w: Windows, k: int) -> np.ndarray:
    return predict_gapped(w, lambda s: cv(s.obs_t, s.obs_p, s.fut_t, min(k, _n(s))), last_k=k)


def ca_g(w: Windows, k: int) -> np.ndarray:
    return predict_gapped(w, lambda s: _or_cv(s, 3, lambda s: ca(s.obs_t, s.obs_p, s.fut_t, min(k, _n(s)))),
                          last_k=k)


def kalman_g(w: Windows, model: str, q: float, r) -> np.ndarray:
    return predict_gapped(w, lambda s: kalman_predict(s.obs_t, s.obs_p, s.fut_t, model, q, r))


def kalman_los_g(w: Windows, q: float, r_los: float, r_lat: float) -> np.ndarray:
    return predict_gapped(w, lambda s: kalman_los_predict(s.obs_t, s.obs_p, s.last_p, s.fut_t, q, r_los, r_lat))


def physics_g(w: Windows, k: int, g: float, max_tilt_deg: float) -> np.ndarray:
    return predict_gapped(w, lambda s: physics_predict(s.obs_t, s.obs_p, s.obs_u, s.fut_t, min(k, _n(s)), g,
                                                       max_tilt_deg), last_k=k)


def ctr_g(w: Windows, method: str, k: int) -> np.ndarray:
    fn = {"ctrv": ctrv, "ctra": ctra}[method]
    return predict_gapped(w, lambda s: fn(s.obs_t, s.obs_p, s.fut_t, min(k, _n(s))), last_k=k)


def acro_g(w: Windows, k: int, lam, thrust, kd_mode) -> np.ndarray:
    """The state is estimated per group of last-k-valid frames, the costly integration runs once per time of the
    last valid frame (the same values as acro_predict group by group)."""
    v = w.obs_valid
    if v is None or np.asarray(v, bool).all():
        return predict_gapped(w, lambda s: _or_cv(s, 3, lambda s: acro_predict(
            s.obs_t, s.obs_p, np.asarray(s.obs_R, float), s.fut_t, min(k, _n(s)), lam, thrust, kd_mode)))
    lam_w = np.broadcast_to(np.asarray(lam, float), (len(w),))
    out = np.empty((len(w), len(w.fut_t), 3))
    by_t_last: dict[float, list] = {}
    for pat, sel in _groups(w, 2, k):
        s = restrict_frames(w.subset(sel), pat)
        if _n(s) < 3:
            out[sel] = cv(s.obs_t, s.obs_p, s.fut_t, _n(s))
            continue
        est = estimate_state(s.obs_t, s.obs_p, np.asarray(s.obs_R, float), min(k, _n(s)), kd_mode, lam_w[sel])
        by_t_last.setdefault(float(s.obs_t[-1]), []).append((sel, est))
    for t_last, parts in by_t_last.items():
        sel = np.concatenate([p[0] for p in parts])
        est = {key: np.concatenate([p[1][key] for p in parts]) for key in parts[0][1]}
        out[sel] = acro_rollout(est, w.fut_t, lam_w[sel], thrust, t_last)
    return out


def rot_hold_g(w: Windows) -> np.ndarray:
    return predict_gapped(w, lambda s: hold_orientation(s.obs_R, len(s.fut_t)), min_valid=1)


def rot_omega_g(w: Windows) -> np.ndarray:
    return predict_gapped(w, lambda s: constant_angular_velocity(s.obs_t, s.obs_R, s.fut_t))
