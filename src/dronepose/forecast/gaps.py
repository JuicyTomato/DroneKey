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

from dronepose.forecast.acro import acro_predict
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


def predict_gapped(windows: Windows, predict: Callable[[Windows], np.ndarray], min_valid: int = 2) -> np.ndarray:
    """predict(windows) on the valid frames of every window, reassembled in window order."""
    v = windows.obs_valid
    if v is None or np.asarray(v, bool).all():
        return predict(windows)
    patterns, inv = np.unique(np.asarray(v, bool), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    out = None
    for j, pat in enumerate(patterns):
        if pat.sum() < min_valid:
            raise ValueError(f"{pat.sum()} valid observed frames, {min_valid} needed: drop_sparse the windows first")
        sel = np.flatnonzero(inv == j)
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
    return predict_gapped(w, lambda s: cv(s.obs_t, s.obs_p, s.fut_t, min(k, _n(s))))


def ca_g(w: Windows, k: int) -> np.ndarray:
    return predict_gapped(w, lambda s: _or_cv(s, 3, lambda s: ca(s.obs_t, s.obs_p, s.fut_t, min(k, _n(s)))))


def kalman_g(w: Windows, model: str, q: float, r) -> np.ndarray:
    return predict_gapped(w, lambda s: kalman_predict(s.obs_t, s.obs_p, s.fut_t, model, q, r))


def kalman_los_g(w: Windows, q: float, r_los: float, r_lat: float) -> np.ndarray:
    return predict_gapped(w, lambda s: kalman_los_predict(s.obs_t, s.obs_p, s.last_p, s.fut_t, q, r_los, r_lat))


def physics_g(w: Windows, k: int, g: float, max_tilt_deg: float) -> np.ndarray:
    return predict_gapped(w, lambda s: physics_predict(s.obs_t, s.obs_p, s.obs_u, s.fut_t, min(k, _n(s)), g,
                                                       max_tilt_deg))


def ctr_g(w: Windows, method: str, k: int) -> np.ndarray:
    fn = {"ctrv": ctrv, "ctra": ctra}[method]
    return predict_gapped(w, lambda s: fn(s.obs_t, s.obs_p, s.fut_t, min(k, _n(s))))


def acro_g(w: Windows, k: int, lam, thrust, kd_mode) -> np.ndarray:
    return predict_gapped(w, lambda s: _or_cv(s, 3, lambda s: acro_predict(
        s.obs_t, s.obs_p, np.asarray(s.obs_R, float), s.fut_t, min(k, _n(s)), lam, thrust, kd_mode)))


def rot_hold_g(w: Windows) -> np.ndarray:
    return predict_gapped(w, lambda s: hold_orientation(s.obs_R, len(s.fut_t)), min_valid=1)


def rot_omega_g(w: Windows) -> np.ndarray:
    return predict_gapped(w, lambda s: constant_angular_velocity(s.obs_t, s.obs_R, s.fut_t))
