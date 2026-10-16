"""Polynomial baselines, vectorised over windows: constant position, CV-k and CA-k (least squares on the last k)."""
import numpy as np


def _design(times: np.ndarray, degree: int) -> np.ndarray:
    return np.stack([times ** d for d in range(degree + 1)], axis=1)


def _fit(obs_t: np.ndarray, obs_p: np.ndarray, k: int, degree: int) -> np.ndarray:
    """Polynomial coefficients (W, degree+1, 3) in time relative to the last observation."""
    N = len(obs_t)
    if k > N:
        raise ValueError(f"k = {k} larger than the observed window ({N})")
    if k <= degree:
        raise ValueError(f"k = {k} too small for degree {degree}")
    tau = np.asarray(obs_t[-k:], float) - obs_t[-1]
    pinv = np.linalg.pinv(_design(tau, degree))                 # (degree+1, k), shared by all windows
    return np.einsum("ck,wkd->wcd", pinv, np.asarray(obs_p, float)[:, -k:])


def poly_fit_predict(obs_t, obs_p, fut_t, k: int, degree: int) -> np.ndarray:
    coef = _fit(obs_t, obs_p, k, degree)
    A = _design(np.asarray(fut_t, float) - obs_t[-1], degree)  # (H, degree+1)
    return np.einsum("hc,wcd->whd", A, coef)


def constant_position(obs_t, obs_p, fut_t) -> np.ndarray:
    return np.repeat(np.asarray(obs_p, float)[:, -1:], len(fut_t), axis=1)


def cv(obs_t, obs_p, fut_t, k: int) -> np.ndarray:
    return poly_fit_predict(obs_t, obs_p, fut_t, k, 1)


def ca(obs_t, obs_p, fut_t, k: int) -> np.ndarray:
    return poly_fit_predict(obs_t, obs_p, fut_t, k, 2)


def fit_state(obs_t, obs_p, k: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity (W,3) of the fitted polynomial at the last observed time."""
    coef = _fit(obs_t, obs_p, k, degree)
    return coef[:, 0], coef[:, 1]
