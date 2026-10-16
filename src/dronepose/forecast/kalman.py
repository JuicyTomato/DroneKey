"""Kalman CV / CA baselines, vectorised over windows and axes (axes independent, position-only measurements).

Process noise: continuous white acceleration (CV) or white jerk (CA) with spectral density q.
Measurement variance r per window, frame and axis (hook for pose uncertainty, plan steps 2.6 / E4b).
"""
from typing import Literal

import numpy as np


def _transition(dt: float, d: int, q: float) -> tuple[np.ndarray, np.ndarray]:
    if d == 2:
        F = np.array([[1, dt], [0, 1]])
        Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    else:
        F = np.array([[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, 1]])
        Q = q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                          [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                          [dt ** 3 / 6, dt ** 2 / 2, dt]])
    return F, Q


def _predict(x, P, dt, d, q):
    F, Q = _transition(dt, d, q)
    return x @ F.T, F @ P @ F.T + Q


def kalman_predict(obs_t, obs_p, fut_t, model: Literal["cv", "ca"], q: float, r: float | np.ndarray) -> np.ndarray:
    obs_t, fut_t = np.asarray(obs_t, float), np.asarray(fut_t, float)
    z = np.asarray(obs_p, float)                                 # (W,N,3)
    W, N, _ = z.shape
    if N < 2:
        raise ValueError("Kalman needs at least two observations")
    d = {"cv": 2, "ca": 3}[model]
    r = np.broadcast_to(np.asarray(r, float), (W, N, 3))

    # initialise at the second observation: position z1, velocity by finite difference, acceleration 0
    dt0 = obs_t[1] - obs_t[0]
    x = np.zeros((W, 3, d))
    x[..., 0] = z[:, 1]
    x[..., 1] = (z[:, 1] - z[:, 0]) / dt0
    P = np.zeros((W, 3, d, d))
    r0, r1 = r[:, 0], r[:, 1]
    P[..., 0, 0] = r1
    P[..., 0, 1] = P[..., 1, 0] = r1 / dt0
    P[..., 1, 1] = (r0 + r1) / dt0 ** 2
    if d == 3:
        # unknown acceleration (prior var sa2, mean 0): the finite difference is v1 - a*dt0/2, so v and a are correlated
        sa2 = 1e2
        P[..., 2, 2] = sa2
        P[..., 1, 1] += sa2 * dt0 ** 2 / 4
        P[..., 1, 2] = P[..., 2, 1] = sa2 * dt0 / 2

    for i in range(2, N):
        x, P = _predict(x, P, obs_t[i] - obs_t[i - 1], d, q)
        S = P[..., 0, 0] + r[:, i]                               # (W,3) scalar innovation per axis
        K = P[..., :, 0] / S[..., None]                          # (W,3,d)
        x = x + K * (z[:, i] - x[..., 0])[..., None]
        P = P - K[..., :, None] * P[..., None, 0, :]

    out = np.empty((W, len(fut_t), 3))
    t_prev = obs_t[-1]
    for j, t in enumerate(fut_t):
        x, P = _predict(x, P, t - t_prev, d, q)
        out[:, j] = x[..., 0]
        t_prev = t
    return out
