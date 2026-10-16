"""Minimal orientation baselines: hold the last attitude, or extrapolate a constant body angular velocity."""
import numpy as np
from scipy.spatial.transform import Rotation


def hold_orientation(obs_R: np.ndarray, H: int) -> np.ndarray:
    return np.repeat(np.asarray(obs_R, float)[:, -1:], H, axis=1)


def constant_angular_velocity(obs_t, obs_R, fut_t) -> np.ndarray:
    """omega = log(R_{N-2}^T R_{N-1}) / dt in the body frame; R(tau) = R_{N-1} exp(omega tau)."""
    obs_R = np.asarray(obs_R, float)
    W = obs_R.shape[0]
    dt = obs_t[-1] - obs_t[-2]
    last = Rotation.from_matrix(obs_R[:, -1])
    omega = (Rotation.from_matrix(obs_R[:, -2]).inv() * last).as_rotvec() / dt           # (W,3)
    tau = np.asarray(fut_t, float) - obs_t[-1]                                             # (H,)
    steps = Rotation.from_rotvec((omega[:, None] * tau[None, :, None]).reshape(-1, 3))
    lasts = Rotation.from_matrix(np.repeat(obs_R[:, -1], len(tau), axis=0))
    return (lasts * steps).as_matrix().reshape(W, len(tau), 3, 3)
