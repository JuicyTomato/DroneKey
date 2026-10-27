"""Kalman CA with the measurement variance split along the line of sight (spec 2026-10-27 §5.5, E4b only).

The PnP error is ~22x larger along the camera -> drone direction than across it (archive 2.6). Per window the line
of sight is the direction of the last observed absolute position (the gravity frame has its origin in the camera):
the observations are expressed in the basis (line of sight, two normals), filtered per axis with r = (r_los, r_lat,
r_lat) and rotated back. With r_los = r_lat it is the Kalman CA (the per-axis filter commutes with rotations).
"""
import numpy as np

from dronepose.forecast.kalman import kalman_predict


def los_basis(absolute: np.ndarray) -> np.ndarray:
    """(W,3,3) rows: unit line of sight, then two unit normals (right-handed)."""
    los = absolute / np.linalg.norm(absolute, axis=-1, keepdims=True)
    ref = np.where(np.abs(los[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    e1 = np.cross(ref, los)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    return np.stack([los, e1, np.cross(los, e1)], axis=1)


def kalman_los_predict(obs_t, obs_p, last_p, fut_t, q: float, r_los: float, r_lat: float) -> np.ndarray:
    obs_p = np.asarray(obs_p, float)
    B = los_basis(np.asarray(last_p, float) + obs_p[:, -1])
    z = np.einsum("wij,wnj->wni", B, obs_p)
    pred = kalman_predict(obs_t, z, fut_t, "ca", q, np.array([r_los, r_lat, r_lat], float))
    return np.einsum("wji,whj->whi", B, pred)
