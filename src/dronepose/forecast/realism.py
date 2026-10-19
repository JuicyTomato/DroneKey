"""Descriptors of trajectories for the realism check of the generator (spec §5.5): generated vs 6DroneSyn.

Per frame (metric trajectories only, 3 frames trimmed at each end for the finite differences): speed, acceleration
and jerk norms, tilt, angular speed, angle between the nose and the horizontal motion (when moving); per run:
duration of hovering (speed < 0.2 m/s) and of turning (course rate > 10 deg/s while moving).
"""
import numpy as np
from scipy.spatial.transform import Rotation

TRIM = 3
HOVER_SPEED = 0.2      # m/s
MOVING_SPEED = 0.3     # m/s
TURN_RATE = 10.0       # deg/s


def _runs(mask: np.ndarray) -> list[int]:
    """Lengths of the runs of True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))


def descriptors(trajs) -> dict[str, np.ndarray]:
    out = {k: [] for k in ("speed", "accel", "jerk", "tilt_deg", "omega_deg_s", "heading_motion_deg", "hover_s",
                           "curve_s")}
    for tr in trajs:
        if not tr.metric:
            continue
        dt = float(tr.times[1] - tr.times[0])
        p, R = np.asarray(tr.positions, float), np.asarray(tr.R_G, float)
        v = np.gradient(p, dt, axis=0)
        a = np.gradient(v, dt, axis=0)
        j = np.gradient(a, dt, axis=0)
        sl = slice(TRIM, len(p) - TRIM)
        speed = np.linalg.norm(v, axis=1)
        out["speed"].append(speed[sl])
        out["accel"].append(np.linalg.norm(a, axis=1)[sl])
        out["jerk"].append(np.linalg.norm(j, axis=1)[sl])
        out["tilt_deg"].append(np.degrees(np.arccos(np.clip(R[sl, 2, 2], -1, 1))))
        rel = Rotation.from_matrix(np.einsum("tji,tjk->tik", R[:-1], R[1:]))
        out["omega_deg_s"].append(np.degrees(np.linalg.norm(rel.as_rotvec(), axis=1) / dt)[TRIM:len(p) - TRIM])
        vh = v[sl, :2]
        sh = np.linalg.norm(vh, axis=1)
        moving = sh > MOVING_SPEED
        nose = R[sl, :2, 0]
        cos = np.sum(nose * vh, 1) / np.maximum(np.linalg.norm(nose, axis=1) * sh, 1e-12)
        out["heading_motion_deg"].append(np.degrees(np.arccos(np.clip(cos[moving], -1, 1))))
        course = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        rate = np.degrees(np.abs(np.gradient(course, dt)))[sl]
        out["hover_s"] += [n * dt for n in _runs(speed[sl] < HOVER_SPEED)]
        out["curve_s"] += [n * dt for n in _runs(moving & (rate > TURN_RATE))]
    return {k: np.concatenate(v) if v and isinstance(v[0], np.ndarray) else np.asarray(v, float) for k, v in out.items()}


def coverage_2d(x, y, bins) -> float:
    """Fraction of the 2D histogram cells holding at least one point (bins as for np.histogram2d)."""
    h, _, _ = np.histogram2d(np.asarray(x, float), np.asarray(y, float), bins=bins)
    return float((h > 0).mean())
