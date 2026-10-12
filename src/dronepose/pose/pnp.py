"""Planar PnP (IPPE): 4 near-coplanar propellers give two candidate poses."""
from dataclasses import dataclass

import cv2
import numpy as np

from dronepose.common.geometry import project


@dataclass
class PnPSolution:
    R: np.ndarray
    t: np.ndarray
    reproj_px: float


def solve_ippe(model_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray) -> list[PnPSolution]:
    obj = np.ascontiguousarray(model_pts, dtype=np.float64)
    img = np.ascontiguousarray(img_pts, dtype=np.float64)
    if not (np.isfinite(obj).all() and np.isfinite(img).all()):
        raise ValueError("non-finite keypoints")
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, np.asarray(K, np.float64), None, flags=cv2.SOLVEPNP_IPPE)
    sols = []
    for rv, tv in zip(rvecs[:n], tvecs[:n]):
        R = cv2.Rodrigues(rv)[0]
        t = tv.ravel()
        err = float(np.linalg.norm(project(K, (R @ obj.T).T + t) - img, axis=1).mean())
        sols.append(PnPSolution(R, t, err))
    return sorted(sols, key=lambda s: s.reproj_px)


def estimate_pose(model_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray) -> PnPSolution:
    sols = solve_ippe(model_pts, img_pts, K)
    if not sols:
        raise ValueError("IPPE returned no solution")
    return sols[0]
