"""PnP solvers behind one interface: IPPE (2 candidates) and EPnP / SQPnP / iterative (1 candidate)."""
import cv2
import numpy as np

from dronepose.common.geometry import project
from dronepose.pose.pnp import PnPSolution, solve_ippe

SOLVERS = ("ippe", "epnp", "sqpnp", "iterative")

_FLAGS = {"epnp": cv2.SOLVEPNP_EPNP, "sqpnp": cv2.SOLVEPNP_SQPNP, "iterative": cv2.SOLVEPNP_ITERATIVE}


def _reproj(obj: np.ndarray, img: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    return float(np.linalg.norm(project(K, (R @ obj.T).T + t) - img, axis=1).mean())


def solve(name: str, model_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray,
          refine: bool = False) -> list[PnPSolution]:
    """Candidate poses sorted by increasing reprojection error (ippe: 2; the others: the best one only,
    since SQPnP can occasionally return a second, worse solution).

    Non-finite keypoints raise ValueError, as does an unknown solver name. An OpenCV error, zero solutions
    or a non-finite R/t yields an empty list (never an exception).
    """
    if name not in SOLVERS:
        raise ValueError(f"unknown PnP solver {name!r}; expected one of {SOLVERS}")
    obj = np.ascontiguousarray(model_pts, dtype=np.float64)
    img = np.ascontiguousarray(img_pts, dtype=np.float64)
    Kf = np.asarray(K, np.float64)
    if not (np.isfinite(obj).all() and np.isfinite(img).all()):
        raise ValueError("non-finite keypoints")
    try:
        if name == "ippe" and not refine:
            sols = solve_ippe(obj, img, Kf)
            return [s for s in sols if np.isfinite(s.R).all() and np.isfinite(s.t).all()]
        flags = cv2.SOLVEPNP_IPPE if name == "ippe" else _FLAGS[name]
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, Kf, None, flags=flags)
        sols = []
        for rv, tv in zip(rvecs[:n], tvecs[:n]):
            rv = np.asarray(rv, np.float64).reshape(3, 1)
            tv = np.asarray(tv, np.float64).reshape(3, 1)
            if refine:
                rv, tv = cv2.solvePnPRefineLM(obj, img, Kf, None, rv.copy(), tv.copy())
            R = cv2.Rodrigues(np.asarray(rv, np.float64).reshape(3, 1))[0]
            t = np.asarray(tv, np.float64).ravel()
            if not (np.isfinite(R).all() and np.isfinite(t).all()):
                continue
            err = _reproj(obj, img, Kf, R, t)
            if not np.isfinite(err):
                continue
            sols.append(PnPSolution(R, t, err))
    except cv2.error:
        return []
    sols = sorted(sols, key=lambda s: s.reproj_px)
    return sols if name == "ippe" else sols[:1]
