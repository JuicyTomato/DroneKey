"""Error metrics shared by the phases."""
import numpy as np

from dronepose.common.geometry import geodesic_deg


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    return geodesic_deg(R_est, R_gt)


def translation_error(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(t_est) - np.asarray(t_gt)))


def ade(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=1).mean())


def fde(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pred)[-1] - np.asarray(gt)[-1]))
