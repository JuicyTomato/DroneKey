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


# Drones whose four propellers form a (nearly) square: the 3D model maps onto itself under the symmetries below, so
# the yaw (and the labels 1-4) cannot be recovered from the keypoints alone. Only the Tello (08) qualifies (0.047 x 0.05).
SYMMETRIC_DRONES = frozenset({"08"})


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


# body-frame symmetries of a square plate: yaw by k * 90 deg, alone or composed with a 180 deg flip about the x axis
BODY_SYMMETRIES = tuple(_rot("z", 90.0 * k) @ f for f in (np.eye(3), _rot("x", 180.0)) for k in range(4))


def symmetric_rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Rotation error that ignores the body symmetries of a square drone (smallest over BODY_SYMMETRIES)."""
    return min(rotation_error_deg(R_est @ S, R_gt) for S in BODY_SYMMETRIES)
