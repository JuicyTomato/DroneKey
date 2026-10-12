"""Geometry shared by all phases: rotations, rigid alignment, projection."""
import numpy as np
from scipy.spatial.transform import Rotation

# Blender body axes (x = left, y = back, z = up) -> FLU (x = front, y = left, z = up). Pure rotation.
BLENDER_BODY_TO_FLU = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def euler_xyz(e) -> np.ndarray:
    """Blender 'XYZ' Euler angles (radians) -> rotation matrix R = Rz @ Ry @ Rx."""
    x, y, z = e
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Proper rotation R and translation t minimising ||dst - (src @ R.T + t)||."""
    sc, dc = src.mean(0), dst.mean(0)
    h = (src - sc).T @ (dst - dc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, dc - r @ sc


def project(K: np.ndarray, X: np.ndarray) -> np.ndarray:
    p = (K @ np.asarray(X, dtype=float).T).T
    return p[:, :2] / p[:, 2:3]


def geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def quat_to_rotmat(q) -> np.ndarray:
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, 0], R[:, 1]])


def sixd_to_rotmat(d6) -> np.ndarray:
    a, b = np.asarray(d6[:3], float), np.asarray(d6[3:], float)
    c1 = a / np.linalg.norm(a)
    b = b - (c1 @ b) * c1
    c2 = b / np.linalg.norm(b)
    return np.stack([c1, c2, np.cross(c1, c2)], axis=1)


def rotmat_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """(yaw, pitch, roll) in degrees, intrinsic Z-Y-X (aerospace / ROS REP-103)."""
    return Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
