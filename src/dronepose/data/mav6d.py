"""MAV6D (Zheng et al., IEEE T-RO 2024): real indoor frames of a DJI Mavic 2 with VICON pose, fixed camera.

Calibration from the authors' util.py: intrinsics K, distortion D (OpenCV order k1 k2 p1 p2 k3) and the fixed
VICON -> camera transform. The skew 0.3984 of K is ignored, by util.py and by OpenCV alike.
Label line: ts_cam, t_cam(3), q_cam(4), ts_uav, t_uav(3), q_uav(4) in VICON (q = x y z w); util.py uses only the
UAV part. The UAV body frame is FLU like ours (x front = gimbal, z up, checked on frames);
its origin is the VICON marker body, not the propeller centroid.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

K = np.array([[1979.4, 0.3984, 976.8189], [0.0, 1979.1, 533.9717], [0.0, 0.0, 1.0]])
DIST = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487])
CAM_FROM_VICON = np.array([[0.6685859, -0.74342, 0.01787715, -0.3814142759180792],
                           [0.01558769, -0.01002444, -0.99982825, 1.604836888783723],
                           [0.74347153, 0.66874974, 0.004886, 3.035574156448842]])
IMAGE_SIZE = (1920, 1080)


def read_label(path: Path, cam_from_vicon: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(R, t): UAV body frame in the camera frame (metres). cam_from_vicon (3x4): default the authors' one."""
    v = [float(x) for x in Path(path).read_text().split()]
    t_vicon, q = np.array(v[9:12]), v[12:16]
    R_vicon = Rotation.from_quat(q).as_matrix()
    T = CAM_FROM_VICON if cam_from_vicon is None else np.asarray(cam_from_vicon, float)
    return T[:, :3] @ R_vicon, T[:, :3] @ t_vicon + T[:, 3]


def project(points_body: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Body points (N,3) -> distorted pixels (N,2) of the original image."""
    p, _ = cv2.projectPoints(np.asarray(points_body, float), cv2.Rodrigues(R)[0], t, K, DIST)
    return p.reshape(-1, 2)


def undistort(px: np.ndarray) -> np.ndarray:
    """Distorted pixels (N,2) -> normalised camera coordinates (N,2)."""
    return cv2.undistortPoints(np.asarray(px, float).reshape(-1, 1, 2), K, DIST).reshape(-1, 2)
