"""Gravity-aligned frame G: x right, y forward, z up, origin in the camera.

The 6DroneSyn camera is exactly level (verified on every frame: y_cam = -0.582 * z_world), so G is a fixed
rotation of the OpenCV camera frame (x right, y down, z forward).
"""
import numpy as np

GRAVITY_FROM_CAMERA = np.array([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, -1.0, 0.0]])


def to_gravity_positions(t_cam: np.ndarray) -> np.ndarray:
    return np.asarray(t_cam, float) @ GRAVITY_FROM_CAMERA.T


def to_gravity_rotations(R_cam: np.ndarray) -> np.ndarray:
    """Body (FLU) -> camera rotations to body -> G."""
    return GRAVITY_FROM_CAMERA @ np.asarray(R_cam, float)


def up_axis(R_G: np.ndarray) -> np.ndarray:
    """Body up axis (thrust direction) in G. Invariant to body yaw by multiples of 90 deg (Tello symmetry)."""
    return np.asarray(R_G, float)[..., :, 2]
