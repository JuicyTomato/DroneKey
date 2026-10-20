"""Translation units and drone attitude helpers."""
import numpy as np

WORLD_PER_CAMERA = 1.718   # world units per camera unit
METERS_PER_UNIT = {"01": 0.7463, "02": 0.8053, "03": 0.8525, "04": 0.9334, "05": 0.9253, "06": 0.8521}
UP_CAM = np.array([0.0, -1.0, 0.0])   # camera y points down


def tilt_deg(R: np.ndarray) -> float:
    """Angle between the drone's up axis (R[:, 2], FLU z in camera frame) and the camera-frame vertical."""
    z = np.asarray(R, float)[:, 2]
    return float(np.degrees(np.arccos(np.clip(z @ UP_CAM / np.linalg.norm(z), -1.0, 1.0))))
