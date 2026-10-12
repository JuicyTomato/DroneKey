"""Keypoint predictors share one contract: keypoints in ORIGINAL frame coordinates, shape (4,3) = x, y, conf."""
from typing import Protocol

import numpy as np

from dronepose.data.crops import CROP_SIZE, crop_box_from_kpts, crop_image, crop_to_frame


class KeypointPredictor(Protocol):
    def predict(self, image: np.ndarray, record: dict) -> np.ndarray: ...


def _no_detection() -> np.ndarray:
    out = np.full((4, 3), np.nan)
    out[:, 2] = 0.0
    return out


class GTPredictor:
    """Ground-truth keypoints, optionally with Gaussian pixel noise (tests and the PnP sensitivity study)."""

    def __init__(self, sigma_px: float = 0.0, seed: int = 0):
        self.sigma = sigma_px
        self.rng = np.random.default_rng(seed)

    def predict(self, image: np.ndarray, record: dict) -> np.ndarray:
        k = np.array(record["kpts2d"], float)
        if self.sigma > 0:
            k = k + self.rng.normal(0.0, self.sigma, k.shape)
        return np.hstack([k, np.ones((4, 1))])


class YoloPosePredictor:
    """Ultralytics YOLO-Pose on the crop around the GT keypoints (setting A)."""

    def __init__(self, weights: str, crop_size: int = CROP_SIZE):
        from ultralytics import YOLO  # imported lazily: not needed for tests or data preparation
        self.model = YOLO(weights)
        self.size = crop_size

    def predict(self, image: np.ndarray, record: dict) -> np.ndarray:
        box = crop_box_from_kpts(np.array(record["kpts2d"], float))
        res = self.model(crop_image(image, box, self.size), imgsz=self.size, verbose=False)[0]
        if res.keypoints is None or len(res.boxes) == 0:
            return _no_detection()
        i = int(res.boxes.conf.argmax())
        xy = res.keypoints.xy[i].cpu().numpy()
        conf = res.keypoints.conf[i].cpu().numpy() if res.keypoints.conf is not None else np.ones(4)
        return np.hstack([crop_to_frame(xy, box, self.size), conf[:, None]])
