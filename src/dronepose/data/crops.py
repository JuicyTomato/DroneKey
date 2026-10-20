"""One crop rule for every model and domain: square around the propeller extent, resized to CROP_SIZE."""
from dataclasses import dataclass

import cv2
import numpy as np

CROP_SIZE = 256
CROP_FACTOR = 1.6
BOX_PAD = 0.2   # YOLO label box = keypoint extent + BOX_PAD * larger extent on every side


@dataclass(frozen=True)
class CropBox:
    x0: float
    y0: float
    side: float


def crop_box_from_kpts(kpts2d: np.ndarray, factor: float = CROP_FACTOR, min_side: float = 64.0) -> CropBox:
    lo, hi = kpts2d.min(0), kpts2d.max(0)
    cx, cy = (lo + hi) / 2
    side = max(factor * float(max(hi - lo)), min_side)
    return CropBox(cx - side / 2, cy - side / 2, side)


def crop_box_from_det_box(xyxy: np.ndarray, factor: float = CROP_FACTOR, pad: float = BOX_PAD,
                          min_side: float = 64.0) -> CropBox:
    """Setting B: crop from a detected box (YOLO label convention), same crop as setting A when the box is exact."""
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    extent = max(x1 - x0, y1 - y0) / (1 + 2 * pad)
    side = max(factor * extent, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return CropBox(cx - side / 2, cy - side / 2, side)


def frame_to_crop(pts: np.ndarray, box: CropBox, size: int = CROP_SIZE) -> np.ndarray:
    return (np.asarray(pts, float) - [box.x0, box.y0]) * (size / box.side)


def crop_to_frame(pts: np.ndarray, box: CropBox, size: int = CROP_SIZE) -> np.ndarray:
    return np.asarray(pts, float) * (box.side / size) + [box.x0, box.y0]


def crop_image(img: np.ndarray, box: CropBox, size: int = CROP_SIZE) -> np.ndarray:
    s = size / box.side
    m = np.array([[s, 0.0, -box.x0 * s], [0.0, s, -box.y0 * s]])
    return cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def label_box_xyxy(kpts2d: np.ndarray, wh=(1920, 1080), pad: float = BOX_PAD) -> np.ndarray:
    """YOLO label box in pixels: keypoint extent + pad * larger extent on every side, clipped to the frame."""
    k = np.asarray(kpts2d, float)
    lo, hi = k.min(0), k.max(0)
    p = pad * float(max(hi - lo))
    return np.r_[np.clip(lo - p, 0, wh), np.clip(hi + p, 0, wh)]
