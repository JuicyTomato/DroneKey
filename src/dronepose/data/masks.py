"""Drone masks of the 6DroneSyn train split: one per frame geometry, shared by the renders on different backgrounds."""
from itertools import combinations

import cv2
import numpy as np

from dronepose.data.crops import CROP_SIZE, CropBox


def geom_key(frame_id: str) -> str:
    """'01_05_13_000_0100' -> '01_05_000_0100': the background field is dropped (the renders share keypoints and box)."""
    scene, drone, _background, degree, idx = frame_id.split("_")
    return f"{scene}_{drone}_{degree}_{idx}"


def keypoints_in_mask(mask: np.ndarray, kpts: np.ndarray, radius: int = 3) -> bool:
    """Every keypoint has a mask pixel within `radius`: a hub can sit just off the segmented motor."""
    h, w = mask.shape[:2]
    for x, y in np.asarray(kpts, float).reshape(-1, 2).round().astype(int):
        if not (0 <= x < w and 0 <= y < h):
            return False
        if not mask[max(y - radius, 0):y + radius + 1, max(x - radius, 0):x + radius + 1].any():
            return False
    return True


def center_in_mask(mask: np.ndarray, kpts: np.ndarray, frac: float = 0.5, radius: float = 0.04) -> bool:
    """At least `frac` of a disk on the hub mean (radius * hub extent) is mask: hubs alone pass a mask without the body."""
    k = np.asarray(kpts, float).reshape(-1, 2)
    disk = np.zeros(mask.shape[:2], np.uint8)
    r = max(2, round(radius * float(np.ptp(k, axis=0).max())))
    cv2.circle(disk, tuple(int(v) for v in k.mean(0).round()), r, 1, -1)
    return bool(disk.any() and (mask[disk > 0] > 0).mean() >= frac)


def diff_mask(renders: list[np.ndarray], kpts: np.ndarray, thr: int = 40) -> np.ndarray:
    """(H, W) uint8 0/1 mask of what stays the same across renders on different backgrounds, kept near the keypoints.
    Quality check only: when the backgrounds look alike it leaks over the whole crop."""
    imgs = [r.astype(np.int16) for r in renders]
    diff = np.zeros(imgs[0].shape[:2], np.int16)
    for a, b in combinations(imgs, 2):
        diff = np.maximum(diff, np.abs(a - b).max(2))
    m = (diff < thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    _, labels = cv2.connectedComponents(m)
    h, w = m.shape
    keep: set[int] = set()
    for x, y in np.asarray(kpts, float).reshape(-1, 2).round().astype(int):
        x, y = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        window = labels[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4]
        keep |= set(window[window > 0].tolist())
    out = np.isin(labels, sorted(keep)).astype(np.uint8)
    flood, seeds = out.copy(), np.zeros((h + 2, w + 2), np.uint8)   # fill the holes: what the border cannot reach
    if not flood[0, 0]:
        cv2.floodFill(flood, seeds, (0, 0), 1)
        out = out | (1 - flood)
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def area_limits(areas: dict[str, list[float]], factor: float = 2.5) -> dict[str, float]:
    """Largest accepted mask area per drone: median area of that drone times `factor`."""
    return {d: float(np.median(v)) * factor for d, v in areas.items()}


def soften(mask: np.ndarray) -> np.ndarray:
    """Soft edge of about 1 px, so that the pasted drone has no halo."""
    return cv2.GaussianBlur(mask, (3, 3), 0)


def native_to_crop(mask: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Mask at the native crop resolution -> the 256 px crop of the dataset."""
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA if mask.shape[0] > size else cv2.INTER_LINEAR)


def native_to_frame(mask: np.ndarray, box: CropBox, wh=(1920, 1080), out=(640, 360)) -> np.ndarray:
    """Mask at the native crop resolution -> full frame `out`, zero outside the crop."""
    s = box.side / mask.shape[0] * out[0] / wh[0]
    m = np.array([[s, 0.0, box.x0 * out[0] / wh[0]], [0.0, s, box.y0 * out[1] / wh[1]]])
    return cv2.warpAffine(mask, m, out, flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
