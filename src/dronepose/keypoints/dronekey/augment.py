"""Ultralytics-style augmentation of a 256 px crop (BGR uint8) and its keypoints, without importing ultralytics.

Math follows Ultralytics 8.4.178, ultralytics/data/augment.py
(https://github.com/ultralytics/ultralytics/blob/v8.4.178/ultralytics/data/augment.py):
- RandomPerspective._compute_affine_matrix: M = T @ S @ R @ P @ C with P = S(hear) = I here; C centres the image,
  R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s), T = U(0.5 - t, 0.5 + t) * size;
  RandomPerspective.apply_image: cv2.warpAffine, borderValue 114; apply_keypoints: xy @ M.T (no clipping here).
- RandomHSV.apply_image (>= 8.3.79): r = U(-1, 1, 3) * (h, s, v); hue LUT (x + r0 * 180) % 180, saturation LUT
  clip(x * (1 + r1)), value LUT clip(x * (1 + r2)), lut_sat[0] = 0 (pure white keeps its colour).
- Albumentations: default list Blur, MedianBlur, ToGray, CLAHE at p = 0.01 (applied to pixels only).
- RandomFlip.apply_image/apply_instances: x' = w - x, keypoints reordered with flip_idx.
- v8_transforms order for pose (mosaic off): RandomPerspective -> Albumentations -> RandomHSV -> RandomFlip.
Keypoints that leave the crop are kept (Ultralytics only clears their visibility flag).
"""
import cv2
import numpy as np

from dronepose.keypoints.recipes import RECIPES, albumentations_for

BORDER = (114, 114, 114)


def affine_matrix(w: int, h: int, angle_deg: float, scale: float, tx_frac: float, ty_frac: float) -> np.ndarray:
    """3x3 affine: centre -> rotation + scale about the centre -> translation of tx_frac*w, ty_frac*h (0.5 = none)."""
    C = np.eye(3)
    C[0, 2], C[1, 2] = -w / 2, -h / 2
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=angle_deg, center=(0, 0), scale=scale)
    T = np.eye(3)
    T[0, 2], T[1, 2] = tx_frac * w, ty_frac * h
    return T @ R @ C


def random_affine(img, kpts_px, degrees: float, translate: float, scale: float, rng):
    """Random rotation U(-degrees, degrees), scale U(1-scale, 1+scale), translation U(0.5-t, 0.5+t); border 114."""
    h, w = img.shape[:2]
    a = rng.uniform(-degrees, degrees)
    s = rng.uniform(1 - scale, 1 + scale)
    tx, ty = rng.uniform(0.5 - translate, 0.5 + translate, 2)
    M = affine_matrix(w, h, a, s, tx, ty)
    out = cv2.warpAffine(img, M[:2], dsize=(w, h), borderValue=BORDER)
    k = np.asarray(kpts_px, np.float64).reshape(-1, 2)
    return out, k @ M[:2, :2].T + M[:2, 2]


def augment_hsv(img, hgain: float, sgain: float, vgain: float, rng):
    """In-place-style HSV jitter with Ultralytics' LUT formula; identity when all gains are 0."""
    if not (hgain or sgain or vgain):
        return img
    r = rng.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain])
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x + r[0] * 180) % 180).astype(np.uint8)
    lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(np.uint8)
    lut_sat[0] = 0  # pure white keeps its colour
    lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def fliplr(img, kpts_px, flip_idx):
    """Horizontal flip: x' = w - x, then keypoints reordered with flip_idx."""
    w = img.shape[1]
    k = np.asarray(kpts_px, np.float64).reshape(-1, 2).copy()
    k[:, 0] = w - k[:, 0]
    return np.ascontiguousarray(img[:, ::-1]), k[list(flip_idx)]


def default_albumentations_w1():
    """Ultralytics' default Albumentations (p = 0.01, the p = 0 ones omitted); None when Albumentations is missing."""
    try:
        import albumentations as A
    except ImportError:
        return None
    return [A.Blur(p=0.01), A.MedianBlur(p=0.01), A.ToGray(p=0.01), A.CLAHE(p=0.01)]


_PIXEL_CACHE: dict = {}  # only the real recipes are cached; ad-hoc (test) recipe names are rebuilt every time


def _pixel_pipeline(recipe: str):
    if recipe in _PIXEL_CACHE:
        return _PIXEL_CACHE[recipe]
    try:
        import albumentations as A
    except ImportError:  # optional, as in the YOLO recipes
        return None
    tfs = default_albumentations_w1() if recipe == "w1" else albumentations_for(recipe)
    pipe = A.Compose(tfs) if tfs else None
    if recipe in ("w1", "final"):
        _PIXEL_CACHE[recipe] = pipe
    return pipe


def _pixel_step(img, recipe: str, rng):
    pipe = _pixel_pipeline(recipe)
    if pipe is None:
        return img
    seed = int(rng.integers(2**31 - 1))
    if hasattr(pipe, "set_random_seed"):
        pipe.set_random_seed(seed)
    else:  # Albumentations 1.x: global generators
        import random

        random.seed(seed)
        np.random.seed(seed)
    return pipe(image=img)["image"]


def apply_recipe(img, kpts_px, recipe: str, rng, flip_idx=(1, 0, 3, 2)):
    """affine -> Albumentations -> HSV -> flip, numbers from RECIPES[recipe]; "none" = identity."""
    kpts_px = np.asarray(kpts_px, np.float64).reshape(-1, 2)
    if recipe == "none":
        return img, kpts_px
    cfg = RECIPES[recipe]
    img, kpts_px = random_affine(img, kpts_px, cfg["degrees"], cfg["translate"], cfg["scale"], rng)
    img = _pixel_step(img, recipe, rng)
    img = augment_hsv(img, cfg["hsv_h"], cfg["hsv_s"], cfg["hsv_v"], rng)
    if rng.random() < cfg["fliplr"]:
        img, kpts_px = fliplr(img, kpts_px, flip_idx)
    return img, kpts_px
