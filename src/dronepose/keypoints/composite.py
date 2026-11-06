"""Background copy-paste and motion blur, applied to the image before every other augmentation.

The same Composer serves DroneKey (CropDataset) and YOLO (yolo_composite.CompositeDataset). Keypoints never move: the
drone stays where it is, the blur kernel is symmetric, so the label stays at the middle of the motion. With a paste the
blur touches the drone layer only (fixed camera, as in MAV6D); without one it touches the whole image, because there
is no clean background to put under the drone.
"""
import math
from pathlib import Path

import cv2
import numpy as np

from dronepose.data.masks import geom_key
from dronepose.keypoints.recipes import COMPOSITE, CompositeSpec


def motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    """Normalised line kernel, odd side, invariant under a 180 degree rotation; length < 1 is the identity."""
    if length < 1:
        return np.ones((1, 1), np.float32)
    half = math.ceil(length / 2) + 1
    side = 2 * half + 1
    a = math.radians(angle_deg)
    t = np.linspace(-length / 2, length / 2, max(4 * side, 32))
    x, y = half + t * math.cos(a), half + t * math.sin(a)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    k = np.zeros((side, side), np.float64)
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(k, (y0 + dy, x0 + dx), w)
    k = (k + k[::-1, ::-1]) / 2     # floor() is not symmetric to the last bit
    return (k / k.sum()).astype(np.float32)


def sample_background(bg: np.ndarray, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Random window of 1-2 times the target inside `bg`, flip with p 0.5, resized to (h, w). When that window does not
    fit (a frame-sized target on a background of the same width) it is the largest one times 0.5-1, so the scale still
    varies; small sources are enlarged."""
    bh, bw = bg.shape[:2]
    s = rng.uniform(1, 2)
    fit = min(1.0, bh / (h * s), bw / (w * s))
    if fit < 1:
        fit *= rng.uniform(0.5, 1)
    ch, cw = max(1, min(bh, round(h * s * fit))), max(1, min(bw, round(w * s * fit)))
    y, x = int(rng.integers(0, bh - ch + 1)), int(rng.integers(0, bw - cw + 1))
    out = bg[y:y + ch, x:x + cw]
    if rng.random() < 0.5:
        out = out[:, ::-1]
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA if ch > h else cv2.INTER_LINEAR)


def paste(img: np.ndarray, alpha: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """alpha * img + (1 - alpha) * bg, uint8; alpha is float in [0, 1] with shape (H, W)."""
    a = alpha[..., None]
    return np.clip(np.rint(a * img + (1 - a) * bg), 0, 255).astype(np.uint8)


def blur_premultiplied(img: np.ndarray, alpha: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blur a layer with premultiplied alpha, so that the colour of what is outside the mask does not leak in."""
    a = cv2.filter2D(alpha.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    pre = cv2.filter2D(img.astype(np.float32) * alpha[..., None].astype(np.float32), -1, kernel,
                       borderType=cv2.BORDER_CONSTANT)
    return np.where(a[..., None] > 1e-3, pre / np.maximum(a[..., None], 1e-3), 0.0), a


def _listing(folder: Path, pattern: str) -> list[Path]:
    """Sorted files, without the AppleDouble `._*` companions that macOS writes on the external disk."""
    return [p for p in sorted(Path(folder).glob(pattern)) if not p.name.startswith(".")]


class Composer:
    """Callable (img, frame_id, rng) -> img. A frame without a mask is never pasted: it keeps its own background."""

    def __init__(self, spec: CompositeSpec, masks_dir: Path | None, bg_dir: Path | None, blur_scale: float = 1.0):
        self.spec = spec
        self.blur_max = spec.blur_max * blur_scale
        self.masks: dict[str, Path] = {}
        self.backgrounds: list[Path] = []
        if spec.p_paste > 0:
            if masks_dir is None or bg_dir is None:
                raise ValueError("copy-paste needs masks_dir and bg_dir")
            self.masks = {p.stem: p for p in _listing(masks_dir, "*.png")}
            self.backgrounds = _listing(bg_dir, "*.jpg")
            if not self.masks:
                raise FileNotFoundError(f"no mask in {masks_dir}")
            if not self.backgrounds:
                raise FileNotFoundError(f"no background in {bg_dir}")

    @property
    def n_masks(self) -> int:
        return len(self.masks)

    @property
    def n_backgrounds(self) -> int:
        return len(self.backgrounds)

    def _alpha(self, path: Path, h: int, w: int) -> np.ndarray:
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(path)
        if abs(m.shape[0] * w - m.shape[1] * h) > 0.02 * m.shape[0] * w:   # crop masks on frames would be misplaced
            raise ValueError(f"mask aspect {m.shape} does not match image {(h, w)}: {path}")
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return m.astype(np.float32) / 255.0

    def _background(self, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
        path = self.backgrounds[int(rng.integers(len(self.backgrounds)))]
        bg = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bg is None:
            raise FileNotFoundError(path)
        return sample_background(bg, h, w, rng)

    def __call__(self, img: np.ndarray, frame_id: str, rng: np.random.Generator) -> np.ndarray:
        spec = self.spec
        mask = self.masks.get(geom_key(frame_id)) if self.masks else None
        pasted = mask is not None and rng.random() < spec.p_paste
        kernel = None
        if spec.p_blur > 0 and rng.random() < spec.p_blur:
            kernel = motion_kernel(rng.uniform(0, self.blur_max), rng.uniform(0, 180))
        h, w = img.shape[:2]
        if pasted:
            alpha = self._alpha(mask, h, w)
            layer = (img, alpha) if kernel is None else blur_premultiplied(img, alpha, kernel)
            return paste(layer[0], layer[1], self._background(h, w, rng))
        return img if kernel is None else cv2.filter2D(img, -1, kernel)


def composer_for(recipe: str, masks_dir: Path | None, bg_dir: Path | None, blur_scale: float = 1.0) -> Composer | None:
    """The Composer of a recipe, None for recipes without composition (w1, final, none)."""
    spec = COMPOSITE.get(recipe)
    return None if spec is None else Composer(spec, masks_dir, bg_dir, blur_scale)
