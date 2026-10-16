"""Drawing helpers for the forecast videos: back-projection to the camera, image overlay and top-down panel."""
import cv2
import numpy as np

from dronepose.common.geometry import project
from dronepose.forecast.frames import GRAVITY_FROM_CAMERA


def to_camera(p_rel: np.ndarray, last_p: np.ndarray, s_c: float | None) -> np.ndarray:
    """Window positions (relative, gravity frame, metres or dataset units) -> camera frame, dataset units."""
    p_G = np.asarray(p_rel, float) + np.asarray(last_p, float)
    if s_c is not None:
        p_G = p_G / s_c
    return p_G @ GRAVITY_FROM_CAMERA                            # inverse of p_G = C t  (C orthogonal)


def _polyline(img, pts_px: np.ndarray, color, thickness: int) -> None:
    if len(pts_px) > 1:
        cv2.polylines(img, [np.round(pts_px).astype(np.int32).reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def draw_camera_overlay(img: np.ndarray, K: np.ndarray, paths: dict[str, np.ndarray], colors: dict[str, tuple],
                        thickness: int = 4) -> np.ndarray:
    """Project 3D paths (camera frame) onto the image; points behind the camera are dropped."""
    out = img.copy()
    for name, P in paths.items():
        P = np.asarray(P, float)
        P = P[P[:, 2] > 1e-3]
        if len(P):
            _polyline(out, project(K, P), colors[name], thickness)
    return out


def nice_length(x: float) -> float:
    """Largest 1, 2 or 5 x 10^n not above x (scale bar length)."""
    e = 10 ** np.floor(np.log10(x))
    return float(max(m * e for m in (1, 2, 5) if m * e <= x))


class TopDownView:
    """Gravity-frame x (right) / y (forward from the camera) mapped to a square panel, equal scale on both axes."""

    def __init__(self, points_G: np.ndarray, size: int = 720, margin: int = 40, pad: float = 0.15):
        xy = np.asarray(points_G, float)[:, :2]
        lo, hi = xy.min(0), xy.max(0)
        span = float(max((hi - lo).max(), 1e-6)) * (1 + 2 * pad)
        self.center, self.size, self.margin = (lo + hi) / 2, size, margin
        self.scale = (size - 2 * margin) / span                 # pixels per unit

    def to_px(self, P: np.ndarray) -> np.ndarray:
        xy = (np.asarray(P, float)[:, :2] - self.center) * self.scale
        return np.stack([self.size / 2 + xy[:, 0], self.size / 2 - xy[:, 1]], axis=1)

    def blank(self, units: str) -> np.ndarray:
        panel = np.full((self.size, self.size, 3), 24, np.uint8)
        bar = nice_length((self.size - 2 * self.margin) / self.scale / 3)
        x0, y0 = self.margin, self.size - self.margin // 2
        cv2.line(panel, (x0, y0), (int(x0 + bar * self.scale), y0), (200, 200, 200), 2)
        cv2.putText(panel, f"{bar:g} {units}", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(panel, "vista dall'alto: x destra, y avanti (dalla camera)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return panel

    def draw(self, panel: np.ndarray, P: np.ndarray, color, thickness: int = 2) -> None:
        _polyline(panel, self.to_px(P), color, thickness)


def zoom_inset(img: np.ndarray, center_px: np.ndarray, crop: int, size: int) -> np.ndarray:
    """Square crop of side `crop` centred on `center_px` (black outside the image), resized to `size`."""
    cx, cy = np.round(np.asarray(center_px, float)).astype(int)
    half = crop // 2
    padded = cv2.copyMakeBorder(img, crop, crop, crop, crop, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x0, y0 = cx - half + crop, cy - half + crop
    x0 = int(np.clip(x0, 0, padded.shape[1] - crop))
    y0 = int(np.clip(y0, 0, padded.shape[0] - crop))
    return cv2.resize(padded[y0:y0 + crop, x0:x0 + crop], (size, size), interpolation=cv2.INTER_LINEAR)


def compose_frame(cam_img: np.ndarray, panel: np.ndarray, height: int = 720) -> np.ndarray:
    h, w = cam_img.shape[:2]
    cam = cv2.resize(cam_img, (int(round(w * height / h)), height), interpolation=cv2.INTER_AREA)
    side = cv2.resize(panel, (height, height), interpolation=cv2.INTER_AREA)
    return np.hstack([cam, side])
