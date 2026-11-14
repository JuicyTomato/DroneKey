"""Frame reduction seen by the detector C: 1920x1080 -> 640x360 INTER_AREA -> JPEG 92 (as scripts/build_yolo_frames.py)."""
import cv2
import numpy as np


def reduce_frame(image: np.ndarray, width: int = 640, jpeg_quality: int = 92) -> np.ndarray:
    """Resize to `width` (aspect kept, INTER_AREA) and round-trip through JPEG in memory."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR)
