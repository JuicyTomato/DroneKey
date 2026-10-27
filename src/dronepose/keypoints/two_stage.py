"""Setting B/C helpers: JSONL row of the top detection, and the crop taken from a detected box (pure NumPy + OpenCV)."""
import cv2
import numpy as np

from dronepose.data.crops import CropBox, crop_box_from_det_box, crop_image
from dronepose.keypoints.predictors import no_detection


def top_detection_row(frame_id: str, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, kxy: np.ndarray,
                      kconf: np.ndarray | None, scale: float) -> dict:
    """JSONL row of the highest-confidence detection (coordinates x scale); no detection -> NaN keypoints."""
    if xyxy is None or len(conf) == 0:
        return {"frame_id": frame_id, "kpts_pred": no_detection().tolist(), "box": None, "box_conf": 0.0,
                "cls": None}
    j = int(np.argmax(conf))
    kc = np.ones(4) if kconf is None else np.asarray(kconf[j], float)
    kpts = np.hstack([np.asarray(kxy[j], float) * scale, kc[:, None]])
    return {"frame_id": frame_id, "kpts_pred": kpts.tolist(), "box": (np.asarray(xyxy[j], float) * scale).tolist(),
            "box_conf": float(conf[j]), "cls": int(cls[j])}


def all_boxes(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, scale: float) -> list[list[float]]:
    """Every detection as [x1, y1, x2, y2, conf, cls] (coordinates x scale), by decreasing confidence."""
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    return [[*(float(v) * scale for v in xyxy[j]), float(conf[j]), float(cls[j])] for j in order]


def crop_from_detection(img: np.ndarray, xyxy, jpeg_quality: int = 95) -> tuple[np.ndarray, CropBox]:
    """Crop around a detected box with the setting-A rule, through JPEG as the training crops were."""
    box = crop_box_from_det_box(np.asarray(xyxy, float))
    ok, jpg = cv2.imencode(".jpg", crop_image(img, box), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR), box
