"""One video frame of the demo: GT keypoints (green rings, when known), predicted keypoints (red), box of C (cyan),
body axes, 1 s forecast (yellow), and a text band above the image (mode, frame, class, time per frame)."""
import cv2
import numpy as np

from dronepose.pipeline.perception import Perception
from dronepose.pipeline.sources import Frame
from dronepose.viz.overlay import draw_overlay

OUT_W, IMG_H, BAND = 960, 540, 64
OUT_H = IMG_H + BAND


def draw_frame(frame: Frame, perc: Perception, res: dict, future_cam: np.ndarray | None, mode: str,
               ms_total: float | None) -> np.ndarray:
    gt = (frame.record or {}).get("kpts2d") or np.zeros((0, 2))
    img = draw_overlay(frame.image, frame.K, gt, perc.kpts, res.get("R"), res.get("t"), future_cam,
                       dist=getattr(frame, "dist", None))
    if perc.box is not None:
        x0, y0, x1, y1 = (int(v) for v in perc.box)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 2)
    img = cv2.resize(img, (OUT_W, IMG_H), interpolation=cv2.INTER_AREA)
    band = np.zeros((BAND, OUT_W, 3), np.uint8)
    cls = res.get("drone_model") or "-"
    true = (frame.record or {}).get("drone")
    pose = "posa" if res.get("R") is not None else "nessuna posa"
    line1 = f"{mode}  {frame.frame_id}  classe {cls}" + (f" (vera {true})" if true else "") + f"  {pose}"
    line2 = "previsione 1 s: " + ("si" if future_cam is not None else "no") + \
        (f"   {ms_total:.0f} ms/frame" if ms_total is not None else "")
    for i, s in enumerate((line1, line2)):
        cv2.putText(band, s, (10, 24 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, img])
