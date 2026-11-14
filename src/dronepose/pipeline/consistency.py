"""Live perception vs the exported files with the same weights (spec 2026-11-14 §6)."""
import numpy as np

from dronepose.eval.detection import iou_xyxy
from dronepose.keypoints.detections import Detection
from dronepose.pipeline.perception import Perception

THRESHOLDS = {"same_presence": 0.995, "iou_ge_099": 0.99, "median_px": 0.1, "p99_px": 1.0, "same_argmax": 0.995,
              "max_abs": 0.02}


def compare_boxes(live: dict[str, Perception], ref: dict[str, Detection]) -> dict:
    ids = sorted(set(live) & set(ref))
    same = [(live[i].box is not None) == (ref[i].box is not None) for i in ids]
    both = [i for i in ids if live[i].box is not None and ref[i].box is not None]
    ious = [iou_xyxy(live[i].box, ref[i].box) for i in both]
    return {"n": len(ids), "same_presence": float(np.mean(same)) if ids else float("nan"),
            "iou_ge_099": float(np.mean(np.array(ious) >= 0.99)) if ious else float("nan"),
            "iou_min": float(min(ious)) if ious else float("nan")}


def compare_kpts(live: dict[str, Perception], ref: dict[str, np.ndarray]) -> dict:
    d = []
    for i in sorted(set(live) & set(ref)):
        a, b = np.asarray(live[i].kpts, float)[:, :2], np.asarray(ref[i], float)[:, :2]
        if np.isfinite(a).all() and np.isfinite(b).all():
            d.append(np.linalg.norm(a - b, axis=1))
    d = np.concatenate(d) if d else np.zeros(0)
    return {"n": len(d) // 4, "median_px": float(np.median(d)) if len(d) else float("nan"),
            "p99_px": float(np.percentile(d, 99)) if len(d) else float("nan"),
            "max_px": float(d.max()) if len(d) else float("nan")}


def compare_probs(live: dict[str, Perception], ref: dict[str, dict | None]) -> dict:
    same, diff = [], []
    for i in sorted(set(live) & set(ref)):
        a, b = live[i].probs, ref[i]
        if a is None or b is None:
            continue
        same.append(max(a, key=a.get) == max(b, key=b.get))
        diff.append(max(abs(a[c] - b.get(c, 0.0)) for c in a))
    return {"n": len(same), "same_argmax": float(np.mean(same)) if same else float("nan"),
            "max_abs": float(max(diff)) if diff else float("nan")}


def verdicts(boxes: dict, kpts: dict, probs: dict) -> dict[str, bool]:
    t = THRESHOLDS
    return {"boxes": boxes["same_presence"] >= t["same_presence"] and boxes["iou_ge_099"] >= t["iou_ge_099"],
            "keypoints": kpts["median_px"] < t["median_px"] and kpts["p99_px"] < t["p99_px"],
            "probs": probs["same_argmax"] >= t["same_argmax"] and probs["max_abs"] < t["max_abs"]}
