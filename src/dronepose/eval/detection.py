"""Detection quality of setting C: IoU with the label box, class accuracy, no-detection rate."""
from collections import defaultdict

import numpy as np

from dronepose.data.crops import label_box_xyxy
from dronepose.keypoints.detections import Detection


def iou_xyxy(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detection_metrics(dets: dict[str, Detection], records: list[dict]) -> dict:
    """IoU stats are over detected frames only; no_det_rate counts frames without a box or without a row."""
    ious, correct = [], []
    by_drone: dict[str, list[bool]] = defaultdict(list)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    nodet = 0
    for r in records:
        d = dets.get(r["frame_id"])
        if d is None or d.box is None:
            nodet += 1
            continue
        ious.append(iou_xyxy(d.box, label_box_xyxy(np.array(r["kpts2d"], float))))
        pred = d.drone
        by_drone[r["drone"]].append(pred == r["drone"])
        correct.append(pred == r["drone"])
        confusion[r["drone"]][str(pred)] += 1
    n = len(records)
    iou = np.array(ious)
    return {
        "n_frames": n,
        "no_det_rate": nodet / n if n else None,
        "iou_median": float(np.median(iou)) if len(iou) else None,
        "iou_p10": float(np.percentile(iou, 10)) if len(iou) else None,
        "iou_ge_05": float(np.mean(iou >= 0.5)) if len(iou) else None,
        "class_acc": float(np.mean(correct)) if correct else None,
        "class_acc_by_drone": {d: float(np.mean(v)) for d, v in sorted(by_drone.items())},
        "confusion": {t: dict(sorted(p.items())) for t, p in sorted(confusion.items())},
    }
