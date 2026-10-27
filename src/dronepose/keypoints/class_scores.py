"""Per-frame class scores (appearance, geometry) and their causal fusion into a drone class per frame.

`FrameScores` maps frame_id -> {drone: score}; a frame that is missing has no score. Higher is better.
"""
import json
from pathlib import Path

import numpy as np

from dronepose.data.drone_models import DroneModel
from dronepose.data.export_yolo import CLASS_IDS
from dronepose.data.sixdronesyn import DRONE_IDS
from dronepose.eval.detection import iou_xyxy
from dronepose.keypoints.detections import Detection
from dronepose.pose.replay import _by_sequence
from dronepose.pose.solvers import solve

FrameScores = dict[str, dict[str, float]]
TEMPORAL = ("F", "S")
_DRONE_OF_CLASS = {i: d for d, i in CLASS_IDS.items()}


def scores_c1(dets: dict[str, Detection], weighted: bool) -> FrameScores:
    """Top box of C only: its class scores 1.0 (or its confidence when `weighted`)."""
    return {fid: {d.drone: d.conf if weighted else 1.0} for fid, d in dets.items() if d.drone is not None}


def load_boxes(path: Path) -> dict[str, list[list[float]]]:
    """`boxes` of every row of an all-boxes export of C (rows without the field give an empty list)."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["frame_id"]: row.get("boxes") or [] for row in rows}


def scores_cm(boxes: dict[str, list[list[float]]], iou_min: float, top_conf_min: float = 0.25) -> FrameScores:
    """Several boxes of C: per class, the highest confidence among boxes with IoU >= iou_min with the top box. Only
    frames where C has a detection (top box conf >= top_conf_min, the threshold of C's top-box file) are scored, so
    CM differs from C1 by the extra classes, not by a lower detection threshold."""
    out: FrameScores = {}
    for fid, bs in boxes.items():
        if not bs:
            continue
        top = max(bs, key=lambda b: b[4])
        if top[4] < top_conf_min:
            continue
        sc: dict[str, float] = {}
        for b in bs:
            if iou_xyxy(b[:4], top[:4]) >= iou_min:
                c = _DRONE_OF_CLASS[int(b[5])]
                sc[c] = max(sc.get(c, 0.0), float(b[4]))
        out[fid] = sc
    return out


def probs_row(frame_id: str, probs: np.ndarray | None, names: dict[int, str]) -> dict:
    """JSONL row of the crop classifier: probabilities keyed by drone name (never by index)."""
    if set(names.values()) != set(DRONE_IDS):
        raise ValueError(f"classifier names {sorted(names.values())} are not the drones {list(DRONE_IDS)}")
    if probs is None:
        return {"frame_id": frame_id, "probs": None}
    return {"frame_id": frame_id, "probs": {names[i]: float(p) for i, p in enumerate(np.asarray(probs, float))}}


def load_probs(path: Path) -> dict[str, dict[str, float] | None]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["frame_id"]: row["probs"] for row in rows}


def scores_k(probs: dict[str, dict[str, float] | None]) -> FrameScores:
    """Crop classifier: softmax probability per class; frames without a crop have no score."""
    return {fid: dict(p) for fid, p in probs.items() if p is not None}


def log_scores(s: FrameScores, a_min: float) -> FrameScores:
    """log(max(a, a_min)) over all 7 classes; a class missing from a frame counts as a_min."""
    return {fid: {c: float(np.log(max(sc.get(c, 0.0), a_min))) for c in DRONE_IDS} for fid, sc in s.items()}


def combine(parts: list[tuple[FrameScores, float]]) -> FrameScores:
    """Weighted sum; a frame is scored if any part scores it, a class missing from a part adds 0."""
    out: FrameScores = {}
    for scores, w in parts:
        for fid, sc in scores.items():
            acc = out.setdefault(fid, {})
            for c, v in sc.items():
                acc[c] = acc.get(c, 0.0) + w * v
    return out


def reproj_per_class(record: dict, kpts: np.ndarray, models: dict[str, DroneModel]) -> dict[str, float] | None:
    """Least reprojection error (px) of IPPE + LM with each drone model; None when the keypoints are not finite or
    the PnP fails for any model (the frame then gives no geometric evidence at all)."""
    img = np.asarray(kpts, float)[:, :2]
    if not np.isfinite(img).all():
        return None
    K = np.asarray(record["K"], float)
    out = {}
    for d, m in models.items():
        try:
            sols = solve("ippe", m.kpts_flu, img, K, refine=True)
        except ValueError:
            return None
        if not sols:
            return None
        out[d] = float(min(s.reproj_px for s in sols))
    return out


def scores_geom(reproj: dict[str, dict[str, float] | None], form: str, eps: float = 0.1, cap: float = 50.0,
                sigma: float = 1.0) -> FrameScores:
    """Geometric scores from `reproj_per_class` rows: log = -log(min(e, cap) + eps), quad = -min(e, cap)^2 / 2 sigma^2."""
    if form not in ("log", "quad"):
        raise ValueError(f"unknown geometry form {form!r}: expected 'log' or 'quad'")
    out: FrameScores = {}
    for fid, e in reproj.items():
        if e is None:
            continue
        out[fid] = {c: float(-np.log(min(v, cap) + eps)) if form == "log" else -min(v, cap) ** 2 / (2 * sigma ** 2)
                    for c, v in e.items()}
    return out


def _argmax(sc: dict[str, float], order: list[str]) -> str:
    """Best class; ties go to the class earliest in `order`, then in DRONE_IDS order."""
    rank = {c: i for i, c in enumerate(order + [c for c in DRONE_IDS if c not in order])}
    return max(sc, key=lambda c: (sc[c], -rank.get(c, len(rank))))


def fuse_causal(records: list[dict], scores: FrameScores, temporal: str) -> dict[str, str]:
    """Class per frame from scores up to that frame. F = the frame's own best class (frames without a score keep the
    previous class); S = best cumulative sum. Ties go to the class that was first the best of a frame in the sequence;
    frames before the first score take the class of the first scored frame (as `class_causal_vote`)."""
    if temporal not in TEMPORAL:
        raise ValueError(f"unknown temporal rule {temporal!r}: expected one of {TEMPORAL}")
    out = {}
    for key, recs in _by_sequence(records).items():
        order: list[str] = []
        total: dict[str, float] = {}
        labels: list[str | None] = []
        current = None
        for r in recs:
            sc = scores.get(r["frame_id"])
            if sc:
                best = _argmax(sc, order)
                if best not in order:
                    order.append(best)
                for c, v in sc.items():
                    total[c] = total.get(c, 0.0) + v
                current = best if temporal == "F" else _argmax(total, order)
            labels.append(current)
        first = next((c for c in labels if c is not None), None)
        if first is None:
            raise ValueError(f"sequence {key}: no frame with a class score")
        out.update({r["frame_id"]: c or first for r, c in zip(recs, labels)})
    return out
