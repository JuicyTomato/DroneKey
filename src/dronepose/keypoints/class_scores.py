"""Per-frame class scores (appearance, geometry) and their causal fusion into a drone class per frame.

`FrameScores` maps frame_id -> {drone: score}; a frame that is missing has no score. Higher is better.
"""
import numpy as np

from dronepose.data.sixdronesyn import DRONE_IDS
from dronepose.keypoints.detections import Detection
from dronepose.pose.replay import _by_sequence

FrameScores = dict[str, dict[str, float]]
TEMPORAL = ("F", "S")


def scores_c1(dets: dict[str, Detection], weighted: bool) -> FrameScores:
    """Top box of C only: its class scores 1.0 (or its confidence when `weighted`)."""
    return {fid: {d.drone: d.conf if weighted else 1.0} for fid, d in dets.items() if d.drone is not None}


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
