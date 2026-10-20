"""Per-frame detections (box, confidence, class) read from a keypoint JSONL, and the predicted drone class per frame."""
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dronepose.data.export_yolo import CLASS_IDS
from dronepose.pose.replay import _by_sequence

_DRONE_OF_CLASS = {i: d for d, i in CLASS_IDS.items()}


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float] | None
    conf: float
    drone: str | None


def load_detections(path: Path) -> dict[str, Detection]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                box = row.get("box")
                cls = row.get("cls")
                out[row["frame_id"]] = Detection(
                    None if box is None else tuple(float(v) for v in box),
                    float(row.get("box_conf") or 0.0),
                    None if cls is None else _DRONE_OF_CLASS[int(cls)])
    return out


def _classes(recs: list[dict], dets: dict[str, Detection]) -> list[str | None]:
    return [dets[r["frame_id"]].drone if r["frame_id"] in dets else None for r in recs]


def _first_class(key: str, classes: list[str | None]) -> str:
    first = next((c for c in classes if c is not None), None)
    if first is None:
        raise ValueError(f"sequence {key}: no detection with a class")
    return first


def class_per_frame(records: list[dict], dets: dict[str, Detection]) -> dict[str, str]:
    """Class of each frame's detection; frames without one take the class of the sequence's first detection."""
    out = {}
    for key, recs in _by_sequence(records).items():
        classes = _classes(recs, dets)
        first = _first_class(key, classes)
        out.update({r["frame_id"]: c or first for r, c in zip(recs, classes)})
    return out


def class_causal_vote(records: list[dict], dets: dict[str, Detection]) -> dict[str, str]:
    """Majority class from the first detection up to the current frame; ties go to the class seen first."""
    out = {}
    for key, recs in _by_sequence(records).items():
        classes = _classes(recs, dets)
        first = _first_class(key, classes)
        votes: Counter = Counter()
        for r, c in zip(recs, classes):
            if c is not None:
                votes[c] += 1
            out[r["frame_id"]] = votes.most_common(1)[0][0] if votes else first   # insertion order breaks ties
    return out
