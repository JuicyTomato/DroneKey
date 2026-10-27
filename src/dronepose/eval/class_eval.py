"""Drone-class quality of a per-frame class map: accuracy per frame, drone, scene, and at the end of each sequence."""
from collections import defaultdict

import numpy as np

from dronepose.pose.replay import _by_sequence


def scene_of(frame_id: str) -> str:
    return frame_id[:2]


def correct_rows(records: list[dict], cls_map: dict[str, str]) -> list[dict]:
    """One row per record with `correct` 1.0/0.0, for `by_trajectory(rows, "correct")`."""
    return [{"frame_id": r["frame_id"], "correct": float(cls_map[r["frame_id"]] == r["drone"])} for r in records]


def class_metrics(records: list[dict], cls_map: dict[str, str]) -> dict:
    """Accuracies over all frames of `records` (frames without a detection included: every frame has a class)."""
    by_drone: dict[str, list[bool]] = defaultdict(list)
    by_scene: dict[str, list[bool]] = defaultdict(list)
    ok = []
    for r in records:
        c = cls_map[r["frame_id"]] == r["drone"]
        ok.append(c)
        by_drone[r["drone"]].append(c)
        by_scene[scene_of(r["frame_id"])].append(c)
    final = [cls_map[recs[-1]["frame_id"]] == recs[-1]["drone"] for recs in _by_sequence(records).values()]
    return {"n_frames": len(ok), "frame_acc": float(np.mean(ok)),
            "by_drone": {d: float(np.mean(v)) for d, v in sorted(by_drone.items())},
            "by_scene": {s: float(np.mean(v)) for s, v in sorted(by_scene.items())},
            "seq_final_acc": float(np.mean(final))}
