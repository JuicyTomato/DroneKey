"""Drone class K+L+g, one frame at a time (spec 2026-11-14 §5.4): the recipe of scripts/class/eval_class.py
(log of K's probabilities + w_g x log geometry, cumulative sum "S") as a library, strictly causal.

Unlike `fuse_causal`, frames before the first score get no class (None) instead of the first scored frame's class.
"""
import json
from pathlib import Path

from dronepose.keypoints.class_scores import _argmax, combine, log_scores, scores_geom

FINAL_PARAMS = Path("experiments/class/eval/final_params.json")


def load_klg_params(path: Path = FINAL_PARAMS) -> dict:
    return dict(json.loads(Path(path).read_text())["best_measured"]["params"])


def klg_frame_scores(probs: dict | None, reproj: dict | None, params: dict) -> dict[str, float] | None:
    """Scores of one frame; None when neither the classifier nor the geometry has evidence."""
    parts = []
    if probs is not None:
        parts.append((combine([(log_scores({"f": probs}, params["a_min"]), 1.0)]), 1.0))
    if reproj is not None:
        parts.append((scores_geom({"f": reproj}, params["form"]), params["w_g"]))
    return combine(parts).get("f") if parts else None


class CausalClassFuser:
    """Cumulative-sum class ("S" of `fuse_causal`), updated frame by frame; same tie rules."""

    def __init__(self):
        self.order: list[str] = []
        self.total: dict[str, float] = {}
        self.current: str | None = None

    def update(self, scores: dict[str, float] | None) -> str | None:
        if scores:
            best = _argmax(scores, self.order)
            if best not in self.order:
                self.order.append(best)
            for c, v in scores.items():
                self.total[c] = self.total.get(c, 0.0) + v
            self.current = _argmax(self.total, self.order)
        return self.current
