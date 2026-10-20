"""Keypoint metrics (pixels of the original frame): per-frame error, PCK, propeller identity, OKS, OKS-AP."""
from itertools import permutations

import numpy as np

_PERMS = list(permutations(range(4)))  # identity first
_OKS_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _dists(pred: np.ndarray, gt: np.ndarray) -> np.ndarray | None:
    """Per-propeller pixel distances (4,), or None if any predicted keypoint is not finite."""
    xy = np.asarray(pred, float)[:, :2]
    if not np.isfinite(xy).all():
        return None
    return np.linalg.norm(xy - np.asarray(gt, float)[:, :2], axis=1)


def frame_px_error(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Median pixel distance over the 4 propellers; None if the prediction is missing."""
    d = _dists(pred, gt)
    return None if d is None else float(np.median(d))


def pck(preds: list[np.ndarray], gts: list[np.ndarray], thresholds: list[float],
        scales: list[float] | None = None) -> dict[float, float]:
    """Fraction of keypoints (4 per frame) within threshold * scale; non-finite keypoints fail."""
    errs, sc = [], []
    for i, (p, g) in enumerate(zip(preds, gts)):
        xy = np.asarray(p, float)[:, :2]
        d = np.linalg.norm(xy - np.asarray(g, float)[:, :2], axis=1)
        errs.append(np.where(np.isfinite(xy).all(axis=1), d, np.inf))
        sc.append(np.full(4, 1.0 if scales is None else float(scales[i])))
    if not errs:
        return {t: 0.0 for t in thresholds}
    err, scale = np.concatenate(errs), np.concatenate(sc)
    return {t: float(np.mean(err <= t * scale)) for t in thresholds}


def identity_ok(pred: np.ndarray, gt: np.ndarray) -> bool | None:
    """True if the closest permutation of the 4 predictions to the GT is the identity; None if not finite."""
    xy = np.asarray(pred, float)[:, :2]
    if not np.isfinite(xy).all():
        return None
    g = np.asarray(gt, float)[:, :2]
    cost = [np.linalg.norm(xy[list(p)] - g, axis=1).sum() for p in _PERMS]
    return bool(int(np.argmin(cost)) == 0)


def oks(pred: np.ndarray, gt: np.ndarray, area: float) -> float:
    """Mean over the 4 propellers of exp(-d^2 / (0.2 * area)); 0 if the prediction is missing."""
    d = _dists(pred, gt)
    return 0.0 if d is None else float(np.mean(np.exp(-(d ** 2) / (0.2 * area))))


def success_rate(oks_values: list[float], threshold: float) -> float:
    """Fraction of frames with OKS >= threshold."""
    return float(np.mean(np.asarray(oks_values, float) >= threshold)) if len(oks_values) else 0.0


def oks_ap(oks_values: list[float], scores: list[float]) -> float:
    """COCO-style AP averaged over OKS thresholds 0.50:0.05:0.95; one GT per frame, score -inf = no prediction."""
    o, s = np.asarray(oks_values, float), np.asarray(scores, float)
    n = len(o)
    if n == 0:
        return 0.0
    present = s > -np.inf
    order = np.argsort(-s[present], kind="stable")
    o_ranked = o[present][order]
    recall_pts = np.linspace(0, 1, 101)
    aps = []
    for thr in _OKS_THRESHOLDS:
        tp = np.cumsum(o_ranked >= thr)
        recall = tp / n
        precision = tp / np.arange(1, len(tp) + 1)
        precision = np.maximum.accumulate(precision[::-1])[::-1]  # interpolation: max to the right
        idx = np.searchsorted(recall, recall_pts, side="left")  # first rank reaching each recall point
        q = np.zeros(101)
        ok = idx < len(precision)
        q[ok] = precision[idx[ok]]
        aps.append(q.mean())
    return float(np.mean(aps))
