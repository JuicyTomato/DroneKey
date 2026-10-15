"""PnP sensitivity (plan step 2.3): GT keypoints + Gaussian pixel noise, and a wrongly scaled drone model."""
import math

import numpy as np

from dronepose.data.drone_models import DroneModel
from dronepose.pose.disambiguation import STRATEGIES, TemporalParams
from dronepose.pose.replay import replay

SIGMAS = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)
DIST_BINS = (0.0, 2.0, 3.0, 4.0, math.inf)
_BIN_NAMES = ("<2", "2-3", "3-4", ">=4")


def noisy_keypoints(records: list[dict], sigma: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for r in records:
        k = np.array(r["kpts2d"], float)
        if sigma > 0:
            k = k + rng.normal(0.0, sigma, k.shape)
        out[r["frame_id"]] = np.hstack([k, np.ones((4, 1))])
    return out


def frame_metrics(results_by_seq: dict[str, list[dict]], records: list[dict]) -> list[dict]:
    by_id = {r["frame_id"]: r for r in records}
    rows = []
    for res in (x for xs in results_by_seq.values() for x in xs):
        rec = by_id[res["frame_id"]]
        z_gt = float(rec["t_gt"][2])
        z = None if res["t"] is None else float(res["t"][2])
        rows.append({"frame_id": res["frame_id"], "drone": rec["drone"], "depth_gt": z_gt,
                     "rot_err_deg": res["rot_err_deg"],
                     "rel_depth_err": None if z is None else abs(z - z_gt) / z_gt,
                     "depth_ratio": None if z is None else z / z_gt})
    return rows


def _bin_name(depth: float) -> str:
    for lo, hi, name in zip(DIST_BINS[:-1], DIST_BINS[1:], _BIN_NAMES):
        if lo <= depth < hi:
            return name
    return _BIN_NAMES[-1]


def _summary(frames: list[dict]) -> dict:
    posed = [f for f in frames if f["rot_err_deg"] is not None]
    rot = [f["rot_err_deg"] for f in posed]
    return {"n": len(frames), "coverage": len(posed) / len(frames) if frames else None,
            "flip_rate": float(np.mean([e > 90 for e in rot])) if rot else None,
            "median_rot_err_deg": float(np.median(rot)) if rot else None,
            "p90_rot_err_deg": float(np.percentile(rot, 90)) if rot else None,
            "median_rel_depth_err": float(np.median([f["rel_depth_err"] for f in posed])) if posed else None,
            "median_depth_ratio": float(np.median([f["depth_ratio"] for f in posed])) if posed else None}


def aggregate(frames: list[dict], **labels) -> list[dict]:
    rows = [{**labels, "bin": "all", **_summary(frames)}]
    for name in _BIN_NAMES:
        sub = [f for f in frames if _bin_name(f["depth_gt"]) == name]
        rows.append({**labels, "bin": name, **_summary(sub)})
    return rows


def run_noise(records: list[dict], models: dict[str, DroneModel], sigmas=SIGMAS, seeds=(0,),
              strategies=STRATEGIES, params: TemporalParams = TemporalParams()) -> list[dict]:
    rows = []
    for sigma in sigmas:
        for seed in seeds:
            kpts = noisy_keypoints(records, sigma, seed)
            for strategy in strategies:
                frames = frame_metrics(replay(records, kpts, models, strategy, params), records)
                rows += aggregate(frames, sigma=sigma, seed=seed, strategy=strategy)
    return rows


def scaled_models(models: dict[str, DroneModel], s: float) -> dict[str, DroneModel]:
    return {d: DroneModel(m.kpts_flu * s, m.origin_offset_flu * s, m.spread) for d, m in models.items()}


def run_scale(records: list[dict], models: dict[str, DroneModel], scales=SCALES) -> list[dict]:
    kpts = noisy_keypoints(records, 0.0, 0)
    rows = []
    for s in scales:
        frames = frame_metrics(replay(records, kpts, scaled_models(models, s), "reproj"), records)
        a = _summary(frames)
        rows.append({"scale": s, "median_depth_ratio": a["median_depth_ratio"],
                     "median_rot_err_deg": a["median_rot_err_deg"]})
    return rows
