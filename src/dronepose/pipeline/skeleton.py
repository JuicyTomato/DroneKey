"""Skeleton pipeline for one sequence: keypoints -> IPPE pose -> constant-velocity forecast."""
from typing import Callable

import numpy as np

from dronepose.data.drone_models import DroneModel
from dronepose.data.sixdronesyn import FPS
from dronepose.eval.metrics import ade, fde, rotation_error_deg, translation_error
from dronepose.keypoints.predictors import KeypointPredictor
from dronepose.pose.pnp import estimate_pose
from dronepose.trajectory.baselines import constant_velocity


def run_sequence(records: list[dict], load_image: Callable[[dict], np.ndarray], predictor: KeypointPredictor,
                 models: dict[str, DroneModel], obs: int = 15, horizon: int = 30) -> list[dict]:
    records = sorted(records, key=lambda r: r["frame_idx"])
    by_idx = {r["frame_idx"]: r for r in records}
    if len(by_idx) != len(records):
        raise ValueError("records contain more than one sequence (duplicate frame_idx): use a full sequence prefix")
    results, history = [], []                        # history: (time, t_est) of valid poses
    for r in records:
        K = np.array(r["K"])
        kp = predictor.predict(load_image(r), r)
        out = {"frame_id": r["frame_id"], "frame_idx": r["frame_idx"], "kpts_pred": kp.tolist(),
               "R": None, "t": None, "rot_err_deg": None, "t_err": None,
               "future_pred": None, "future_gt": None, "ade": None, "fde": None}
        try:
            sol = estimate_pose(models[r["drone"]].kpts_flu, kp[:, :2], K)
        except ValueError:
            results.append(out)
            continue
        out.update(R=sol.R.tolist(), t=sol.t.tolist(),
                   rot_err_deg=rotation_error_deg(sol.R, np.array(r["R_gt"])),
                   t_err=translation_error(sol.t, np.array(r["t_gt"])))
        history.append((r["frame_idx"] / FPS, sol.t))
        future_idx = [r["frame_idx"] + k for k in range(1, horizon + 1)]
        if len(history) >= obs and all(i in by_idx for i in future_idx):
            past = history[-obs:]
            pred = constant_velocity(np.array([h[0] for h in past]), np.array([h[1] for h in past]),
                                     np.array(future_idx) / FPS)
            gt = np.array([by_idx[i]["t_gt"] for i in future_idx])
            out.update(future_pred=pred.tolist(), future_gt=gt.tolist(), ade=ade(pred, gt), fde=fde(pred, gt))
        results.append(out)
    return results


def summarize(results: list[dict]) -> dict:
    posed = [r for r in results if r["rot_err_deg"] is not None]
    fc = [r for r in results if r["ade"] is not None]
    med = lambda xs: float(np.median(xs)) if xs else None
    return {
        "n_frames": len(results), "n_pose": len(posed),
        "median_rot_err_deg": med([r["rot_err_deg"] for r in posed]),
        "median_t_err": med([r["t_err"] for r in posed]),
        "flip_rate": float(np.mean([r["rot_err_deg"] > 90 for r in posed])) if posed else None,
        "median_ade": med([r["ade"] for r in fc]), "median_fde": med([r["fde"] for r in fc]),
    }
