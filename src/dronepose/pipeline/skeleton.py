"""Skeleton pipeline for one sequence: keypoints -> IPPE candidates -> disambiguation -> pose -> CV forecast.
Two passes, so a temporal strategy can see the whole sequence; assemble_results is shared with the offline replay."""
import math
from typing import Callable

import numpy as np

from dronepose.data.drone_models import DroneModel
from dronepose.data.sixdronesyn import FPS
from dronepose.eval.metrics import ade, fde, rotation_error_deg, translation_error
from dronepose.keypoints.predictors import KeypointPredictor
from dronepose.pose.disambiguation import Choice, disambiguate
from dronepose.pose.pnp import PnPSolution, solve_ippe
from dronepose.trajectory.baselines import constant_velocity


def ippe_candidates(record: dict, kpts: np.ndarray, models: dict[str, DroneModel]) -> list[PnPSolution]:
    try:
        return solve_ippe(models[record["drone"]].kpts_flu, np.asarray(kpts)[:, :2], np.array(record["K"]))
    except ValueError:
        return []


def _check_single_sequence(records: list[dict]) -> list[dict]:
    records = sorted(records, key=lambda r: r["frame_idx"])
    if len({r["frame_idx"] for r in records}) != len(records):
        raise ValueError("records contain more than one sequence (duplicate frame_idx): use a full sequence prefix")
    return records


def assemble_results(records: list[dict], kpts: list[np.ndarray], candidates: list[list[PnPSolution]],
                     choices: list[Choice], obs: int = 15, horizon: int = 30) -> list[dict]:
    by_idx = {r["frame_idx"]: r for r in records}
    results, history = [], []                       # history: (time, t_est) of accepted poses
    for r, kp, sols, ch in zip(records, kpts, candidates, choices):
        R_gt = np.array(r["R_gt"])
        out = {"frame_id": r["frame_id"], "frame_idx": r["frame_idx"], "kpts_pred": np.asarray(kp).tolist(),
               "R": None, "t": None, "rot_err_deg": None, "t_err": None,
               "future_pred": None, "future_gt": None, "ade": None, "fde": None,
               "solution_index": ch.index, "rejected": ch.rejected, "reject_reason": ch.reason,
               "reproj_ratio": None if math.isinf(ch.ratio) else ch.ratio,
               "oracle_rot_err_deg": min((rotation_error_deg(s.R, R_gt) for s in sols), default=None)}
        if ch.rejected:
            results.append(out)
            continue
        sol = ch.solution
        out.update(R=sol.R.tolist(), t=sol.t.tolist(), rot_err_deg=rotation_error_deg(sol.R, R_gt),
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


def run_sequence(records: list[dict], load_image: Callable[[dict], np.ndarray], predictor: KeypointPredictor,
                 models: dict[str, DroneModel], obs: int = 15, horizon: int = 30, strategy: str = "reproj",
                 params=None, up=None) -> list[dict]:
    records = _check_single_sequence(records)
    kpts = [predictor.predict(load_image(r), r) for r in records]
    cands = [ippe_candidates(r, kp, models) for r, kp in zip(records, kpts)]
    choices = disambiguate(cands, strategy, params, frame_idx=[r["frame_idx"] for r in records], up=up)
    return assemble_results(records, kpts, cands, choices, obs, horizon)


def summarize(results: list[dict]) -> dict:
    posed = [r for r in results if r["rot_err_deg"] is not None]
    fc = [r for r in results if r["ade"] is not None]
    with_cands = [r for r in results if r.get("oracle_rot_err_deg") is not None]
    med = lambda xs: float(np.median(xs)) if xs else None
    n = len(results)
    return {
        "n_frames": n, "n_pose": len(posed), "n_rejected": sum(bool(r.get("rejected")) for r in results),
        "coverage": len(posed) / n if n else None,
        "median_rot_err_deg": med([r["rot_err_deg"] for r in posed]),
        "p90_rot_err_deg": float(np.percentile([r["rot_err_deg"] for r in posed], 90)) if posed else None,
        "median_t_err": med([r["t_err"] for r in posed]),
        "flip_rate": float(np.mean([r["rot_err_deg"] > 90 for r in posed])) if posed else None,
        "oracle_flip_rate": float(np.mean([r["oracle_rot_err_deg"] > 90 for r in with_cands])) if with_cands else None,
        "median_ade": med([r["ade"] for r in fc]), "median_fde": med([r["fde"] for r in fc]),
    }
