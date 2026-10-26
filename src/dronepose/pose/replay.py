"""Offline replay: run the pose stage on keypoints read from a file (no GPU), one sequence at a time."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dronepose.common.geometry import geodesic_deg
from dronepose.data.drone_models import DroneModel
from dronepose.keypoints.predictors import no_detection
from dronepose.pipeline.skeleton import assemble_results, ippe_candidates, summarize
from dronepose.pose.disambiguation import disambiguate


def sequence_key(frame_id: str) -> str:
    return frame_id.rsplit("_", 1)[0]


def load_keypoints(path: Path) -> dict[str, np.ndarray]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["frame_id"]] = np.array(row["kpts_pred"], float)
    return out


def check_no_test(records: list[dict], allow_test: bool) -> None:
    if not allow_test and any(r["split"] == "test" for r in records):
        raise ValueError("records include the test split: thresholds are tuned on val only (pass allow_test to override)")


def _by_sequence(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[sequence_key(r["frame_id"])].append(r)
    return {k: sorted(v, key=lambda r: r["frame_idx"]) for k, v in sorted(groups.items())}


def replay(records: list[dict], kpts_by_id: dict[str, np.ndarray], models: dict[str, DroneModel],
           strategy: str = "reproj", params=None, obs: int = 15, horizon: int = 30,
           up=None, model_drone: dict[str, str] | None = None, solver: str = "ippe",
           refine: bool = False) -> dict[str, list[dict]]:
    """Replay saved keypoints; `model_drone[frame_id]` overrides the 3D model (metrics stay on the true drone).
    `solver`/`refine` pick the PnP solver (see `dronepose.pose.solvers`); the default is the original IPPE path."""
    out = {}
    for key, recs in _by_sequence(records).items():
        kpts = [kpts_by_id.get(r["frame_id"], no_detection()) for r in recs]
        cands = [ippe_candidates(r, kp, models, model_drone[r["frame_id"]] if model_drone else None,
                                solver, refine)
                 for r, kp in zip(recs, kpts)]
        choices = disambiguate(cands, strategy, params, frame_idx=[r["frame_idx"] for r in recs], up=up)
        out[key] = assemble_results(recs, kpts, cands, choices, obs, horizon)
        for res in out[key]:
            res["kpts_missing"] = res["frame_id"] not in kpts_by_id
    return out


def strategy_table(results_by_seq: dict[str, list[dict]]) -> dict:
    allres = [r for rs in results_by_seq.values() for r in rs]
    by_drone: dict[str, list[dict]] = defaultdict(list)
    for key, rs in results_by_seq.items():
        by_drone[key.split("_")[1]].extend(rs)
    return {"all": summarize(allres), "by_drone": {d: summarize(rs) for d, rs in sorted(by_drone.items())},
            "n_missing_kpts": sum(bool(r.get("kpts_missing")) for r in allres)}


def gt_rotation_steps_deg(records: list[dict]) -> np.ndarray:
    steps = []
    for recs in _by_sequence(records).values():
        for a, b in zip(recs[:-1], recs[1:]):
            steps.append(geodesic_deg(np.array(a["R_gt"]), np.array(b["R_gt"])))
    return np.array(steps)
