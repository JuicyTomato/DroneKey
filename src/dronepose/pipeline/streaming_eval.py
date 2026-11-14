"""Offline vs streaming on the 6DroneSyn val (spec 2026-11-14 §7): pose summaries, choice of W and of the
observation mode (e = emitted poses, r = poses re-decided at the window's last frame), forecast errors."""
import numpy as np

from dronepose.eval.bootstrap import paired_bootstrap
from dronepose.forecast.e4_eval import degradation, method_errors
from dronepose.forecast.gaps import drop_sparse
from dronepose.pipeline.skeleton import summarize

MIN_COVERAGE = 0.95
NET = "R-CV-6D"


def _drone(r: dict) -> str:
    return r["frame_id"].split("_")[1]


def pose_summary(results: list[dict]) -> dict:
    s = summarize(results)
    posed = [r for r in results if r["rot_err_deg"] is not None and _drone(r) != "08"]
    s["flip_rate_no08"] = float(np.mean([r["rot_err_deg"] > 90 for r in posed])) if posed else None
    return s


def select_window(summaries: dict[int, dict], rot_by_traj: dict[int, dict[str, list[float]]], n_boot: int = 2000,
                  seed: int = 0) -> dict:
    """Coverage >= 95%, then lowest median rotation; W tied with the best (bootstrap interval of the difference
    includes 0) -> the smallest W."""
    eligible = sorted(W for W, s in summaries.items() if s["coverage"] is not None and s["coverage"] >= MIN_COVERAGE)
    if not eligible:
        raise ValueError("no W reaches the coverage threshold")
    best = min(eligible, key=lambda W: summaries[W]["median_rot_err_deg"])
    boot = paired_bootstrap({str(W): rot_by_traj[W] for W in eligible}, ref=str(best), n_boot=n_boot, seed=seed)
    tied = [W for W in eligible if W == best or boot[str(W)][1] <= 0 <= boot[str(W)][2]]
    return {"chosen": min(tied), "best": best, "eligible": eligible, "tied": tied,
            "ci_vs_best": {W: boot[str(W)] for W in eligible}}


def select_obs_mode(err_e: dict, err_r: dict, windows, b: int) -> dict:
    """(r) only if its ADE at 1 s is lower than (e) with an interval that excludes 0; otherwise (e)."""
    d = degradation(err_r, err_e, windows, b)[NET]
    return {"chosen": "r" if d["ci"][1] < 0 else "e", "r_minus_e": d}


def window_errors(w_all, cfg, params_for, nets_for, device) -> tuple[dict, np.ndarray, list]:
    """As `pipeline_errors`, on windows already built: position errors per method, GT window index, row keys."""
    w_all, _keep = drop_sparse(w_all)
    errs, idx, keys = [], [], []
    for scene in sorted(set(w_all.scene)):
        sel = (w_all.scene == scene) & np.asarray(w_all.metric, bool)
        if not sel.any():
            continue
        ws = w_all.subset(sel)
        e, _r = method_errors(ws, params_for(scene), cfg, nets_for(scene), device, los=True)
        errs.append(e)
        idx.append(ws.meta["gt_index"])
        keys += [(int(a), str(c), int(g)) for a, c, g in zip(ws.meta["pose_seed"], ws.meta["copy"], ws.meta["gt_index"])]
    err = {k: np.concatenate([e[k] for e in errs]) for k in errs[0]}
    return err, np.concatenate(idx), keys
