"""E4a / E4b evaluation (spec 2026-10-27 §5.6): errors of every candidate on the same windows, tables, selection.

Candidates (positions): cv (tuned k), ca, kf_cv, kf_ca, kf_ca_los (E4b), physics (tilt), the 6DroneSyn nets (mean of
their seeds); reference CV-2 = CV through the last 2 VALID frames, never re-tuned. Orientation: rot_hold and
rot_const_omega, reported without selection. Unit of the statistics: the trajectory (all its windows, copies and
pose seeds), paired bootstrap stratified by scene.
"""
from pathlib import Path

import numpy as np

from dronepose.forecast.config import ForecastConfig
from dronepose.forecast.gaps import cv_g
from dronepose.forecast.metrics import (ade_at, bootstrap_counts, fde_at, geodesic_deg_batch, position_errors,
                                        skill_ci, up_angle_deg)
from dronepose.forecast.nets.evaluate import predict_run
from dronepose.forecast.nets.features import build_batch
from dronepose.forecast.selection import select_method
from dronepose.forecast.tuning import predict_all
from dronepose.forecast.windows import Windows

BASELINES = ("cv", "ca", "kf_cv", "kf_ca", "physics")
REFERENCE = "CV-2"


def method_errors(windows: Windows, params: dict, cfg: ForecastConfig, nets: dict | None = None, device=None,
                  los: bool = False) -> tuple[dict, dict]:
    """Position errors (W,H) per candidate and the reference, orientation errors (W,H) per orientation baseline.
    nets: name -> list of run directories (one per seed); their bases use params with cv_k = 2."""
    w = windows
    pred = predict_all(w, params, cfg, los=los)
    names = BASELINES + (("kf_ca_los",) if los else ())
    err = {k: position_errors(pred[k], w.fut_p) for k in names}
    err[REFERENCE] = position_errors(cv_g(w, 2), w.fut_p)
    if nets:
        batch = build_batch(w, {**params, "cv_k": 2}, cfg, mask_channel=True)
        for name, runs in nets.items():
            err[name] = np.mean([position_errors(predict_run(Path(r), batch, device), batch.target) for r in runs], 0)
    rot = {k: {"geodesic": geodesic_deg_batch(pred[k], w.fut_R), "up": up_angle_deg(pred[k][..., :, 2], w.fut_u)}
           for k in ("rot_hold", "rot_const_omega")}
    return err, rot


def _groups(w: Windows) -> list[tuple[str, np.ndarray, bool]]:
    m = np.asarray(w.metric, bool)
    out = [("metri: tutti", m, True), ("metri: lineare", m & (w.motion == "linear"), True),
           ("metri: non lineare", m & (w.motion == "nonlinear"), True)]
    out += [(f"metri: scena {s}", m & (w.scene == s), True) for s in sorted(set(w.scene[m]))]
    out += [(f"drone {d}", m & (w.drone == d), False) for d in sorted(set(w.drone[m]))]
    out += [(f"drone {d} (unità dataset)", ~m & (w.drone == d), False) for d in sorted(set(w.drone[~m]))]
    return [g for g in out if g[1].any()]


def summarize_e4(err: dict, rot: dict, windows: Windows, cfg: ForecastConfig, b: int, seed: int = 0) -> dict:
    w, hs = windows, cfg.horizons_frames
    H = len(w.fut_t)
    groups = []
    for name, mask, with_ci in _groups(w):
        inv, counts = bootstrap_counts(w.traj[mask], w.scene[mask], b, seed)
        g = {"group": name, "n_windows": int(mask.sum()), "n_traj": int(len(set(w.traj[mask]))), "position": {},
             "skill_vs_CV-2": {}, "orientation": {}}
        for k, e in err.items():
            em = e[mask]
            g["position"][k] = {"ade": [float(ade_at(em, h).mean()) for h in hs],
                                "fde": [float(fde_at(em, h).mean()) for h in hs]}
            if k != REFERENCE:
                for metric, fn in (("ade_1s", ade_at), ("fde_1s", fde_at)):
                    v = skill_ci(fn(em, H), fn(err[REFERENCE][mask], H), inv, counts)
                    g["skill_vs_CV-2"].setdefault(k, {})[metric] = list(v) if with_ci else [v[0]]
        for k, r in rot.items():
            g["orientation"][k] = {"geodesic_deg_1s": float(r["geodesic"][mask][:, -1].mean()),
                                   "up_deg_1s": float(r["up"][mask][:, -1].mean())}
        groups.append(g)
    return {"n_windows": len(w), "n_traj": len(set(w.traj)), "horizons_s": [h / cfg.fps for h in hs],
            "reference": REFERENCE, "groups": groups}


def select_on(err: dict, windows: Windows, candidates, b: int, seed: int = 0) -> dict:
    m = np.asarray(windows.metric, bool)
    H = len(windows.fut_t)
    return select_method({k: ade_at(err[k][m], H) for k in candidates}, windows.traj[m], windows.scene[m], b=b,
                         seed=seed)


def degradation(err_b: dict, err_a: dict, windows: Windows, b: int, seed: int = 0) -> dict:
    """E4b - E4a ADE at 1 s per method on the same windows (metric), mean and 95% interval (paired per trajectory)."""
    m = np.asarray(windows.metric, bool)
    H = len(windows.fut_t)
    inv, counts = bootstrap_counts(windows.traj[m], windows.scene[m], b, seed)
    n_win = counts @ np.bincount(inv, minlength=counts.shape[1])
    out = {}
    for k in err_b:
        if k not in err_a:
            continue
        d = ade_at(err_b[k][m], H) - ade_at(err_a[k][m], H)
        boot = (counts @ np.bincount(inv, weights=d, minlength=counts.shape[1])) / n_win
        out[k] = {"mean": float(d.mean()), "ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                  "ade_1s_e4a": float(ade_at(err_a[k][m], H).mean()), "ade_1s_e4b": float(ade_at(err_b[k][m], H).mean())}
    return out


def concat_errors(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def pipeline_errors(trajs, pose_paths: list[Path], cfg: ForecastConfig, params_for, nets_for, device=None,
                    allow_test: bool = False) -> tuple[dict, dict, np.ndarray, dict]:
    """E4b errors over every pose seed, copy and scene: params_for(scene, metric) -> parameters, nets_for(scene) ->
    nets (metric windows only; NaN for drone 08). Returns position errors, orientation errors, the GT window index of
    every row (windows = make_windows(trajs).subset(index)) and counts (rescaled, lost, dropped windows)."""
    from dronepose.forecast.e4 import load_pose_windows
    from dronepose.forecast.gaps import drop_sparse
    err_parts, rot_parts, idx_parts = [], [], []
    info = {"n_rescaled": 0, "n_lost_pred08": 0, "n_invalid": 0, "n_windows": 0, "n_dropped": 0}
    for path in pose_paths:
        w_all, inf = load_pose_windows(Path(path), trajs, cfg, allow_test=allow_test)
        for k in ("n_rescaled", "n_lost_pred08", "n_invalid"):
            info[k] += inf[k]
        w_all, keep = drop_sparse(w_all)
        info["n_windows"] += int(keep.sum())
        info["n_dropped"] += int((~keep).sum())
        for scene in sorted(set(w_all.scene)):
            for metric in (True, False):
                sel = (w_all.scene == scene) & (np.asarray(w_all.metric, bool) == metric)
                if not sel.any():
                    continue
                ws = w_all.subset(sel)
                nets = nets_for(scene) if metric else None
                e, r = method_errors(ws, params_for(scene, metric), cfg, nets, device, los=True)
                if not metric:
                    e.update({n: np.full(e["cv"].shape, np.nan) for n in nets_for(scene)})
                err_parts.append(e)
                rot_parts.append(r)
                idx_parts.append(ws.meta["gt_index"])
    rot = {k: {m: np.concatenate([p[k][m] for p in rot_parts]) for m in ("geodesic", "up")} for k in rot_parts[0]}
    return concat_errors(err_parts), rot, np.concatenate(idx_parts), info


def guard_test_run(allow_test: bool, out_dir: Path) -> None:
    """The test (scenes 06, 07) is used once, at the end, with the choices fixed on the val (spec 2026-10-27 §5.6)."""
    if not allow_test:
        raise ValueError("the test split (scenes 06, 07) is used once, at the end: pass --allow-test explicitly")
    if Path(out_dir).exists():
        raise ValueError(f"{out_dir} exists: il test si usa una volta sola (nessuna seconda esecuzione)")
