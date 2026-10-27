"""Baseline tuning on the train windows (decision 2026-10-16) and the full set of baseline predictions.

Plan step 3.1b adds turn-rate (CTRV/CTRA) and generalised-acro baselines; on generated windows the objective is
weighted by the importance weights (meta "weight") so the parameters fit the natural distribution, and the
mode-aware acro variant is tuned per command mode. Windows without meta (6DroneSyn) are tuned as before.
"""
import itertools

import numpy as np

from dronepose.forecast.config import ForecastConfig
from dronepose.forecast.gaps import (acro_g, ca_g, const_g, ctr_g, cv_g, kalman_g, kalman_los_g, physics_g,
                                     rot_hold_g, rot_omega_g)
from dronepose.forecast.metrics import ade_at, metric_group, position_errors
from dronepose.forecast.windows import Windows

POSITION_METHODS = ("const", "cv15", "cv", "ca", "kf_cv", "kf_ca", "physics", "ctrv", "ctra", "physics_acro",
                    "physics_acro_gen")
MODE_METHOD = "physics_acro_mode"          # only when the windows carry the command mode (generated sets)
KINEMATIC_METHODS = ("cv", "ca", "kf_cv", "kf_ca", "ctrv", "ctra")
ORIENTATION_METHODS = ("rot_hold", "rot_const_omega")


def check_split(split: str, allow_test: bool) -> None:
    if split == "test" and not allow_test:
        raise ValueError("the test split (scenes 06, 07) is used once, at the end: pass --allow-test explicitly")


def _weights(windows: Windows) -> np.ndarray:
    return np.asarray(windows.meta["weight"], float) if "weight" in windows.meta else np.ones(len(windows))


def _objective(windows: Windows, pred: np.ndarray, H: int) -> float:
    m = metric_group(windows)
    e = ade_at(position_errors(pred[m], windows.fut_p[m]), H)
    return float(np.average(e, weights=_weights(windows)[m])) if "weight" in windows.meta else float(e.mean())


def _ctr_k(params: dict, method: str) -> int:
    k = params["ctr_k"]
    return int(k[method]) if isinstance(k, dict) else int(k)


def _acro_variant(params: dict, key: str) -> tuple:
    v = params[key]
    return v["lam"], v["thrust"], v.get("k", params["acro_k"]), v.get("kd", params["acro_kd"])


def tune(windows: Windows, cfg: ForecastConfig, los: bool = False) -> dict:
    """Grid search minimising the mean ADE at the full horizon over the windows in metres; los=True also tunes the
    line-of-sight Kalman CA (E4b: positions must have their origin in the camera)."""
    w, H = windows, cfg.pred_len
    grid, best = [], {}

    def run(method, params, pred):
        score = _objective(w, pred, H)
        grid.append({"method": method, **params, "ade_1s": score})
        if method not in best or score < best[method][1]:
            best[method] = (params, score)

    for k in cfg.cv_k_grid:
        run("cv", {"k": k}, cv_g(w, k))
    for k in cfg.ca_k_grid:
        run("ca", {"k": k}, ca_g(w, k))
    for model, qs in (("kf_cv", cfg.kf_cv_q_grid), ("kf_ca", cfg.kf_ca_q_grid)):
        for q, r in itertools.product(qs, cfg.kf_r_grid):
            run(model, {"q": q, "r": r}, kalman_g(w, model[3:], q, r))
    if los:
        for q, r_los, r_lat in itertools.product(cfg.kf_ca_q_grid, cfg.kf_los_r_los_grid, cfg.kf_los_r_lat_grid):
            run("kf_ca_los", {"q": q, "r_los": r_los, "r_lat": r_lat}, kalman_los_g(w, q, r_los, r_lat))

    for name in ("ctrv", "ctra"):
        for k in cfg.ctr_k_grid:
            run(name, {"k": k}, ctr_g(w, name, k))
    acro = _tune_acro(w, cfg, H, grid)
    best.update({m: (p, sc) for m, (p, sc) in acro["best"].items()})

    warnings = []
    # k = degree + 1 is the smallest legal fit length: the lower end of a k grid starting there is not an edge
    for method, name, values, floor in (("cv", "k", cfg.cv_k_grid, 2), ("ca", "k", cfg.ca_k_grid, 3),
                                        ("kf_cv", "q", cfg.kf_cv_q_grid, None), ("kf_ca", "q", cfg.kf_ca_q_grid, None),
                                        ("kf_cv", "r", cfg.kf_r_grid, None), ("kf_ca", "r", cfg.kf_r_grid, None)):
        chosen = best[method][0][name]
        edges = {values[-1]} | ({values[0]} if values[0] != floor else set())
        if len(values) > 1 and chosen in edges:
            warnings.append(f"{method}: {name} = {chosen} is at the edge of the grid {list(values)}")
    for method, name, values in (("ctrv", "k", cfg.ctr_k_grid), ("ctra", "k", cfg.ctr_k_grid),
                                 ("physics_acro", "k", cfg.acro_k_grid), ("physics_acro_gen", "k", cfg.acro_k_grid),
                                 ("physics_acro_gen", "lam", cfg.acro_lam_grid)):
        chosen = best[method][0][name]
        # k = 3 is the smallest fit length (degree 2 / two chords) and lambda = 0 is pure acro: not edges
        edges = {values[-1]} | ({values[0]} if values[0] not in (3, 0.0) else set())
        if len(values) > 1 and chosen in edges:
            warnings.append(f"{method}: {name} = {chosen} is at the edge of the grid {list(values)}")
    out = {"cv_k": best["cv"][0]["k"], "ca_k": best["ca"][0]["k"],
           "kf_cv": best["kf_cv"][0], "kf_ca": best["kf_ca"][0],
           "ctr_k": {"ctrv": best["ctrv"][0]["k"], "ctra": best["ctra"][0]["k"]},
           "acro_k": best["physics_acro"][0]["k"], "acro_kd": best["physics_acro"][0]["kd"],
           "acro_gen": dict(best["physics_acro_gen"][0]),
           "scores": {m: s for m, (_, s) in best.items()}, "grid": grid, "edge_warnings": warnings}
    if los:
        out["kf_ca_los"] = best["kf_ca_los"][0]
    if acro["mode"]:
        out["acro_mode"] = acro["mode"]
        out["scores"][MODE_METHOD] = acro["mode_score"]
    return out


def _tune_acro(w: Windows, cfg: ForecastConfig, H: int, grid: list) -> dict:
    """Generalised acro on the metric windows: pure acro (lambda 0, constant thrust), one setting for all, and,
    when the windows carry the command mode, the best setting per mode (same predictions, objective per mode)."""
    m = metric_group(w)
    wt = _weights(w)[m]
    wm, fut_p = w.subset(m), w.fut_p[m]
    modes = np.asarray(w.meta["mode"])[m] if "mode" in w.meta else None
    combos = list(itertools.product(cfg.acro_k_grid, cfg.acro_kd, cfg.acro_lam_grid, cfg.acro_thrust))
    if (0.0, "constant") not in {(lam, th) for _, _, lam, th in combos}:
        combos += [(k, kd, 0.0, "constant") for k, kd in itertools.product(cfg.acro_k_grid, cfg.acro_kd)]
    errs = {}
    for k, kd, lam, th in combos:
        e = ade_at(position_errors(acro_g(wm, k, lam, th, kd), fut_p), H)
        errs[(k, kd, lam, th)] = e
        grid.append({"method": "physics_acro_gen", "k": k, "kd": kd, "lam": lam, "thrust": th,
                     "ade_1s": float(np.average(e, weights=wt))})
    as_params = lambda c: {"k": c[0], "kd": c[1], "lam": c[2], "thrust": c[3]}
    score = lambda c, mask=slice(None): float(np.average(errs[c][mask], weights=wt[mask]))
    pure = min((c for c in errs if c[2] == 0.0 and c[3] == "constant"), key=score)
    gen = min(errs, key=score)
    out = {"best": {"physics_acro": (as_params(pure), score(pure)), "physics_acro_gen": (as_params(gen), score(gen))},
           "mode": {}, "mode_score": None}
    if modes is not None:
        e_mode = np.empty(len(modes))
        for mode in sorted(set(modes)):
            sel = modes == mode
            c = min(errs, key=lambda c: score(c, sel))
            out["mode"][str(mode)] = as_params(c)
            e_mode[sel] = errs[c][sel]
        out["mode_score"] = float(np.average(e_mode, weights=wt))
    return out


def predict_all(windows: Windows, params: dict, cfg: ForecastConfig, los: bool = False) -> dict[str, np.ndarray]:
    missing = [k for k in ("ctr_k", "acro_k", "acro_kd", "acro_gen") if k not in params]
    if missing:
        raise ValueError(f"tuning parameters without {missing} (written before step 3.1b): "
                         "re-run scripts/forecast/tune_baselines.py")
    w = windows
    out = {
        "const": const_g(w),
        "cv15": cv_g(w, len(w.obs_t)),
        "cv": cv_g(w, params["cv_k"]),
        "ca": ca_g(w, params["ca_k"]),
        "kf_cv": kalman_g(w, "cv", params["kf_cv"]["q"], params["kf_cv"]["r"]),
        "kf_ca": kalman_g(w, "ca", params["kf_ca"]["q"], params["kf_ca"]["r"]),
        "physics": physics_g(w, params["cv_k"], cfg.g, cfg.max_tilt_deg),
        "rot_hold": rot_hold_g(w),
        "rot_const_omega": rot_omega_g(w),
    }
    out["ctrv"] = ctr_g(w, "ctrv", _ctr_k(params, "ctrv"))
    out["ctra"] = ctr_g(w, "ctra", _ctr_k(params, "ctra"))
    out["physics_acro"] = acro_g(w, params["acro_k"], 0.0, "constant", params["acro_kd"])
    lam, th, k, kd = _acro_variant(params, "acro_gen")
    out["physics_acro_gen"] = acro_g(w, k, lam, th, kd)
    if los:
        v = params["kf_ca_los"]
        out["kf_ca_los"] = kalman_los_g(w, v["q"], v["r_los"], v["r_lat"])
    if "mode" in w.meta and "acro_mode" in params:
        pred = out["physics_acro_gen"].copy()          # a mode not seen in tuning keeps the mode-blind setting
        modes = np.asarray(w.meta["mode"])
        for mode, v in params["acro_mode"].items():
            sel = modes == mode
            if sel.any():
                pred[sel] = acro_g(w.subset(sel), v.get("k", params["acro_k"]), v["lam"], v["thrust"],
                                   v.get("kd", params["acro_kd"]))
        out[MODE_METHOD] = pred
    return out


def best_kinematic(params: dict) -> str:
    """The kinematic baseline with the lowest train ADE at 1 s (the reference of criteria C1, C2)."""
    return min(KINEMATIC_METHODS, key=lambda m: params["scores"][m])
