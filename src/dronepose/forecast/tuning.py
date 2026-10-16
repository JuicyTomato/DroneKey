"""Baseline tuning on the train windows (decision 2026-10-16) and the full set of baseline predictions."""
import itertools

import numpy as np

from dronepose.forecast.baselines import ca, constant_position, cv
from dronepose.forecast.config import ForecastConfig
from dronepose.forecast.kalman import kalman_predict
from dronepose.forecast.metrics import ade_at, metric_group, position_errors
from dronepose.forecast.orientation import constant_angular_velocity, hold_orientation
from dronepose.forecast.physics import physics_predict
from dronepose.forecast.windows import Windows

POSITION_METHODS = ("const", "cv15", "cv", "ca", "kf_cv", "kf_ca", "physics")
ORIENTATION_METHODS = ("rot_hold", "rot_const_omega")


def check_split(split: str, allow_test: bool) -> None:
    if split == "test" and not allow_test:
        raise ValueError("the test split (scenes 06, 07) is used once, at the end: pass --allow-test explicitly")


def _objective(windows: Windows, pred: np.ndarray, H: int) -> float:
    m = metric_group(windows)
    return float(ade_at(position_errors(pred[m], windows.fut_p[m]), H).mean())


def tune(windows: Windows, cfg: ForecastConfig) -> dict:
    """Grid search minimising the mean ADE at the full horizon over the windows in metres."""
    w, H = windows, cfg.pred_len
    grid, best = [], {}

    def run(method, params, pred):
        score = _objective(w, pred, H)
        grid.append({"method": method, **params, "ade_1s": score})
        if method not in best or score < best[method][1]:
            best[method] = (params, score)

    for k in cfg.cv_k_grid:
        run("cv", {"k": k}, cv(w.obs_t, w.obs_p, w.fut_t, k))
    for k in cfg.ca_k_grid:
        run("ca", {"k": k}, ca(w.obs_t, w.obs_p, w.fut_t, k))
    for model, qs in (("kf_cv", cfg.kf_cv_q_grid), ("kf_ca", cfg.kf_ca_q_grid)):
        for q, r in itertools.product(qs, cfg.kf_r_grid):
            run(model, {"q": q, "r": r}, kalman_predict(w.obs_t, w.obs_p, w.fut_t, model[3:], q, r))

    warnings = []
    # k = degree + 1 is the smallest legal fit length: the lower end of a k grid starting there is not an edge
    for method, name, values, floor in (("cv", "k", cfg.cv_k_grid, 2), ("ca", "k", cfg.ca_k_grid, 3),
                                        ("kf_cv", "q", cfg.kf_cv_q_grid, None), ("kf_ca", "q", cfg.kf_ca_q_grid, None),
                                        ("kf_cv", "r", cfg.kf_r_grid, None), ("kf_ca", "r", cfg.kf_r_grid, None)):
        chosen = best[method][0][name]
        edges = {values[-1]} | ({values[0]} if values[0] != floor else set())
        if len(values) > 1 and chosen in edges:
            warnings.append(f"{method}: {name} = {chosen} is at the edge of the grid {list(values)}")
    return {"cv_k": best["cv"][0]["k"], "ca_k": best["ca"][0]["k"],
            "kf_cv": best["kf_cv"][0], "kf_ca": best["kf_ca"][0],
            "scores": {m: s for m, (_, s) in best.items()}, "grid": grid, "edge_warnings": warnings}


def predict_all(windows: Windows, params: dict, cfg: ForecastConfig) -> dict[str, np.ndarray]:
    w = windows
    args = (w.obs_t, w.obs_p, w.fut_t)
    return {
        "const": constant_position(*args),
        "cv15": cv(*args, len(w.obs_t)),
        "cv": cv(*args, params["cv_k"]),
        "ca": ca(*args, params["ca_k"]),
        "kf_cv": kalman_predict(*args, "cv", params["kf_cv"]["q"], params["kf_cv"]["r"]),
        "kf_ca": kalman_predict(*args, "ca", params["kf_ca"]["q"], params["kf_ca"]["r"]),
        "physics": physics_predict(w.obs_t, w.obs_p, w.obs_u, w.fut_t, params["cv_k"], cfg.g, cfg.max_tilt_deg),
        "rot_hold": hold_orientation(w.obs_R, len(w.fut_t)),
        "rot_const_omega": constant_angular_velocity(w.obs_t, w.obs_R, w.fut_t),
    }
