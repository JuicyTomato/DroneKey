"""Gaussian noise on the observed poses (spec §5.5) and baselines re-tuned on equally noisy train windows.

The clean-parameter curve (noise.json) asks how parameters tuned on the clean ground truth degrade; the re-tuned curve
asks how much of the physical advantage survives when every baseline is tuned for the noise it will see (a first
answer to E4b, which uses the realistic pose errors instead). The val stream reproduces noise.json exactly; the train
stream uses the same key plus a salt, so train and val noise are independent.
"""
import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.forecast.config import ForecastConfig
from dronepose.forecast.metrics import ade_at, fde_at, position_errors
from dronepose.forecast.tuning import predict_all, tune
from dronepose.forecast.windows import Windows

SIGMA_P_CM = (0, 1, 2, 3, 5)
SIGMA_R_DEG = (0, 0.5, 1, 2, 3)
# wider than configs/forecast.json: noise favours long fits and large measurement variances (k <= obs_len = 15;
# r up to 10: at 5 cm kf_ca still gains ~6% from r = 0.1 to r = 1 on the train)
RETUNE_GRIDS = {"cv_k_grid": (2, 3, 5, 10, 15), "ca_k_grid": (3, 5, 10, 15),
                "kf_r_grid": (1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0),
                "ctr_k_grid": (3, 5, 10, 15), "acro_k_grid": (3, 5, 10, 15)}
_INT_GRIDS = {"cv_k_grid", "ca_k_grid", "ctr_k_grid", "acro_k_grid"}
_FLOAT_GRIDS = {"kf_cv_q_grid", "kf_ca_q_grid", "kf_r_grid", "acro_lam_grid"}
KEPT_PARAMS = ("cv_k", "ca_k", "kf_cv", "kf_ca", "ctr_k", "acro_k", "acro_kd", "acro_gen", "acro_mode")
SKILL = ("physics_acro_gen", "kf_ca")


def level_key(sigma_p_cm, sigma_r_deg) -> str:
    return f"{sigma_p_cm}cm_{sigma_r_deg}deg"


def noise_rng(seed: int, sigma_r_deg: float, salt: int | None = None) -> np.random.Generator:
    key = [seed, int(10 * sigma_r_deg)] + ([] if salt is None else [salt])
    return np.random.default_rng(key)


def add_pose_noise(windows: Windows, sigma_p_cm: float, sigma_r_deg: float, rng: np.random.Generator) -> Windows:
    """Isotropic position noise (cm) and a random rotation vector (deg per axis) on every observed pose.

    The future stays in the clean frame: the noisy last observation is simply not at the origin."""
    w = windows
    obs_p = w.obs_p + rng.normal(scale=sigma_p_cm / 100, size=w.obs_p.shape)
    W, N = w.obs_R.shape[:2]
    dR = Rotation.from_rotvec(rng.normal(scale=np.radians(sigma_r_deg), size=(W * N, 3))).as_matrix()
    obs_R = np.asarray(w.obs_R, float) @ dR.reshape(W, N, 3, 3)
    return dataclasses.replace(w, obs_p=obs_p, obs_R=obs_R, obs_u=obs_R[..., :, 2])


def parse_grid(spec: str) -> tuple[str, tuple]:
    """'kf_r_grid=1e-6,1e-1' -> ('kf_r_grid', (1e-06, 0.1)); only numeric grids of ForecastConfig."""
    name, _, values = spec.partition("=")
    if name not in _INT_GRIDS | _FLOAT_GRIDS or not values:
        raise ValueError(f"unknown or empty grid {spec!r}: use one of {sorted(_INT_GRIDS | _FLOAT_GRIDS)}")
    cast = int if name in _INT_GRIDS else float
    return name, tuple(cast(v) for v in values.split(","))


def with_grids(cfg: ForecastConfig, grids: dict) -> ForecastConfig:
    return dataclasses.replace(cfg, **{k: tuple((int if k in _INT_GRIDS else float)(v) for v in vs)
                                       for k, vs in grids.items()})


def evaluate_positions(windows: Windows, params: dict, cfg: ForecastConfig) -> dict:
    """Weighted (importance weights, else uniform) ADE / FDE at the full horizon per position method."""
    w, H = windows, len(windows.fut_t)
    wts = np.asarray(w.meta["weight"], float) if "weight" in w.meta else np.ones(len(w))
    out = {}
    for k, pred in predict_all(w, params, cfg).items():
        if pred.ndim != 3 or pred.shape[-1] != 3:
            continue                                       # orientation baselines
        e = position_errors(pred, w.fut_p)
        out[k] = {"ade_1s": float(np.average(ade_at(e, H), weights=wts)),
                  "fde_1s": float(np.average(fde_at(e, H), weights=wts))}
    return out


def chosen_params(params: dict) -> dict:
    return {k: params[k] for k in KEPT_PARAMS if k in params}


def retune_level(train: Windows, val: Windows, sigma_p_cm: float, sigma_r_deg: float, seed: int,
                 cfg: ForecastConfig) -> dict:
    """Tune every baseline on the noisy train, evaluate on the val with independent noise of the same level."""
    params = tune(add_pose_noise(train, sigma_p_cm, sigma_r_deg, noise_rng(seed, sigma_r_deg, salt=1)), cfg)
    val_noisy = add_pose_noise(val, sigma_p_cm, sigma_r_deg, noise_rng(seed, sigma_r_deg))
    return {"sigma_p_cm": sigma_p_cm, "sigma_r_deg": sigma_r_deg, "seed": seed, "params": chosen_params(params),
            "train_scores": params["scores"], "edge_warnings": params["edge_warnings"],
            "val": evaluate_positions(val_noisy, params, cfg)}


def summarize_retuned(runs: list[dict]) -> dict:
    """Per level: ADE / FDE means over the seeds, the chosen parameters per seed and the acro_gen vs kf_ca skill."""
    out = {}
    for key in dict.fromkeys(level_key(r["sigma_p_cm"], r["sigma_r_deg"]) for r in runs):
        rs = sorted((r for r in runs if level_key(r["sigma_p_cm"], r["sigma_r_deg"]) == key), key=lambda r: r["seed"])
        lvl = {k: {m: float(np.mean([r["val"][k][m] for r in rs])) for m in ("ade_1s", "fde_1s")} for k in rs[0]["val"]}
        a, b = SKILL
        lvl["skill_acro_gen_vs_kf_ca"] = {
            **{m: 1 - lvl[a][m] / lvl[b][m] for m in ("ade_1s", "fde_1s")},
            "per_seed_ade_1s": [1 - r["val"][a]["ade_1s"] / r["val"][b]["ade_1s"] for r in rs]}
        lvl["params"] = [r["params"] for r in rs]
        lvl["edge_warnings"] = [r.get("edge_warnings", []) for r in rs]
        out[key] = lvl
    return out
