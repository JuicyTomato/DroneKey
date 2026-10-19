"""Phase 3 settings (configs/forecast.json) and per-class metric scale factors."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dronepose.data.drone_models import DroneModel


@dataclass(frozen=True)
class ForecastConfig:
    fps: int
    obs_len: int
    pred_len: int
    horizons_frames: tuple[int, ...]
    dji_diagonal_mm: dict[str, float | None]
    motion_type: dict[str, str]          # folder scene -> "linear" | "nonlinear"
    cv_k_grid: tuple[int, ...]
    ca_k_grid: tuple[int, ...]
    kf_cv_q_grid: tuple[float, ...]
    kf_ca_q_grid: tuple[float, ...]
    kf_r_grid: tuple[float, ...]
    g: float
    max_tilt_deg: float
    bootstrap_b: int
    bootstrap_seed: int
    # plan step 3.1b (generated sets): turn-rate and generalised-acro grids
    ctr_k_grid: tuple[int, ...] = (3, 5, 10)
    acro_k_grid: tuple[int, ...] = (3, 5, 10)
    acro_lam_grid: tuple[float, ...] = (0.0, 2.0, 5.0, 8.0, 12.0)
    acro_thrust: tuple[str, ...] = ("constant", "hold_vz")
    acro_kd: tuple = ("estimate", 0.0, 0.7)


def load_config(path: Path = Path("configs/forecast.json")) -> ForecastConfig:
    d = json.loads(Path(path).read_text())
    return ForecastConfig(
        fps=d["fps"], obs_len=d["obs_len"], pred_len=d["pred_len"],
        horizons_frames=tuple(int(round(h * d["fps"])) for h in d["horizons_s"]),
        dji_diagonal_mm=dict(d["dji_diagonal_mm"]), motion_type=dict(d["motion_type"]),
        cv_k_grid=tuple(d["cv_k_grid"]), ca_k_grid=tuple(d["ca_k_grid"]),
        kf_cv_q_grid=tuple(float(q) for q in d["kf_cv_q_grid"]),
        kf_ca_q_grid=tuple(float(q) for q in d["kf_ca_q_grid"]),
        kf_r_grid=tuple(float(r) for r in d["kf_r_grid"]),
        g=float(d["g"]), max_tilt_deg=float(d["max_tilt_deg"]),
        bootstrap_b=int(d["bootstrap_b"]), bootstrap_seed=int(d["bootstrap_seed"]),
        **{k: tuple(float(x) if k == "acro_lam_grid" else x for x in d[k])
           for k in ("ctr_k_grid", "acro_k_grid", "acro_lam_grid", "acro_thrust", "acro_kd") if k in d},
    )


def model_diagonal(model: DroneModel) -> float:
    """Mean distance between opposite hubs (circular order: key1-key3, key2-key4), dataset units."""
    k = model.kpts_flu
    return float((np.linalg.norm(k[0] - k[2]) + np.linalg.norm(k[1] - k[3])) / 2)


def metric_scales(models: dict[str, DroneModel], dji_diagonal_mm: dict[str, float | None]) -> dict[str, float | None]:
    """s_c = real motor-to-motor diagonal / model diagonal; None when no verified real size exists (Tello)."""
    out = {}
    for drone, model in models.items():
        mm = dji_diagonal_mm.get(drone)
        out[drone] = None if mm is None else mm / 1000 / model_diagonal(model)
    return out


def load_generator_config(path: Path = Path("configs/generator.json")) -> dict:
    """Trajectory generator settings (plan step 3.1b): dynamics, controller, pilot, maneuvers, classes, generator."""
    return json.loads(Path(path).read_text())
