"""R-CV-6D forecast of one window at a time (streaming): the last 15 poses in the gravity frame -> next second.

Same features as E4b (`build_batch(..., mask_channel=True)` with the fold's parameters and cv_k = 2); the network
and the parameters are those of the fold of the scene (spec 2026-11-14 §4).
"""
import json
import sys
from pathlib import Path

import numpy as np

from dronepose.forecast.config import load_config
from dronepose.forecast.nets.device import pick_device
import torch

from dronepose.forecast.nets.features import Normaliser, build_batch, fill_lost
from dronepose.forecast.nets.train import build_model, load_checkpoint
from dronepose.forecast.windows import Windows
from dronepose.eval.units import METERS_PER_UNIT

NET = "R-CV-6D"
_E4 = Path(__file__).resolve().parents[3] / "scripts" / "forecast" / "e4"


def _run_name(scene: str, seed: int) -> str:
    if str(_E4) not in sys.path:
        sys.path.insert(0, str(_E4))
    from e4_train import e4_run_name
    return e4_run_name(NET, f"fold{scene}", seed)


class Forecaster:
    def __init__(self, scene: str, cfg_path: Path = Path("configs/pipeline.json"),
                 forecast_cfg: Path = Path("configs/forecast.json"), device: str = "auto"):
        cfg = json.loads(Path(cfg_path).read_text())
        self.cfg = load_config(Path(forecast_cfg))
        self.params = json.loads((Path(cfg["forecast_params"]) / f"params_fold{scene}.json").read_text())
        self.run = Path(cfg["forecast_runs"]) / _run_name(scene, int(cfg["seed"]))
        self.device = pick_device(device)
        self.scene = scene
        ck = load_checkpoint(self.run / "best.pt")          # loaded once (predict_run reloads it at every call)
        self.model = build_model(ck["arm"], ck["cfg"], ck["rnn"], ck["use_mode"], ck.get("n_in", 9)).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.norm = Normaliser.from_json(ck["normaliser"])
        self.use_mode = ck["use_mode"]

    @torch.no_grad()
    def _net(self, batch) -> np.ndarray:
        nb = self.norm.apply(batch)
        to = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.device)
        mc = np.concatenate([nb.mode_onehot, nb.class_onehot], 1)
        out = self.model(to(nb.x), {k: to(v) for k, v in nb.bases.items()}, to(mc) if self.use_mode else None)
        return self.norm.pos_scale * out["pred"].cpu().numpy().astype(np.float64)

    def predict(self, obs_t_g: np.ndarray, obs_R_g: np.ndarray, valid: np.ndarray, drone: str) -> np.ndarray | None:
        """(H, 3) absolute positions (metres, gravity frame) for the next second; None with fewer than 2 valid
        frames or a drone without a metric size."""
        valid = np.asarray(valid, bool)
        if valid.sum() < 2 or METERS_PER_UNIT.get(drone) is None:
            return None
        N, H, fps = self.cfg.obs_len, self.cfg.pred_len, self.cfg.fps
        R = np.asarray(obs_R_g, float)[None]
        w = Windows(obs_t=(np.arange(N) - (N - 1)) / fps, fut_t=np.arange(1, H + 1) / fps,
                    obs_p=np.asarray(obs_t_g, float)[None], fut_p=np.zeros((1, H, 3)), obs_R=R,
                    fut_R=np.tile(np.eye(3), (1, H, 1, 1)), obs_u=R[..., :, 2], fut_u=np.zeros((1, H, 3)),
                    last_p=np.zeros((1, 3)), traj=np.zeros(1, int), start=np.zeros(1, int), metric=np.ones(1, bool),
                    scene=np.array([self.scene]), drone=np.array([drone]), motion=np.array(["linear"]),
                    traj_keys=["live"], meta={}, obs_valid=valid[None])
        batch = build_batch(w, {**self.params, "cv_k": 2}, self.cfg, mask_channel=True)
        anchor = w.obs_p[0, fill_lost(w.obs_valid)[0, -1]]
        return self._net(batch)[0] + anchor
