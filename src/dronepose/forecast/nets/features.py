"""Windows -> network inputs (plan step 3.3, spec §5.2): features, base predictions, normalisation, yaw rotation, pools.

NumPy only. Every position (observed, bases, target) is relative to the last OBSERVED position, which is the origin
of a clean window and not of a noisy one (add_pose_noise keeps the future in the clean frame): re-anchoring there is
what a real system can do, and errors are translation-invariant, so the ADE is the error in the window frame.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.forecast.acro import acro_predict
from dronepose.forecast.baselines import cv
from dronepose.forecast.config import ForecastConfig
from dronepose.forecast.kalman import kalman_predict
from dronepose.forecast.noise import add_pose_noise_per_window, noise_rng, sample_mixture_sigmas
from dronepose.forecast.tuning import _acro_variant
from dronepose.forecast.windows import Windows

BASES = ("cv", "phys", "phys_mode", "kf_ca")
MODES = ("acro", "angle", "pmode")
CLASSES = ("01", "02", "03", "04", "05", "06", "08")
_HORIZ = np.array([0, 1, 3, 4, 6, 7])          # x, y of position, velocity and up axis in x (W,N,9)


@dataclass
class Batch:
    x: np.ndarray               # (W,N,9) position rel. to the last observation, velocity, up axis u
    bases: dict                 # name -> (W,H,3), same origin as x
    target: np.ndarray          # (W,H,3) future positions, same origin
    weight: np.ndarray          # (W,) importance weight (1 without meta)
    mode_onehot: np.ndarray     # (W,3) MODES, zero when unknown (6DroneSyn)
    class_onehot: np.ndarray    # (W,7) CLASSES, zero when unknown
    traj: np.ndarray            # (W,)
    metric: np.ndarray          # (W,) bool

    def __len__(self) -> int:
        return len(self.x)

    def subset(self, idx) -> "Batch":
        return Batch(x=self.x[idx], bases={k: v[idx] for k, v in self.bases.items()}, target=self.target[idx],
                     weight=self.weight[idx], mode_onehot=self.mode_onehot[idx], class_onehot=self.class_onehot[idx],
                     traj=self.traj[idx], metric=self.metric[idx])


def base_predictions(windows: Windows, params: dict, cfg: ForecastConfig) -> dict[str, np.ndarray]:
    """The four bases of BASES, computed exactly as tuning.predict_all computes cv, physics_acro_gen,
    physics_acro_mode and kf_ca (without the other nine methods). Without the mode: phys_mode = phys."""
    w = windows
    args = (w.obs_t, w.obs_p, w.fut_t)
    obs_R = np.asarray(w.obs_R, float)
    lam, th, k, kd = _acro_variant(params, "acro_gen")
    out = {"cv": cv(*args, params["cv_k"]),
           "phys": acro_predict(w.obs_t, w.obs_p, obs_R, w.fut_t, k, lam, th, kd),
           "kf_ca": kalman_predict(*args, "ca", params["kf_ca"]["q"], params["kf_ca"]["r"])}
    pred = out["phys"].copy()
    if "mode" in w.meta and "acro_mode" in params:
        modes = np.asarray(w.meta["mode"])
        for mode, v in params["acro_mode"].items():
            sel = modes == mode
            if sel.any():
                pred[sel] = acro_predict(w.obs_t, w.obs_p[sel], obs_R[sel], w.fut_t, v.get("k", params["acro_k"]),
                                         v["lam"], v["thrust"], v.get("kd", params["acro_kd"]))
    out["phys_mode"] = pred
    return out


def _onehot(values, names) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    out = np.zeros((len(values), len(names)), np.float32)
    for r, v in enumerate(values):
        if v in idx:
            out[r, idx[v]] = 1
    return out


def build_batch(windows: Windows, params: dict, cfg: ForecastConfig) -> Batch:
    w = windows
    W = len(w)
    anchor = np.asarray(w.obs_p, float)[:, -1:, :]                         # (W,1,3)
    pos = w.obs_p - anchor
    vel = np.diff(w.obs_p, axis=1) * cfg.fps
    vel = np.concatenate([vel[:, :1], vel], axis=1)                       # first frame: the first difference
    x = np.concatenate([pos, vel, np.asarray(w.obs_u, float)], axis=-1)
    bases = {k: (v - anchor).astype(np.float32) for k, v in base_predictions(w, params, cfg).items()}
    meta = w.meta
    return Batch(x=x.astype(np.float32), bases=bases, target=(w.fut_p - anchor).astype(np.float32),
                 weight=np.asarray(meta["weight"], np.float32) if "weight" in meta else np.ones(W, np.float32),
                 mode_onehot=_onehot(meta["mode"], MODES) if "mode" in meta else np.zeros((W, 3), np.float32),
                 class_onehot=(_onehot(meta["drone_class"], CLASSES) if "drone_class" in meta
                               else np.zeros((W, len(CLASSES)), np.float32)),
                 traj=np.asarray(w.traj), metric=np.asarray(w.metric, bool))


@dataclass
class Normaliser:
    """x: per-channel mean / std, with the horizontal channels of each 3-vector at zero mean and one shared std
    (so a yaw rotation commutes with the normalisation); positions (bases, target): one scale for all."""
    x_mean: np.ndarray
    x_std: np.ndarray
    pos_scale: float

    @classmethod
    def fit(cls, batch: Batch) -> "Normaliser":
        x = batch.x.reshape(-1, 9).astype(np.float64)
        mean, std = x.mean(0), x.std(0)
        for c in (0, 3, 6):
            mean[c:c + 2] = 0
            std[c:c + 2] = np.sqrt(np.mean(x[:, c:c + 2] ** 2))
        return cls(x_mean=mean, x_std=np.maximum(std, 1e-6),
                   pos_scale=float(max(np.sqrt(np.mean(batch.target.astype(np.float64) ** 2)), 1e-6)))

    def apply(self, batch: Batch) -> Batch:
        s = self.pos_scale
        return replace(batch, x=((batch.x - self.x_mean) / self.x_std).astype(np.float32),
                       bases={k: (v / s).astype(np.float32) for k, v in batch.bases.items()},
                       target=(batch.target / s).astype(np.float32))

    def to_json(self) -> dict:
        return {"x_mean": [float(v) for v in self.x_mean], "x_std": [float(v) for v in self.x_std],
                "pos_scale": self.pos_scale}

    @classmethod
    def from_json(cls, d: dict) -> "Normaliser":
        return cls(x_mean=np.array(d["x_mean"]), x_std=np.array(d["x_std"]), pos_scale=float(d["pos_scale"]))


def rotate_yaw(batch: Batch, angles) -> Batch:
    """Rotate about e3 (one angle per window) positions, velocities, up axis, bases and target."""
    Rz = Rotation.from_euler("z", np.asarray(angles, float)[:, None]).as_matrix().astype(np.float32)   # (W,3,3)
    rot = lambda p: np.einsum("wij,w...j->w...i", Rz, p).astype(np.float32)
    x = np.concatenate([rot(batch.x[..., c:c + 3]) for c in (0, 3, 6)], axis=-1)
    return replace(batch, x=x, bases={k: rot(v) for k, v in batch.bases.items()}, target=rot(batch.target))


def build_pool(windows: Windows, params_noisy: dict, cfg: ForecastConfig, K: int, seed: int,
               max_p_cm: float = 5.0, max_r_deg: float = 3.0) -> list[Batch]:
    """K noisy copies (noise mixture per window, Gaussian: not the real PnP error), bases on the noisy poses."""
    return [noisy_copy(windows, params_noisy, cfg, seed, k, max_p_cm, max_r_deg) for k in range(K)]


def noisy_copy(windows: Windows, params_noisy: dict, cfg: ForecastConfig, seed: int, k: int, max_p_cm: float = 5.0,
               max_r_deg: float = 3.0) -> Batch:
    """Copy k of build_pool: its own stream noise_rng(seed, 0, salt=k) (sigmas first, then the noise)."""
    rng = noise_rng(seed, 0, salt=k)
    sp, sr = sample_mixture_sigmas(len(windows), rng, max_p_cm, max_r_deg)
    return build_batch(add_pose_noise_per_window(windows, sp, sr, rng), params_noisy, cfg)


def save_pool(pool: list[Batch], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b0 = pool[0]
    np.savez(path, x=np.stack([b.x for b in pool]), target=np.stack([b.target for b in pool]),
             **{f"base_{k}": np.stack([b.bases[k] for b in pool]) for k in BASES},
             weight=b0.weight, mode_onehot=b0.mode_onehot, class_onehot=b0.class_onehot, traj=b0.traj, metric=b0.metric)


def load_pool(path: Path, bases=BASES) -> list[Batch]:
    """Read a pool; only the bases listed are loaded (the noisy train pool is ~1 GB with all four)."""
    with np.load(Path(path)) as z:
        d = {k: z[k] for k in z.files if not k.startswith("base_") or k[5:] in bases}
    return [Batch(x=d["x"][k], bases={b: d[f"base_{b}"][k] for b in bases}, target=d["target"][k], weight=d["weight"],
                  mode_onehot=d["mode_onehot"], class_onehot=d["class_onehot"], traj=d["traj"], metric=d["metric"])
            for k in range(len(d["x"]))]
