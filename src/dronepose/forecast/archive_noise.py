"""Real pose errors from the 2.6 archive as observation noise (spec 2026-10-27 §5.1, §5.7).

The archive (experiments/pose_errors/archive_val_s*.jsonl, one per keypoint seed, val scenes 03 / 09 / 14) holds per
frame dp_g = G (t_est - t_gt) s_c (metres, gravity frame), dR = R_gt^T R_est (body frame), validity, distance and
line of sight. It is cut into tracts of consecutive frames (the same rule as error_archive.windows) that replace the
Gaussian noise on observed windows: positions shifted, attitudes rotated, frames without a pose lost.

Rows are read directly from the JSONL (error_archive.SequenceErrors carries neither los_g nor the predicted class,
and error_archive.py is not to be changed). The archive converts to metres with the TRUE class; the full pipeline
only knows the predicted one, so by default dp_g is rescaled to the predicted class's s_c (a true 01-06 predicted
as 08, without s_c, becomes a lost frame; a true 08 is left in dataset units).
"""
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.eval.units import METERS_PER_UNIT
from dronepose.forecast.windows import Windows


@dataclass
class Tracts:
    dp: np.ndarray          # (M,L,3) position error, gravity frame (0 in lost frames)
    dR: np.ndarray          # (M,L,3,3) body-frame rotation error (identity in lost frames)
    valid: np.ndarray       # (M,L) bool
    distance: np.ndarray    # (M,L) camera-drone distance of the ground truth (metres; dataset units for 08)
    los: np.ndarray         # (M,L,3) unit line of sight camera -> drone, gravity frame
    scene: np.ndarray       # (M,) str
    drone: np.ndarray       # (M,) str, true class
    seed: np.ndarray        # (M,) int, keypoint-model seed
    meters: np.ndarray      # (M,) bool
    n_rescaled: int = 0     # valid frames whose dp was rescaled to the predicted class
    n_lost_pred08: int = 0  # frames lost because a metric drone was predicted as 08

    def __len__(self) -> int:
        return len(self.scene)

    def take(self, idx) -> "Tracts":
        cut = lambda a: np.asarray(a)[idx]
        return replace(self, dp=cut(self.dp), dR=cut(self.dR), valid=cut(self.valid), distance=cut(self.distance),
                       los=cut(self.los), scene=cut(self.scene), drone=cut(self.drone), seed=cut(self.seed),
                       meters=cut(self.meters))

    def select(self, scenes: tuple[str, ...], meters: bool) -> "Tracts":
        return self.take(np.isin(self.scene, list(scenes)) & (self.meters == meters))


def archive_noise_rng(seed: int, *salt: int) -> np.random.Generator:
    return np.random.default_rng([seed, 2026, *salt])


def _frame(r: dict, rescale: bool) -> tuple:
    """(valid, dp, dR, rescaled, lost_pred08) of one row."""
    s_true, s_pred = METERS_PER_UNIT.get(r["drone"]), METERS_PER_UNIT.get(r["drone_model"])
    valid = bool(r["valid"])
    if not valid:
        return False, np.zeros(3), np.eye(3), False, False
    dp, dR = np.asarray(r["dp_g"], float), np.asarray(r["dR"], float)
    if not rescale or s_true is None or r["drone_model"] == r["drone"]:
        return True, dp, dR, False, False
    if s_pred is None:
        return False, np.zeros(3), np.eye(3), False, True
    gt = np.asarray(r["los_g"], float) * float(r["distance_m"])        # G t_gt s_true
    return True, (dp + gt) * (s_pred / s_true) - gt, dR, True, False


def load_tracts(paths: list[Path], length: int = 15, rescale: bool = True) -> Tracts:
    seqs: dict[tuple, list[dict]] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") == "header":
                    continue
                seqs.setdefault((int(r["seed"]), r["sequence"]), []).append(r)
    cols = {k: [] for k in ("dp", "dR", "valid", "distance", "los", "scene", "drone", "seed", "meters")}
    n_rescaled = n_lost = 0
    for (seed, _), rows in sorted(seqs.items()):
        rows.sort(key=lambda r: r["frame_idx"])
        idx = np.array([r["frame_idx"] for r in rows])
        frames = [_frame(r, rescale) for r in rows]
        n_rescaled += sum(f[3] for f in frames)
        n_lost += sum(f[4] for f in frames)
        valid = np.array([f[0] for f in frames])
        dp, dR = np.stack([f[1] for f in frames]), np.stack([f[2] for f in frames])
        dist = np.array([r["distance_m"] for r in rows], float)
        los = np.array([r["los_g"] for r in rows], float)
        for i in range(len(rows) - length + 1):
            if idx[i + length - 1] - idx[i] != length - 1:
                continue
            sl = slice(i, i + length)
            for k, v in (("dp", dp[sl]), ("dR", dR[sl]), ("valid", valid[sl]), ("distance", dist[sl]),
                         ("los", los[sl])):
                cols[k].append(v)
            cols["scene"].append(rows[0]["scene"])
            cols["drone"].append(rows[0]["drone"])
            cols["seed"].append(seed)
            cols["meters"].append(bool(rows[0]["meters"]))
    if not cols["scene"]:
        L = length
        return Tracts(np.zeros((0, L, 3)), np.zeros((0, L, 3, 3)), np.zeros((0, L), bool), np.zeros((0, L)),
                      np.zeros((0, L, 3)), np.zeros(0, str), np.zeros(0, str), np.zeros(0, int), np.zeros(0, bool),
                      n_rescaled, n_lost)
    stack = {k: np.stack(cols[k]) for k in ("dp", "dR", "valid", "distance", "los")}
    return Tracts(**stack, scene=np.array(cols["scene"]), drone=np.array(cols["drone"]), seed=np.array(cols["seed"]),
                  meters=np.array(cols["meters"]), n_rescaled=int(n_rescaled), n_lost_pred08=int(n_lost))


def sample_tracts(tracts: Tracts, W: int, rng: np.random.Generator, distance: np.ndarray | None = None,
                  threshold_m: float = 3.0) -> tuple[np.ndarray, int]:
    """(W,) tract indices, uniform; with `distance` (W,) from the tracts of the same band of last-frame distance
    (<= threshold / beyond), falling back to all tracts when the band is empty. Returns the fallback count."""
    M = len(tracts)
    if M == 0:
        raise ValueError("no tracts to sample from")
    if distance is None:
        return rng.integers(M, size=W), 0
    far_tract = tracts.distance[:, -1] > threshold_m
    far_win = np.asarray(distance, float) > threshold_m
    idx, n_fb = np.empty(W, int), 0
    for band in (False, True):
        sel = np.flatnonzero(far_win == band)
        pool = np.flatnonzero(far_tract == band)
        if len(pool) == 0:
            pool, n_fb = np.arange(M), n_fb + len(sel)
        idx[sel] = pool[rng.integers(len(pool), size=len(sel))]
    return idx, n_fb


def _azimuth(v: np.ndarray) -> np.ndarray:
    return np.arctan2(v[..., 1], v[..., 0])


def yaw_angles(tracts: Tracts, idx: np.ndarray, rng: np.random.Generator,
               target_los: np.ndarray | None = None) -> np.ndarray:
    """Rotation about the vertical per window: uniform (generated windows, no camera) or the one that brings the
    azimuth of the tract's last line of sight onto the window's (6DroneSyn: elevation stays the tract's)."""
    if target_los is None:
        return rng.uniform(0, 2 * np.pi, len(idx))
    return _azimuth(np.asarray(target_los, float)) - _azimuth(tracts.los[idx, -1])


def apply_tracts(windows: Windows, tracts: Tracts, idx: np.ndarray, yaw: np.ndarray) -> Windows:
    """Tract idx[i] on the observed frames of window i; the future stays ground truth (same frame as before).
    Lost frames are only masked (their pose is not used by any predictor); an existing mask is kept."""
    w = windows
    idx = np.asarray(idx, int)
    if len(idx) != len(w) or len(yaw) != len(w):
        raise ValueError(f"{len(idx)} tracts / {len(yaw)} angles for {len(w)} windows")
    if tracts.dp.shape[1] != w.obs_p.shape[1]:
        raise ValueError(f"tracts of {tracts.dp.shape[1]} frames, windows observe {w.obs_p.shape[1]}")
    Rz = Rotation.from_euler("z", np.asarray(yaw, float)[:, None]).as_matrix()                     # (W,3,3)
    dp = np.einsum("wij,wnj->wni", Rz, tracts.dp[idx])
    obs_R = np.asarray(w.obs_R, float) @ tracts.dR[idx]
    valid = tracts.valid[idx] if w.obs_valid is None else tracts.valid[idx] & np.asarray(w.obs_valid, bool)
    return replace(w, obs_p=w.obs_p + dp, obs_R=obs_R, obs_u=obs_R[..., :, 2], obs_valid=valid)
