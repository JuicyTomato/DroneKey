"""E4a / E4b on the 6DroneSyn (spec 2026-10-27 §5.4-§5.7): estimated poses as observation windows.

E4b observes the poses of the full pipeline (experiments/poses/poses_{split}_s{seed}_{causal,gt}.jsonl: R_g, t_g in
the gravity frame, metres with the TRUE class's s_c) and predicts the ground-truth future. Every (scene, drone)
trajectory has three background copies (`copies`), each a separate noisy view: one window per GT window and copy,
ordered trajectory by trajectory, copy by copy, in the GT window order. By default t_g is rescaled to the PREDICTED
class (t_g * s_pred / s_true: the pipeline only knows the predicted class); a metric drone predicted as 08 (no s_c)
is a lost frame, a true 08 stays in dataset units. Positions stay relative to the GT last observation (the future
is unchanged, as with the archive noise: errors are translation-invariant).
"""
import dataclasses
import json
from pathlib import Path

import numpy as np

from dronepose.eval.units import METERS_PER_UNIT
from dronepose.forecast.config import ForecastConfig
from dronepose.forecast.dataset import Trajectory
from dronepose.forecast.windows import Windows, make_windows

NETS_SELECT_SCENES = ("02", "05")


def _read_sequences(path: Path, allow_test: bool, rescale: bool) -> tuple[dict, dict]:
    seqs, info = {}, {"n_rescaled": 0, "n_lost_pred08": 0, "n_invalid": 0, "seeds": set()}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test" and not allow_test:
                raise ValueError(f"{path}: test poses (scenes 06, 07) are used once, at the end: pass allow_test=True")
            seqs.setdefault(r["sequence"], []).append(r)
            info["seeds"].add(int(r["seed"]))
    out = {}
    for seq, rows in seqs.items():
        rows.sort(key=lambda r: r["frame_idx"])
        T = len(rows)
        t, R, valid = np.zeros((T, 3)), np.tile(np.eye(3), (T, 1, 1)), np.zeros(T, bool)
        for i, r in enumerate(rows):
            if not r["valid"]:
                info["n_invalid"] += 1
                continue
            s_true, s_pred = METERS_PER_UNIT.get(r["drone"]), METERS_PER_UNIT.get(r["drone_model"])
            k = 1.0
            if rescale and s_true is not None and r["drone_model"] != r["drone"]:
                if s_pred is None:
                    info["n_lost_pred08"] += 1
                    continue
                k = s_pred / s_true
                info["n_rescaled"] += 1
            t[i], R[i], valid[i] = np.asarray(r["t_g"], float) * k, np.asarray(r["R_g"], float), True
        out[seq] = (np.array([r["frame_idx"] for r in rows]), t, R, valid, int(rows[0]["seed"]))
    info["seeds"] = sorted(info["seeds"])
    return out, info


def load_pose_windows(pose_path: Path, trajs: list[Trajectory], cfg: ForecastConfig, rescale: bool = True,
                      allow_test: bool = False) -> tuple[Windows, dict]:
    """Windows observing the estimated poses of every copy; meta: copy ('bg_ddd'), pose_seed, gt_index (index into
    make_windows(trajs)). Raises if a copy has no pose sequence or a sequence is not frame-aligned."""
    seqs, info = _read_sequences(Path(pose_path), allow_test, rescale)
    gt = make_windows(trajs, cfg.obs_len, cfg.pred_len, cfg.fps)
    N = cfg.obs_len
    rows, copies, seeds, obs_p, obs_R, valid = [], [], [], [], [], []
    for j, tr in enumerate(trajs):
        win = np.flatnonzero(gt.traj == j)
        for bg, deg in tr.copies:
            seq = f"{tr.scene}_{tr.drone}_{bg}_{int(deg):03d}"
            if seq not in seqs:
                raise ValueError(f"{pose_path}: no poses for {seq}")
            fidx, t, R, v, seed = seqs[seq]
            if not np.array_equal(fidx, np.arange(1, len(tr.times) + 1)):
                raise ValueError(f"{seq}: pose frames are not 1..{len(tr.times)}")
            frames = gt.start[win][:, None] + np.arange(N)[None]
            rows.append(win)
            copies += [f"{bg}_{int(deg):03d}"] * len(win)
            seeds += [seed] * len(win)
            obs_p.append(t[frames] - gt.last_p[win][:, None])
            obs_R.append(R[frames])
            valid.append(v[frames])
    idx = np.concatenate(rows)
    w = gt.subset(idx)
    obs_R = np.concatenate(obs_R)
    w = dataclasses.replace(w, obs_p=np.concatenate(obs_p), obs_R=obs_R, obs_u=obs_R[..., :, 2],
                            obs_valid=np.concatenate(valid),
                            meta={**w.meta, "copy": np.array(copies), "pose_seed": np.array(seeds), "gt_index": idx})
    return w, info


def split_nets_scenes(windows: Windows, select_scenes: tuple[str, ...] = NETS_SELECT_SCENES) -> tuple[Windows, Windows]:
    """Metric windows for training the 6DroneSyn nets and for their early stopping (train scenes 02, 05)."""
    m = np.asarray(windows.metric, bool)
    sel = np.isin(windows.scene, list(select_scenes))
    return windows.subset(m & ~sel), windows.subset(m & sel)
