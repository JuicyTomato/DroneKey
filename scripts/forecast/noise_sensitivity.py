"""Sensitivity of the baselines to Gaussian noise on the observed poses (spec §5.5), on the generated val.

Position noise sigma in {0, 1, 2, 3, 5} cm paired with rotation noise in {0, 0.5, 1, 2, 3} deg (random rotation
vector), 3 seeds; parameters tuned on the clean generated train. Weighted (natural) ADE / FDE at 1 s per method.
Not a substitute for the realistic pose errors of E4b.

Usage: python scripts/forecast/noise_sensitivity.py [--split val] [--params experiments/forecast/generator/tuning.json]
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.metrics import ade_at, fde_at, position_errors
from dronepose.forecast.tuning import check_split, predict_all
from dronepose.forecast.windows import freeze_windows

SIGMA_P_CM = (0, 1, 2, 3, 5)
SIGMA_R_DEG = (0, 0.5, 1, 2, 3)

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--data", default="data/processed/forecast_gen")
ap.add_argument("--config", default="configs/forecast.json")
ap.add_argument("--params", default="experiments/forecast/generator/tuning.json")
ap.add_argument("--out", default="experiments/forecast/generator")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--allow-test", action="store_true")
args = ap.parse_args()

check_split(args.split, args.allow_test)
cfg = load_config(Path(args.config))
params = json.loads(Path(args.params).read_text())
clean = freeze_windows(load_trajectories(Path(args.data), args.split), cfg.obs_len, cfg.pred_len, cfg.fps)
wts = np.asarray(clean.meta["weight"], float)
H = len(clean.fut_t)
t0 = time.time()
res = {}
for sp, sr in zip(SIGMA_P_CM, SIGMA_R_DEG):
    acc = {}
    for seed in range(args.seeds if sp or sr else 1):
        rng = np.random.default_rng([seed, int(10 * sr)])
        obs_p = clean.obs_p + rng.normal(scale=sp / 100, size=clean.obs_p.shape)
        # same frame as the clean future: the noisy last observation is simply not at the origin
        W, N = clean.obs_R.shape[:2]
        dR = Rotation.from_rotvec(rng.normal(scale=np.radians(sr), size=(W * N, 3))).as_matrix().reshape(W, N, 3, 3)
        obs_R = np.asarray(clean.obs_R, float) @ dR
        w = dataclasses.replace(clean, obs_p=obs_p, obs_R=obs_R, obs_u=obs_R[..., :, 2])
        for k, pred in predict_all(w, params, cfg).items():
            if pred.shape[-1] != 3 or pred.ndim != 3:
                continue                                       # orientation baselines
            e = position_errors(pred, w.fut_p)
            acc.setdefault(k, []).append((np.average(ade_at(e, H), weights=wts), np.average(fde_at(e, H), weights=wts)))
    res[f"{sp}cm_{sr}deg"] = {k: {"ade_1s": float(np.mean([a for a, _ in v])), "fde_1s": float(np.mean([f for _, f in v]))}
                              for k, v in acc.items()}
    print(f"rumore {sp} cm / {sr}°: " + ", ".join(f"{k} {v['ade_1s']:.3f}" for k, v in res[f'{sp}cm_{sr}deg'].items()), flush=True)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "noise.json").write_text(json.dumps({"sigma_p_cm": SIGMA_P_CM, "sigma_r_deg": SIGMA_R_DEG, "seeds": args.seeds,
                                            "split": args.split, "results": res}, indent=1))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    x = np.arange(len(SIGMA_P_CM))
    for k in next(iter(res.values())):
        for ax, m in zip(axes, ("ade_1s", "fde_1s")):
            ax.plot(x, [res[f"{sp}cm_{sr}deg"][k][m] for sp, sr in zip(SIGMA_P_CM, SIGMA_R_DEG)], marker="o", label=k)
    for ax, m in zip(axes, ("ADE 1 s (m, pesata)", "FDE 1 s (m, pesata)")):
        ax.set_xticks(x, [f"{sp} cm\n{sr}°" for sp, sr in zip(SIGMA_P_CM, SIGMA_R_DEG)])
        ax.set_ylabel(m)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Sensibilità al rumore gaussiano sulle pose osservate ({args.split} generata)")
    fig.tight_layout()
    fig.savefig(out / "noise.png", dpi=150)
    plt.close(fig)
except ImportError:
    print("matplotlib assente: figura saltata")
print(f"scritto {out / 'noise.json'} in {time.time() - t0:.0f} s")
