"""Realism check of the generator (spec §5.5): descriptors of the generated train vs the 6DroneSyn train.

Histograms (natural-weighted for the generated set: the stratified set is resampled by its importance weights) of
speed, acceleration, jerk, tilt, angular speed, nose-motion angle, hover and turn durations; quantiles; coverage of
2D descriptor spaces. 6DroneSyn is not a realism reference (keyframed): it only gives the calm end of the scale.

Usage: python scripts/forecast/generator_realism.py [--gen data/processed/forecast_gen] [--real data/processed/forecast]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.realism import coverage_2d, descriptors

ap = argparse.ArgumentParser()
ap.add_argument("--gen", default="data/processed/forecast_gen")
ap.add_argument("--real", default="data/processed/forecast")
ap.add_argument("--out", default="experiments/forecast/generator")
ap.add_argument("--resample", type=int, default=4000, help="natural resample size of the generated train")
args = ap.parse_args()

gen = load_trajectories(Path(args.gen), "train")
w = np.array([t.meta["weight"] for t in gen])
rng = np.random.default_rng(0)
natural = [gen[i] for i in rng.choice(len(gen), size=args.resample, p=w / w.sum())]
sets = {"generato (train, naturale)": descriptors(natural), "generato (train, stratificato)": descriptors(gen),
        "6DroneSyn (train)": descriptors(load_trajectories(Path(args.real), "train"))}
keys = ("speed", "accel", "jerk", "tilt_deg", "omega_deg_s", "heading_motion_deg", "hover_s", "curve_s")
qs = (5, 50, 95, 99, 100)
summary = {name: {k: dict(zip([f"p{q}" for q in qs], [float(x) for x in np.percentile(d[k], qs)])) if len(d[k]) else None
                  for k in keys} for name, d in sets.items()}
pairs = (("speed", "accel", 20), ("tilt_deg", "omega_deg_s", 20))
cov = {}
for a, b, n in pairs:
    allx = np.concatenate([d[a] for d in sets.values()])
    ally = np.concatenate([d[b] for d in sets.values()])
    edges = (np.linspace(0, np.percentile(allx, 99.5), n + 1), np.linspace(0, np.percentile(ally, 99.5), n + 1))
    cov[f"{a} x {b}"] = {name: coverage_2d(d[a], d[b], edges) for name, d in sets.items()}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "realism.json").write_text(json.dumps({"quantiles": summary, "coverage_2d": cov}, indent=1))
for name, s in summary.items():
    print(name, {k: (round(v["p50"], 2), round(v["p95"], 2)) if v else None for k, v in s.items()})
print("copertura 2D:", cov)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, k in zip(axes.ravel(), keys):
        hi = max(np.percentile(d[k], 99) for d in sets.values() if len(d[k]))
        bins = np.linspace(0, hi, 40)
        for name, d in sets.items():
            if len(d[k]):
                ax.hist(np.clip(d[k], 0, hi), bins=bins, density=True, histtype="step", label=name)
        ax.set_title(k)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Generatore contro 6DroneSyn (train): densità per descrittore")
    fig.tight_layout()
    fig.savefig(out / "realism.png", dpi=130)
    plt.close(fig)
except ImportError:
    print("matplotlib assente: figura saltata")
print(f"scritto {out / 'realism.json'}")
