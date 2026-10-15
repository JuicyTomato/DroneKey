"""PnP sensitivity on the val split (GT keypoints + noise, wrong scale). CSV + 4 figures in experiments/sensitivity/.

Usage: python scripts/sensitivity_pnp.py [--seeds 0 1 2]
"""
import argparse
import csv
from pathlib import Path

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.pose.disambiguation import STRATEGIES
from dronepose.pose.replay import check_no_test
from dronepose.pose.sensitivity import SIGMAS, run_noise, run_scale

# Validated categorical palette (dataviz reference, light): fixed order, one colour per strategy.
COLORS = {"reproj": "#2a78d6", "temporal": "#eb6834", "temporal_reject": "#1baf7a"}
MARKERS = {"reproj": "o", "temporal": "s", "temporal_reject": "^"}
LABELS = {"reproj": "(a) riproiezione", "temporal": "(b) continuità", "temporal_reject": "(c) continuità + rifiuto"}
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a64", "#fcfcfb"

ap = argparse.ArgumentParser()
ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
ap.add_argument("--models", default="configs/drone_models.json")
ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
ap.add_argument("--allow-test", action="store_true")
ap.add_argument("--out", default="experiments/sensitivity")
args = ap.parse_args()

records = read_jsonl(Path(args.canonical))
check_no_test(records, args.allow_test)
models = load_drone_models(Path(args.models))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

noise_rows = run_noise(records, models, SIGMAS, tuple(args.seeds), STRATEGIES)
scale_rows = run_scale(records, models)
for name, rows in (("noise", noise_rows), ("scale", scale_rows)):
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print(f"CSV: {out}/noise.csv, {out}/scale.csv")

import matplotlib  # noqa: E402  (plots only)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def mean_over_seeds(rows, strategy, sigma, key, bin_="all"):
    vals = [r[key] for r in rows if r["strategy"] == strategy and r["sigma"] == sigma and r["bin"] == bin_
            and r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def styled(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color="#e6e5df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c0")
    ax.tick_params(colors=MUTED)


def line_per_strategy(key, scale, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    for st in STRATEGIES:
        ys = [mean_over_seeds(noise_rows, st, s, key) for s in SIGMAS]
        ys = [None if y is None else y * scale for y in ys]
        ax.plot(SIGMAS, ys, color=COLORS[st], marker=MARKERS[st], markersize=8, linewidth=2, label=LABELS[st])
        dy = {"reproj": 7, "temporal": -7, "temporal_reject": 0}[st]     # keep direct labels apart at sigma = 10
        ax.annotate(LABELS[st], (SIGMAS[-1], ys[-1]), xytext=(6, dy), textcoords="offset points",
                    color=INK, fontsize=8, va="center")
    styled(ax, title, "rumore sui keypoint σ (px)", ylabel)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=200)
    plt.close(fig)


line_per_strategy("flip_rate", 100, "Flip (>90°) in funzione del rumore", "frame in flip (%)", "flip_vs_sigma.png")
line_per_strategy("median_rot_err_deg", 1, "Errore di rotazione mediano", "errore (°)", "rot_vs_sigma.png")

# depth error vs distance, strategy (b), one line per sigma (sequential blue ramp, light -> dark)
ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#0d366b"]
bins = ["<2", "2-3", "3-4", ">=4"]
fig, ax = plt.subplots(figsize=(7.4, 4.0), facecolor=SURFACE)
for color, s in zip(ramp, SIGMAS):
    ys = [mean_over_seeds(noise_rows, "temporal", s, "median_rel_depth_err", b) for b in bins]
    ax.plot(bins, [None if y is None else 100 * y for y in ys], color=color, marker="o", markersize=8, linewidth=2,
            label=f"σ = {s:g} px")
styled(ax, "Errore relativo di profondità, strategia (b)", "distanza del drone (unità camera)", "errore (%)")
ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig(out / "depth_vs_distance.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
xs = [r["scale"] for r in scale_rows]
ax.plot(xs, [r["median_depth_ratio"] for r in scale_rows], color=COLORS["reproj"], marker="o", markersize=8,
        linewidth=2)
for x, r in zip(xs, scale_rows):
    ax.annotate(f"{r['median_depth_ratio']:.2f}", (x, r["median_depth_ratio"]), xytext=(0, 8),
                textcoords="offset points", ha="center", color=INK, fontsize=8)
styled(ax, "Modello 3D in scala sbagliata: la profondità scala uguale", "scala del modello usato nel PnP",
       "profondità stimata / vera")
fig.tight_layout()
fig.savefig(out / "scale.png", dpi=200)
plt.close(fig)
print("figure:", ", ".join(p.name for p in sorted(out.glob("*.png"))))
