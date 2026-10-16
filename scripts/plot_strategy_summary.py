"""One-page summary figure of the IPPE strategies on the whole val (reads the tables written by replay_disambiguation.py).

Usage: python scripts/plot_strategy_summary.py [--replay experiments/replay_val_global] [--out experiments/summary/strategies.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STRATS = ("reproj", "temporal", "temporal_reject", "global")
LABELS = {"reproj": "(a) riproiezione", "temporal": "(b) continuità", "temporal_reject": "(c) continuità + rifiuto",
          "global": "(d) globale"}
COLORS = {"reproj": "#2a78d6", "temporal": "#eb6834", "temporal_reject": "#1baf7a", "global": "#8e5ad1"}
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a64", "#fcfcfb"

ap = argparse.ArgumentParser()
ap.add_argument("--replay", default="experiments/replay_val_global")
ap.add_argument("--out", default="experiments/summary/strategies.png")
args = ap.parse_args()
T = {s: json.loads((Path(args.replay) / f"table_{s}.json").read_text()) for s in STRATS}

fig = plt.figure(figsize=(14.5, 6.4), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.28)

# left: whole val, raw flip vs flip up to the Tello's symmetry
ax = fig.add_subplot(gs[0])
x = range(len(STRATS))
raw = [100 * T[s]["all"]["flip_rate"] for s in STRATS]
sym = [100 * T[s]["all"]["flip_rate_sym"] for s in STRATS]
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], raw, w, color=[COLORS[s] for s in STRATS], label="flip grezzo (> 90°)")
b2 = ax.bar([i + w / 2 for i in x], sym, w, color=[COLORS[s] for s in STRATS], alpha=0.45, hatch="//",
            edgecolor=SURFACE, label="flip a meno della simmetria del drone 08")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
ax.set_xticks(list(x))
ax.set_xticklabels(["(a)\nriproiezione", "(b)\ncontinuità", "(c)\ncont. + rifiuto", "(d)\nglobale"], color=INK)
ax.set_ylabel("frame in flip (%)", color=MUTED)
ax.set_title("Intera val (12600 frame, keypoint YOLO)", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.set_facecolor(SURFACE)
ax.grid(axis="y", color="#e6e5df")
ax.set_axisbelow(True)

# right: per-drone table of raw flip %, colour = value
ax2 = fig.add_subplot(gs[1])
drones = sorted(T["reproj"]["by_drone"])
data = [[100 * T[s]["by_drone"][d]["flip_rate"] for d in drones] for s in STRATS]
im = ax2.imshow(data, cmap="Reds", vmin=0, vmax=45, aspect="auto")
ax2.set_xticks(range(len(drones)))
ax2.set_xticklabels([f"drone {d}" for d in drones], color=INK)
ax2.set_yticks(range(len(STRATS)))
ax2.set_yticklabels([LABELS[s] for s in STRATS], color=INK)
for i, row in enumerate(data):
    for j, v in enumerate(row):
        ax2.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10, color="white" if v > 25 else INK)
ax2.set_title("Flip grezzo (%) per drone: il 08 (Tello) è un problema di identità, non di scelta", loc="left",
              color=INK, fontsize=11)
ax2.tick_params(length=0)
for sp in ax2.spines.values():
    sp.set_visible(False)
fig.suptitle("Da (a) a (d): la decisione globale elimina i flip dei droni 01–06; il drone 08 resta (ambiguità del quadrato)",
             color=INK, fontsize=12.5, x=0.01, ha="left")
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
print("scritto", out)
