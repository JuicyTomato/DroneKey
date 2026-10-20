"""Training curves of YOLO pose runs: val mAP50-95(P) and val pose loss, one line per run.

Usage: python scripts/plot_training_curves.py --run LABEL=RUN_DIR [--run ...] [--mark-epoch LABEL=N] --out OUT.png
Up to 4 runs (fixed colour order); RUN_DIR holds results.csv. The curve-end label is LABEL up to the first ":"
(so "C: frame 640 px" is tagged "C"). --mark-epoch LABEL=N draws a dotted line at epoch N (best.pt of that short label).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order
PANELS = [("metrics/mAP50-95(P)", "mAP50-95 keypoint (val)"), ("val/pose_loss", "Loss di posa (val)")]


def load(run_dir: Path) -> list[dict]:
    """Rows of results.csv as floats, header names stripped."""
    with open(run_dir / "results.csv") as f:
        return [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _label_dir(spec: str) -> tuple[str, str]:
    label, sep, value = spec.partition("=")
    if not sep or not label or not value:
        raise ValueError(f"bad argument {spec!r}: expected LABEL=VALUE")
    return label, value


def plot(runs: list[tuple[str, Path]], marks: dict[str, int], out: Path) -> None:
    """Draw the two panels for `runs` and save the figure to `out`."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb")
    for ax, (key, title) in zip(axes, PANELS):
        ax.set_facecolor("#fcfcfb")
        for (label, run_dir), color in zip(runs, COLORS):
            rows = load(run_dir)
            ep, y = [r["epoch"] for r in rows], [r[key] for r in rows]
            ax.plot(ep, y, color=color, lw=2, label=label)
            ax.annotate(label.split(":")[0], (ep[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("epoca", color=INK2)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    for label, ep in marks.items():
        axes[0].axvline(ep, color=INK2, lw=1, ls=":")
        axes[0].annotate(f"best.pt {label} (ep. ~{ep})", (ep, 0.36), xytext=(-4, 0), textcoords="offset points",
                         ha="right", color=INK2, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=len(runs), labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    print(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR")
    ap.add_argument("--mark-epoch", action="append", default=[], metavar="LABEL=N")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    runs = [(label, Path(d)) for label, d in map(_label_dir, args.run)]
    if len(runs) > len(COLORS):
        ap.error(f"at most {len(COLORS)} runs")
    marks = {label: int(n) for label, n in map(_label_dir, args.mark_epoch)}
    unknown = set(marks) - {label.split(":")[0] for label, _ in runs}
    if unknown:
        ap.error(f"--mark-epoch labels not in --run: {sorted(unknown)}")
    plot(runs, marks, Path(args.out))


if __name__ == "__main__":
    main()
