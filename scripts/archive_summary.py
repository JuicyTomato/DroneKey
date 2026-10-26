"""Summary of pose-error archives: statistics (JSON) and two figures showing that the error is not i.i.d.

Usage: python scripts/archive_summary.py ARCHIVE.jsonl [ARCHIVE.jsonl ...] --out-dir DIR
Writes DIR/archive_summary.json, DIR/error_vs_distance.png, DIR/autocorrelation.png. Archives (one per seed) are
pooled. Statistics per drone and overall: fraction of valid frames, flips (> 90 deg, over valid frames), median and p90
of |dp_g| (metres, drones with s_c only) and of the rotation angle of dR (degrees), median ratio of the error along the
line of sight to the lateral error (unit-free). Figures use the drones 01-06 only (metres; drone 08 is symmetric and in
dataset units) and class `causal` as stored in the archive.
  error_vs_distance.png: median position error, along-line-of-sight and lateral components, per quintile of distance.
  autocorrelation.png: autocorrelation of |dp_g| and of the rotation angle over lags 0-30, computed on runs of
      consecutive valid frames (no gap in frame_idx). Each run is centred on its own mean, lagged products and variance
      are pooled over runs (runs shorter than lag+1 add nothing at that lag), lag 0 = 1: only within-run correlation
      counts, level differences between runs do not.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FIGURE_DRONES = ("01", "02", "03", "04", "05", "06")
N_BINS = 5
MAX_LAG = 30


def read_archive_rows(path: Path) -> tuple[dict, list[dict]]:
    head, rows = None, []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "header":
                head = row
            else:
                rows.append(row)
    if head is None:
        raise ValueError(f"{path}: no header line")
    return head, rows


def frame_table(rows: list[dict]) -> dict[str, np.ndarray]:
    """Per-frame arrays over all `rows`: bookkeeping, and for valid frames |dp_g|, its components along and across the
    line of sight, and the angle of dR (NaN where not valid)."""
    n = len(rows)
    t = {"drone": np.array([r["drone"] for r in rows], dtype=object),
         "sequence": np.array([f"{r.get('seed')}|{r['sequence']}" for r in rows], dtype=object),
         "frame_idx": np.array([r["frame_idx"] for r in rows], dtype=int),
         "valid": np.array([bool(r["valid"]) for r in rows]),
         "flip": np.array([bool(r["flip"]) for r in rows]),
         "meters": np.array([bool(r["meters"]) for r in rows]),
         "distance_m": np.array([r["distance_m"] for r in rows], float)}
    for k in ("dp_norm", "los", "lateral", "rot_deg"):
        t[k] = np.full(n, np.nan)
    for i, r in enumerate(rows):
        if not r["valid"]:
            continue
        dp, los = np.asarray(r["dp_g"], float), np.asarray(r["los_g"], float)
        along = float(dp @ los)
        t["dp_norm"][i] = np.linalg.norm(dp)
        t["los"][i] = abs(along)
        t["lateral"][i] = np.linalg.norm(dp - along * los)
        t["rot_deg"][i] = float(np.degrees(Rotation.from_matrix(np.asarray(r["dR"], float)).magnitude()))
    return t


def _q(x: np.ndarray, p: float) -> float | None:
    return float(np.percentile(x, p)) if x.size else None


def block(t: dict[str, np.ndarray], mask: np.ndarray, with_position: bool) -> dict:
    """Statistics of the frames selected by `mask`. Position norms only when all frames are in metres."""
    n, v = int(mask.sum()), mask & t["valid"]
    out = {"n_frames": n, "n_valid": int(v.sum()), "valid_fraction": float(v.sum() / n) if n else None,
           "n_flip": int((v & t["flip"]).sum()), "flip_rate": float((v & t["flip"]).sum() / v.sum()) if v.any() else None}
    pos, rot = t["dp_norm"][v], t["rot_deg"][v]
    out.update(dp_norm_median=_q(pos, 50) if with_position else None, dp_norm_p90=_q(pos, 90) if with_position else None,
               rot_deg_median=_q(rot, 50), rot_deg_p90=_q(rot, 90))
    lat = t["lateral"][v]
    ok = lat > 1e-12
    out["los_over_lateral_median"] = float(np.median(t["los"][v][ok] / lat[ok])) if ok.any() else None
    return out


def summary_stats(t: dict[str, np.ndarray]) -> dict:
    out = {"by_drone": {}}
    for d in sorted(set(t["drone"])):
        m = t["drone"] == d
        out["by_drone"][d] = block(t, m, with_position=bool(t["meters"][m].all()))
    m_all = np.ones(len(t["drone"]), bool)
    out["overall_all_drones"] = block(t, m_all, with_position=False)           # mixes metres and dataset units
    out["overall_all_drones"]["note"] = "dp_norm omitted: drone 08 is in dataset units"
    out["overall_metric_drones"] = block(t, t["meters"], with_position=True)
    return out


def distance_bins(t: dict[str, np.ndarray], n_bins: int = N_BINS) -> list[dict]:
    """Quantile bins of distance_m over the valid frames of the figure drones; median errors per bin."""
    sel = t["valid"] & np.isin(t["drone"], FIGURE_DRONES) & np.isfinite(t["distance_m"])
    if not sel.any():
        return []
    dist = t["distance_m"][sel]
    edges = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                     "distance_median": float(np.median(dist[m])),
                     **{k: float(np.median(t[k][sel][m])) for k in ("dp_norm", "los", "lateral")}})
    return bins


def autocorrelation(t: dict[str, np.ndarray], key: str, max_lag: int = MAX_LAG,
                    drones: tuple[str, ...] = FIGURE_DRONES) -> dict:
    """Autocorrelation of t[key] at lags 0..max_lag over runs of consecutive valid frames of `drones`; each run is
    centred on its own mean, products and variance are pooled over runs."""
    sel = np.isin(t["drone"], drones)
    runs = []
    for seq in sorted(set(t["sequence"][sel])):
        m = np.where(sel & (t["sequence"] == seq))[0]
        m = m[np.argsort(t["frame_idx"][m])]
        cur = []
        for i in m:
            if t["valid"][i] and cur and t["frame_idx"][i] == t["frame_idx"][cur[-1]] + 1:
                cur.append(i)
            else:
                if len(cur) > 1:
                    runs.append(t[key][cur])
                cur = [i] if t["valid"][i] else []
        if len(cur) > 1:
            runs.append(t[key][cur])
    if not runs:
        return {"lags": list(range(max_lag + 1)), "acf": [None] * (max_lag + 1), "pairs": [0] * (max_lag + 1)}
    runs = [r - r.mean() for r in runs]            # each run centred on its own mean: within-run correlation only
    allv = np.concatenate(runs)
    var = float((allv ** 2).mean())
    acf, pairs = [], []
    for k in range(max_lag + 1):
        prods = [r[: len(r) - k] * r[k:] for r in runs if len(r) > k]
        n = sum(p.size for p in prods)
        pairs.append(int(n))
        acf.append(float(np.concatenate(prods).mean() / var) if n and var > 1e-12 else None)
    return {"lags": list(range(max_lag + 1)), "acf": acf, "pairs": pairs, "n_runs": len(runs),
            "n_values": int(allv.size)}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def plot_distance(bins: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    _style(ax)
    x = [b["distance_median"] for b in bins]
    ax.plot(x, [b["dp_norm"] for b in bins], color=INK, lw=1.2, ls=(0, (4, 3)), zorder=3, label="errore di posizione")
    ax.plot(x, [b["los"] for b in bins], color=BLUE, lw=2, marker="o", ms=4, label="lungo la linea di vista")
    ax.plot(x, [b["lateral"] for b in bins], color=ORANGE, lw=2, marker="o", ms=4, label="laterale")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b['lo']:.1f}-{b['hi']:.1f}" for b in bins], fontsize=8)
    ax.set_xlabel("distanza dalla camera, quintili (m)", color=INK2)
    ax.set_ylabel("errore mediano (m)", color=INK2)
    ax.set_ylim(bottom=0)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=3, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_autocorrelation(acfs: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    _style(ax)
    for (label, a), color in zip(acfs.items(), (BLUE, ORANGE)):
        y = [np.nan if v is None else v for v in a["acf"]]
        ax.plot(a["lags"], y, color=color, lw=2, marker="o", ms=3, label=label)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("ritardo (frame)", color=INK2)
    ax.set_ylabel("autocorrelazione", color=INK2)
    ax.set_xlim(0, MAX_LAG)
    ax.set_ylim(top=1.02)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=2, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def summarize(archives: list[str], out_dir: str) -> dict:
    out_dir = Path(out_dir)
    heads, rows = [], []
    for p in archives:
        head, r = read_archive_rows(Path(p))
        heads.append({"path": str(p), "combo": head.get("combo"), "seed": head.get("seed"),
                      "commit": head.get("commit"), "n_frames": len(r)})
        rows += r
    t = frame_table(rows)
    result = {"archives": heads, **summary_stats(t)}
    result["by_seed"] = {str(h["seed"]): summary_stats(frame_table([r for r in rows if r.get("seed") == h["seed"]]))
                         ["overall_metric_drones"] for h in heads}
    result["distance_bins"] = distance_bins(t)
    acfs = {"|dp_g|": autocorrelation(t, "dp_norm"), "angolo di dR": autocorrelation(t, "rot_deg")}
    result["autocorrelation"] = {"drones": list(FIGURE_DRONES), **acfs}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "archive_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    if result["distance_bins"]:
        plot_distance(result["distance_bins"], out_dir / "error_vs_distance.png")
    if all("n_values" in a for a in acfs.values()):
        plot_autocorrelation({"|dp_g| (posizione)": acfs["|dp_g|"], "angolo di dR (rotazione)": acfs["angolo di dR"]},
                             out_dir / "autocorrelation.png")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archives", nargs="+", metavar="ARCHIVE.jsonl")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    res = summarize(args.archives, args.out_dir)
    o = res["overall_metric_drones"]
    print(f"valid {o['valid_fraction']:.4f}  flips {o['n_flip']}  |dp| median {o['dp_norm_median']:.4f} m  "
          f"rot median {o['rot_deg_median']:.3f} deg  los/lateral {o['los_over_lateral_median']:.2f}")
    print(Path(args.out_dir) / "archive_summary.json")


if __name__ == "__main__":
    main()
