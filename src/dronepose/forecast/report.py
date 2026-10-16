"""Baseline tables: ADE/FDE per horizon and group, skill vs CV with paired bootstrap, orientation errors, report files."""
import csv
import json
from pathlib import Path

import numpy as np

from dronepose.forecast.config import ForecastConfig
from dronepose.forecast.metrics import (ade_at, fde_at, geodesic_deg_batch, metric_group, paired_bootstrap_skill,
                                        position_errors, up_angle_deg)
from dronepose.forecast.tuning import ORIENTATION_METHODS, POSITION_METHODS
from dronepose.forecast.windows import Windows

REFERENCE = "cv"
NOTES_NON_METRIC = ["unità del dataset: nessuna dimensione verificata del drone, fuori dalle medie in metri",
                    "geodetico solo oracle: con pose stimate l'imbardata è ambigua (usare l'asse su)",
                    "fisica nominale: g applicato a unità del dataset"]


def _groups(w: Windows) -> list[tuple[str, np.ndarray, str, bool]]:
    """(name, mask, units, with confidence interval)."""
    m = metric_group(w)
    out = [("metri: tutti", m, "m", True),
           ("metri: lineare", m & (w.motion == "linear"), "m", True),
           ("metri: non lineare", m & (w.motion == "nonlinear"), "m", True)]
    out += [(f"metri: scena {s}", m & (w.scene == s), "m", True) for s in sorted(set(w.scene[m]))]
    out += [(f"drone {d}", m & (w.drone == d), "m", False) for d in sorted(set(w.drone[m]))]
    out += [(f"drone {d} (unità dataset)", ~m & (w.drone == d), "dataset", False) for d in sorted(set(w.drone[~m]))]
    return [g for g in out if g[1].any()]


def summarize(windows: Windows, preds: dict, cfg: ForecastConfig, bootstrap_b: int | None = None) -> dict:
    w, hs = windows, cfg.horizons_frames
    b = cfg.bootstrap_b if bootstrap_b is None else bootstrap_b
    H = len(w.fut_t)
    err = {k: position_errors(preds[k], w.fut_p) for k in POSITION_METHODS}
    geo = {k: geodesic_deg_batch(preds[k], w.fut_R) for k in ORIENTATION_METHODS}
    up = {k: up_angle_deg(preds[k][..., :, 2], w.fut_u) for k in ORIENTATION_METHODS}
    groups = []
    for name, mask, units, with_ci in _groups(w):
        e = {k: v[mask] for k, v in err.items()}
        g = {"group": name, "units": units, "n_windows": int(mask.sum()),
             "n_traj": int(len(set(w.traj[mask]))), "notes": [] if units == "m" else list(NOTES_NON_METRIC),
             "position": {}, "skill": {}, "orientation": {}}
        for k in POSITION_METHODS:
            ades = [ade_at(e[k], h) for h in hs]
            fdes = [fde_at(e[k], h) for h in hs]
            g["position"][k] = {"ade_mean": [float(a.mean()) for a in ades], "ade_median": [float(np.median(a)) for a in ades],
                                "fde_mean": [float(f.mean()) for f in fdes], "fde_median": [float(np.median(f)) for f in fdes]}
            g["skill"][k] = {}
            for metric, fn in (("ade_1s", ade_at), ("fde_1s", fde_at)):
                a, r = fn(e[k], H), fn(e[REFERENCE], H)
                res = paired_bootstrap_skill(a, r, w.traj[mask], w.scene[mask], b, cfg.bootstrap_seed)
                g["skill"][k][metric] = list(res) if with_ci else [res[0]]
        for k in ORIENTATION_METHODS:
            g["orientation"][k] = {"geodesic_deg_at": [float(geo[k][mask][:, h - 1].mean()) for h in hs],
                                   "up_deg_at": [float(up[k][mask][:, h - 1].mean()) for h in hs]}
        if name == "metri: tutti":
            g["ade_curve"] = {k: [float(ade_at(e[k], h).mean()) for h in range(1, H + 1)] for k in POSITION_METHODS}
            g["fde_curve"] = {k: [float(x) for x in e[k].mean(0)] for k in POSITION_METHODS}
        groups.append(g)
    return {"n_windows": len(w), "n_traj": len(set(w.traj)), "reference": REFERENCE, "horizons_frames": list(hs),
            "horizons_s": [h / cfg.fps for h in hs], "bootstrap_b": b, "methods": list(POSITION_METHODS),
            "orientation_methods": list(ORIENTATION_METHODS), "groups": groups}


def _fmt_skill(v: list[float]) -> str:
    return f"{100 * v[0]:+.1f}%" + (f" [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]" if len(v) == 3 else "")


def _markdown(s: dict) -> str:
    hs = " / ".join(f"{h:.2f}" for h in s["horizons_s"])
    lines = [f"# Baseline Fase 3 — {s['n_windows']} finestre, {s['n_traj']} traiettorie",
             "", f"Riferimento della skill: `{s['reference']}`. IC 95% bootstrap appaiato per traiettoria, "
             f"stratificato per scena ({s['bootstrap_b']} repliche). Orizzonti (s): {hs}.", ""]
    for g in s["groups"]:
        u = "m" if g["units"] == "m" else "unità dataset"
        lines += [f"## {g['group']} ({g['n_windows']} finestre, {g['n_traj']} traiettorie, {u})", ""]
        lines += [f"- {n}" for n in g["notes"]] + ([""] if g["notes"] else [])
        lines += ["| metodo | ADE media | FDE media | FDE mediana | skill ADE 1 s | skill FDE 1 s |", "|---|---|---|---|---|---|"]
        for k in s["methods"]:
            p, sk = g["position"][k], g["skill"][k]
            lines.append(f"| {k} | {' / '.join(f'{x:.3f}' for x in p['ade_mean'])} | "
                         f"{' / '.join(f'{x:.3f}' for x in p['fde_mean'])} | {' / '.join(f'{x:.3f}' for x in p['fde_median'])} | "
                         f"{_fmt_skill(sk['ade_1s'])} | {_fmt_skill(sk['fde_1s'])} |")
        lines += ["", "| orientamento | geodetico ° | asse su ° |", "|---|---|---|"]
        for k in s["orientation_methods"]:
            o = g["orientation"][k]
            lines.append(f"| {k} | {' / '.join(f'{x:.1f}' for x in o['geodesic_deg_at'])} | "
                         f"{' / '.join(f'{x:.1f}' for x in o['up_deg_at'])} |")
        lines.append("")
    return "\n".join(lines)


def write_report(summary: dict, windows: Windows, preds: dict, out_dir: Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps(summary, indent=1))
    (out / "table.md").write_text(_markdown(summary))
    w, hs = windows, summary["horizons_frames"]
    with open(out / "per_window.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["window", "trajectory", "scene", "drone", "start", "units", "method"]
                    + [f"ade_{h}" for h in hs] + [f"fde_{h}" for h in hs])
        for k in POSITION_METHODS:
            e = position_errors(preds[k], w.fut_p)
            ades, fdes = np.stack([ade_at(e, h) for h in hs], 1), np.stack([fde_at(e, h) for h in hs], 1)
            for i in range(len(w)):
                wr.writerow([i, w.traj_keys[w.traj[i]], w.scene[i], w.drone[i], int(w.start[i]),
                             "m" if w.metric[i] else "dataset", k] + [f"{x:.6g}" for x in ades[i]] + [f"{x:.6g}" for x in fdes[i]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib assente: figura saltata (pip install -e '.[plots]')")
        return
    g = next(g for g in summary["groups"] if g["group"] == "metri: tutti")
    t = np.arange(1, len(next(iter(g["ade_curve"].values()))) + 1) / (hs[-1] / summary["horizons_s"][-1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, key, label in ((axes[0], "ade_curve", "ADE medio (m)"), (axes[1], "fde_curve", "FDE medio (m)")):
        for k, curve in g[key].items():
            ax.plot(t, curve, label=k)
        ax.set_xlabel("orizzonte (s)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Baseline, {g['group']} ({g['n_windows']} finestre)")
    fig.tight_layout()
    fig.savefig(out / "ade_fde_vs_horizon.png", dpi=150)
    plt.close(fig)
