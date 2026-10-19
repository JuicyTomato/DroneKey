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


# ----- generated sets (plan step 3.1b) ---------------------------------------------------------------------------

LEVELS = ("basso", "medio", "alto")
REQUIRED_META = ("mode", "style", "cell", "abrupt_last_second", "flip", "level")
CELL_INDEX = {"|ω_vert|": 0, "|ω_oriz|": 1, "|a|": 2, "|v|": 3, "t dal cambio": 4}   # stratify.STRAT_VARS order


def _generated_groups(w: Windows) -> list[tuple[str, np.ndarray]]:
    m = w.meta
    cells = np.array([[int(ch) for ch in c] for c in m["cell"]]).reshape(len(w), -1)
    out = [("tutti", np.ones(len(w), bool))]
    out += [(f"modo {x}", m["mode"] == x) for x in sorted(set(m["mode"]))]
    out += [(f"stile {x}", m["style"] == x) for x in sorted(set(m["style"]))]
    for name in ("|ω_vert|", "|ω_oriz|", "t dal cambio"):
        out += [(f"{name} {LEVELS[lv]}", cells[:, CELL_INDEX[name]] == lv) for lv in range(3)]
    ab, flip = np.asarray(m["abrupt_last_second"], bool), np.asarray(m["flip"], bool)
    out += [("brusco nell'ultimo secondo", ab), ("senza bruschi nell'ultimo secondo", ~ab), ("flip", flip),
            ("senza flip", ~flip)]
    out += [(f"livello {x}", m["level"] == x) for x in sorted(set(m["level"]))]
    out.append(("regime", (cells[:, 0] == 0) & (cells[:, 1] == 0)))     # |ω_vert| and |ω_oriz| in the low tercile
    return [g for g in out if g[1].any()]


def _wmean(x: np.ndarray, wt: np.ndarray) -> float:
    """Weighted mean; NaN for a group the natural distribution never produces (all weights zero)."""
    return float(np.average(x, weights=wt)) if wt.sum() > 0 else float("nan")


def _oracle_c4(w: Windows) -> dict:
    """C4: on level-0 acro samples the true dynamics from the true freeze state reproduces the future (< 1e-4 m)."""
    from dronepose.forecast.acro import acro_oracle
    from dronepose.forecast.dynamics import QuadParams, QuadState, orthonormalize

    definition = "oracolo acro (stato, parametri e comandi veri), val livello 0: errore massimo a 1 s < 1e-4 m"
    m = w.meta
    sel = (np.asarray(m["mode"]) == "acro") & (np.asarray(m["level"]) == 0)
    if not sel.any() or (np.asarray(m["level"]) != 0).any():
        return {"definition": definition, "value": None, "ci": [None, None], "pass": None, "n": 0,
                "note": "si valuta sulla val al livello 0"}
    B = int(sel.sum())
    col = lambda k: np.asarray(m[k], float)[sel]
    state = QuadState(p=np.zeros((B, 3)), v=col("fz_v"), R=orthonormalize(np.asarray(w.obs_R, float)[sel, -1]),
                      omega=col("fz_omega"), c=col("fz_c"))
    prm = QuadParams(k_d=col("k_d"), tau_w=col("tau_w"), tau_c=col("tau_c"), k_q=np.zeros(B), d_xy=np.zeros((B, 2)),
                     c_max=np.full(B, np.inf), delta=np.zeros(B), level=np.zeros(B, int))
    pred = acro_oracle(state, prm, col("fz_omega_cmd"), col("fz_c_cmd"), w.fut_t)
    err = float(np.linalg.norm(pred[:, -1] - w.fut_p[sel, -1], axis=-1).max())
    return {"definition": definition, "value": err, "ci": [None, None], "pass": bool(err < 1e-4), "n": B}


def summarize_generated(windows: Windows, preds: dict, cfg: ForecastConfig, weights, best_kin: str,
                        bootstrap_b: int | None = None) -> dict:
    """Report on generated windows: natural-weighted and balanced (the stratified set as is) versions, groups from
    meta, skill vs CV and vs the best kinematic baseline (chosen on train), criteria C1-C4 (spec §5.7)."""
    from dronepose.forecast.metrics import bootstrap_counts, skill_ci
    from dronepose.forecast.tuning import MODE_METHOD

    w, hs = windows, cfg.horizons_frames
    missing = [k for k in REQUIRED_META if k not in w.meta]
    if missing:
        raise ValueError(f"not generated windows: meta lacks {missing} (use summarize for 6DroneSyn)")
    b = cfg.bootstrap_b if bootstrap_b is None else bootstrap_b
    H = len(w.fut_t)
    methods = list(POSITION_METHODS) + ([MODE_METHOD] if MODE_METHOD in preds else [])
    err = {k: position_errors(preds[k], w.fut_p) for k in methods}
    ade1 = {k: ade_at(e, H) for k, e in err.items()}
    fde1 = {k: fde_at(e, H) for k, e in err.items()}
    versions = {"weighted": np.asarray(weights, float), "balanced": np.ones(len(w))}
    out = {name: {"groups": []} for name in versions}
    for gname, mask in _generated_groups(w):
        inv, counts = bootstrap_counts(w.traj[mask], w.meta["mode"][mask], b, cfg.bootstrap_seed)
        for vname, wt_all in versions.items():
            wt = wt_all[mask]
            g = {"group": gname, "n_windows": int(mask.sum()), "weight_sum": float(wt.sum()), "position": {},
                 "skill_cv": {}, "skill_best": {}}
            for k in methods:
                g["position"][k] = {"ade_mean": [_wmean(ade_at(err[k][mask], h), wt) for h in hs],
                                    "fde_mean": [_wmean(fde_at(err[k][mask], h), wt) for h in hs]}
                for key, ref in (("skill_cv", REFERENCE), ("skill_best", best_kin)):
                    g[key][k] = {mt: list(skill_ci(wt * v[k][mask], wt * v[ref][mask], inv, counts))
                                 for mt, v in (("ade_1s", ade1), ("fde_1s", fde1))}
            out[vname]["groups"].append(g)
    groups = {g["group"]: g for g in out["weighted"]["groups"]}
    crit = {}
    hi = groups.get("|ω_oriz| alto")
    s = hi["skill_best"]["physics_acro_gen"]["ade_1s"] if hi else [float("nan")] * 3
    crit["C1"] = {"definition": f"|ω_oriz| alto, pesata: skill ADE 1 s di physics_acro_gen vs {best_kin} > 0 "
                                "con IC 95% che esclude lo zero", "value": s[0], "ci": s[1:], "pass": bool(s[1] > 0)}
    rg = groups.get("regime")
    s = rg["skill_best"]["physics_acro_gen"]["ade_1s"] if rg else [float("nan")] * 3
    crit["C2"] = {"definition": f"regime, pesata: skill ADE 1 s di physics_acro_gen vs {best_kin} ≥ −0.10 "
                                "(unilaterale, confermato 2026-10-19)", "value": s[0], "ci": s[1:], "pass": bool(s[0] >= -0.10)}
    if MODE_METHOD in preds:
        t = groups["tutti"]
        gap = t["position"]["physics_acro_gen"]["ade_mean"][-1] - t["position"][MODE_METHOD]["ade_mean"][-1]
        inv, counts = bootstrap_counts(w.traj, w.meta["mode"], b, cfg.bootstrap_seed)
        wt = versions["weighted"]
        sk = skill_ci(wt * ade1[MODE_METHOD], wt * ade1["physics_acro_gen"], inv, counts)
        n = counts.shape[1]
        boot = (counts @ np.bincount(inv, weights=wt * (ade1["physics_acro_gen"] - ade1[MODE_METHOD]), minlength=n)) \
            / (counts @ np.bincount(inv, weights=wt, minlength=n))
        crit["C3"] = {"definition": "tutti, pesata: ADE 1 s (gen) − ADE 1 s (modo noto) ≥ 0 m (divario riportato, "
                                    "IC bootstrap del divario in m)", "value": gap,
                      "ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                      "skill": sk[0], "skill_ci": list(sk[1:]), "pass": bool(gap >= 0)}
    else:
        crit["C3"] = {"definition": "modo noto vs modo ignoto", "value": None, "ci": [None, None], "pass": None}
    crit["C4"] = _oracle_c4(w)
    return {"n_windows": len(w), "reference": REFERENCE, "best_kinematic": best_kin, "horizons_frames": list(hs),
            "horizons_s": [h / cfg.fps for h in hs], "bootstrap_b": b, "bootstrap_strata": "modo",
            "methods": methods, "versions": out, "criteria": crit}


def _markdown_generated(s: dict, title: str) -> str:
    hs = " / ".join(f"{h:.2f}" for h in s["horizons_s"])
    lines = [f"# {title} — {s['n_windows']} campioni", "",
             f"Migliore cinematica (sul train): `{s['best_kinematic']}`. IC 95% bootstrap appaiato per campione, "
             f"stratificato per modo ({s['bootstrap_b']} repliche). Orizzonti (s): {hs}.", "", "## Criteri (spec §5.7)", ""]
    for c, v in s["criteria"].items():
        val = "—" if v["value"] is None else f"{v['value']:+.4g}"
        ci = "" if v["ci"][0] is None else f" [{v['ci'][0]:+.4g}, {v['ci'][1]:+.4g}]"
        esito = {True: "**passa**", False: "**non passa**", None: "n/d"}[v["pass"]]
        sk = f"; skill del modo noto vs gen {_fmt_skill([v['skill'], *v['skill_ci']])}" if "skill_ci" in v else ""
        lines.append(f"- **{c}** — {v['definition']}: {val}{ci}{sk} → {esito}")
    for vname, label in (("weighted", "pesata (distribuzione naturale)"), ("balanced", "bilanciata (insieme stratificato)")):
        lines += ["", f"## Versione {label}", ""]
        for g in s["versions"][vname]["groups"]:
            lines += [f"### {g['group']} ({g['n_windows']} campioni)", "",
                      f"| metodo | ADE media | FDE media | skill ADE 1 s vs cv | skill ADE 1 s vs {s['best_kinematic']} |",
                      "|---|---|---|---|---|"]
            for k in s["methods"]:
                p = g["position"][k]
                lines.append(f"| {k} | {' / '.join(f'{x:.3f}' for x in p['ade_mean'])} | "
                             f"{' / '.join(f'{x:.3f}' for x in p['fde_mean'])} | {_fmt_skill(g['skill_cv'][k]['ade_1s'])} | "
                             f"{_fmt_skill(g['skill_best'][k]['ade_1s'])} |")
            lines.append("")
    return "\n".join(lines)


def write_generated_report(summary: dict, out_dir: Path, title: str = "Baseline sulla val generata") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps(summary, indent=1))
    (out / "table.md").write_text(_markdown_generated(summary, title))
