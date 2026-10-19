"""Sensitivity of the baselines to Gaussian noise on the observed poses (spec §5.5), on the generated val.

Position noise sigma in {0, 1, 2, 3, 5} cm paired with rotation noise in {0, 0.5, 1, 2, 3} deg (random rotation
vector), 3 seeds. Weighted (natural) ADE / FDE at 1 s per method. Not a substitute for the realistic pose errors of E4b.

Default: parameters tuned on the clean generated train (noise.json / noise.png).
--retune: per level and seed, the same noise on the generated TRAIN (independent draws), every baseline re-tuned on it
(importance-weighted objective, wider grids than configs/forecast.json, override with --grid), then evaluated on the
noisy val (noise_retuned.json / noise_retuned.png, with the physics_acro_gen vs kf_ca skill per level).

Usage: python scripts/forecast/noise_sensitivity.py [--split val] [--params experiments/forecast/generator/tuning.json]
       python scripts/forecast/noise_sensitivity.py --retune [--jobs 4] [--grid kf_r_grid=1e-6,1e-2,1e-1 ...]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.noise import (RETUNE_GRIDS, SIGMA_P_CM, SIGMA_R_DEG, add_pose_noise, evaluate_positions,
                                      level_key, noise_rng, parse_grid, retune_level, summarize_retuned, with_grids)
from dronepose.forecast.tuning import check_split
from dronepose.forecast.windows import freeze_windows

_STATE = {}


def _init(data, split, cfg):
    _STATE["cfg"] = cfg
    _STATE["train"], _STATE["val"] = (freeze_windows(load_trajectories(Path(data), s), cfg.obs_len, cfg.pred_len,
                                                     cfg.fps) for s in ("train", split))


def _job(sp, sr, seed):
    t0 = time.time()
    run = retune_level(_STATE["train"], _STATE["val"], sp, sr, seed, _STATE["cfg"])
    run["seconds"] = time.time() - t0
    return run


def _plot(path, title, res, ref=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = [level_key(sp, sr) for sp, sr in zip(SIGMA_P_CM, SIGMA_R_DEG)]
    x = np.arange(len(levels))
    methods = [k for k in res[levels[0]] if "ade_1s" in res[levels[0]][k] and "skill" not in k]
    axes = plt.subplots(1, 3 if ref is not None else 2, figsize=(16, 4.2) if ref is not None else (11, 4))[1]
    colors = plt.get_cmap("tab20").colors
    for i, k in enumerate(methods):
        style = "-" if k.startswith("physics") else "--"           # physics solid, kinematic dashed
        for ax, m in zip(axes, ("ade_1s", "fde_1s")):
            ax.plot(x, [res[lv][k][m] for lv in levels], style, marker="o", color=colors[2 * i % 20 + i // 10],
                    label=k)
    for ax, m in zip(axes, ("ADE 1 s (m, pesata)", "FDE 1 s (m, pesata)")):
        ax.set_ylabel(m)
        ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    if ref is not None:
        ax = axes[2]
        for r, style, lab in ((res, "-o", "ritarati sul train rumoroso"), (ref, "--s", "tarati sul train pulito")):
            ax.plot(x, [1 - r[lv]["physics_acro_gen"]["ade_1s"] / r[lv]["kf_ca"]["ade_1s"] for lv in levels], style,
                    label=lab)
        seeds = [res[lv]["skill_acro_gen_vs_kf_ca"]["per_seed_ade_1s"] for lv in levels]
        for i, s in enumerate(seeds):
            ax.scatter([i] * len(s), s, color="k", s=8, zorder=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("skill ADE 1 s: 1 - acro_gen / kf_ca")
        ax.legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x, [f"{sp} cm\n{sr}°" for sp, sr in zip(SIGMA_P_CM, SIGMA_R_DEG)])
        ax.grid(alpha=0.3)
    fig = axes[0].figure
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def clean_params(args, cfg, out):
    params = json.loads(Path(args.params).read_text())
    clean = freeze_windows(load_trajectories(Path(args.data), args.split), cfg.obs_len, cfg.pred_len, cfg.fps)
    res = {}
    for sp, sr in zip(SIGMA_P_CM, SIGMA_R_DEG):
        acc = {}
        for seed in range(args.seeds if sp or sr else 1):
            for k, v in evaluate_positions(add_pose_noise(clean, sp, sr, noise_rng(seed, sr)), params, cfg).items():
                acc.setdefault(k, []).append((v["ade_1s"], v["fde_1s"]))
        res[level_key(sp, sr)] = {k: {"ade_1s": float(np.mean([a for a, _ in v])),
                                      "fde_1s": float(np.mean([f for _, f in v]))} for k, v in acc.items()}
        print(f"rumore {sp} cm / {sr}°: " + ", ".join(f"{k} {v['ade_1s']:.3f}" for k, v in res[level_key(sp, sr)].items()),
              flush=True)
    (out / "noise.json").write_text(json.dumps({"sigma_p_cm": SIGMA_P_CM, "sigma_r_deg": SIGMA_R_DEG,
                                                "seeds": args.seeds, "split": args.split, "results": res}, indent=1))
    try:
        _plot(out / "noise.png", f"Sensibilità al rumore gaussiano sulle pose osservate ({args.split} generata)", res)
    except ImportError:
        print("matplotlib assente: figura saltata")
    return out / "noise.json"


def retuned(args, cfg, out):
    grids = dict(RETUNE_GRIDS, **dict(parse_grid(g) for g in args.grid))
    cfg = with_grids(cfg, grids)
    jobs = [(sp, sr, seed) for sp, sr in zip(SIGMA_P_CM, SIGMA_R_DEG) for seed in range(args.seeds if sp or sr else 1)]
    print(f"{len(jobs)} tarature su {args.jobs} processi, griglie {grids}", flush=True)
    runs = []
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(args.data, args.split, cfg)) as pool:
        for f in as_completed([pool.submit(_job, *j) for j in jobs]):
            r = f.result()
            runs.append(r)
            print(f"rumore {r['sigma_p_cm']} cm / {r['sigma_r_deg']}° seme {r['seed']} ({r['seconds']:.0f} s): "
                  f"kf_ca {r['val']['kf_ca']['ade_1s']:.3f} [q {r['params']['kf_ca']['q']:g} r {r['params']['kf_ca']['r']:g}]"
                  f", acro_gen {r['val']['physics_acro_gen']['ade_1s']:.3f} {r['params']['acro_gen']}", flush=True)
    res = summarize_retuned(sorted(runs, key=lambda r: (r["sigma_p_cm"], r["seed"])))
    for key, lvl in res.items():
        print(f"{key}: skill acro_gen vs kf_ca ADE {lvl['skill_acro_gen_vs_kf_ca']['ade_1s']:+.3f} "
              f"FDE {lvl['skill_acro_gen_vs_kf_ca']['fde_1s']:+.3f}")
    (out / "noise_retuned.json").write_text(json.dumps(
        {"sigma_p_cm": SIGMA_P_CM, "sigma_r_deg": SIGMA_R_DEG, "seeds": args.seeds, "split": args.split,
         "tuned_on": "train generato con lo stesso livello di rumore (estrazioni indipendenti, salt 1)",
         "grids": grids, "results": res, "runs": runs}, indent=1))
    ref_path = out / "noise.json"
    ref = json.loads(ref_path.read_text())["results"] if ref_path.exists() else None
    try:
        _plot(out / "noise_retuned.png", f"Rumore sulle pose osservate, baseline ritarate sul train rumoroso "
                                         f"({args.split} generata)", res, ref)
    except ImportError:
        print("matplotlib assente: figura saltata")
    return out / "noise_retuned.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--data", default="data/processed/forecast_gen")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--params", default="experiments/forecast/generator/tuning.json")
    ap.add_argument("--out", default="experiments/forecast/generator")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--retune", action="store_true", help="re-tune every baseline on the train with the same noise")
    ap.add_argument("--grid", action="append", default=[], help="NAME=v1,v2,... overrides a --retune grid")
    ap.add_argument("--jobs", type=int, default=4, help="--retune: parallel tunings")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()

    check_split(args.split, args.allow_test)
    cfg = load_config(Path(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    path = retuned(args, cfg, out) if args.retune else clean_params(args, cfg, out)
    print(f"scritto {path} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
