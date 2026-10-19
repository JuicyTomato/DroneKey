"""Evaluate all baselines on a generated set (plan step 3.1b): weighted and balanced report, criteria C1-C4.

Usage: python scripts/forecast/run_generated.py --split val [--level 1] [--tune]
           [--params experiments/forecast/generator/tuning.json] [--out experiments/forecast/generator/val_L1]

--tune first tunes every baseline on the GENERATED TRAIN (importance-weighted objective) and writes the params.
Level 0 / 2 read the val regenerated at that level (val_L0, val_L2: same samples). `--split active` (the pilot keeps
flying) also writes how much every method degrades against the frozen-sticks val. When the val reports of levels
0, 1 and 2 exist, the mismatch curve (physics advantage vs level) is drawn.
"""
import argparse
import json
import time
from pathlib import Path

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.report import write_generated_report, summarize_generated
from dronepose.forecast.tuning import best_kinematic, check_split, predict_all, tune
from dronepose.forecast.windows import freeze_windows

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--level", type=int, default=1)
ap.add_argument("--data", default="data/processed/forecast_gen")
ap.add_argument("--config", default="configs/forecast.json")
ap.add_argument("--params", default="experiments/forecast/generator/tuning.json")
ap.add_argument("--out", default=None, help="default experiments/forecast/generator/<split>_L<level>")
ap.add_argument("--tune", action="store_true", help="tune on the generated train first")
ap.add_argument("--allow-test", action="store_true")
args = ap.parse_args()

check_split(args.split, args.allow_test)
cfg = load_config(Path(args.config))
data, root = Path(args.data), Path("experiments/forecast/generator")
params_path = Path(args.params)


def windows_of(name):
    return freeze_windows(load_trajectories(data, name), cfg.obs_len, cfg.pred_len, cfg.fps)


if args.tune:
    t0 = time.time()
    train = windows_of("train")
    params = tune(train, cfg)
    params.update({"split": "train (generato)", "n_windows": len(train), "seconds": time.time() - t0})
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(params, indent=1))
    print(f"taratura sul train generato ({len(train)} campioni) in {params['seconds']:.0f} s")
    print("ADE a 1 s (train, pesata):", {m: round(s, 4) for m, s in params["scores"].items()})
    print("avvisi di griglia:", params["edge_warnings"] or "nessuno")
params = json.loads(params_path.read_text())
best = best_kinematic(params)
name = args.split if args.level == 1 else f"{args.split}_L{args.level}"
t0 = time.time()
w = windows_of(name)
preds = predict_all(w, params, cfg)
summary = summarize_generated(w, preds, cfg, w.meta["weight"], best)
keep = ("cv_k", "ca_k", "kf_cv", "kf_ca", "ctr_k", "acro_k", "acro_kd", "acro_gen", "acro_mode")
summary.update({"split": name, "level": args.level, "params": {k: params[k] for k in keep if k in params},
                "params_from": str(params_path), "edge_warnings": params.get("edge_warnings", [])})
out = Path(args.out or root / f"{args.split}_L{args.level}")
write_generated_report(summary, out, f"Baseline su {name} (generato)")
print(f"{name}: {len(w)} campioni, migliore cinematica sul train: {best}")
tutti = next(g for g in summary["versions"]["weighted"]["groups"] if g["group"] == "tutti")
for k in summary["methods"]:
    sk = tutti["skill_best"][k]["ade_1s"]
    print(f"  {k:18s} ADE 1 s {tutti['position'][k]['ade_mean'][-1]:.3f} m | skill vs {best} {100 * sk[0]:+.1f}% "
          f"[{100 * sk[1]:+.1f}, {100 * sk[2]:+.1f}]")
for c, v in summary["criteria"].items():
    print(f"  {c}: {v['value']} {v['ci']} -> {v['pass']}")

if args.split == "active":
    frozen = root / "val_L1" / "table.json"
    if frozen.exists():
        f = json.loads(frozen.read_text())
        ft = next(g for g in f["versions"]["weighted"]["groups"] if g["group"] == "tutti")
        deg = {k: {"ade_1s_frozen": ft["position"][k]["ade_mean"][-1], "ade_1s_active": tutti["position"][k]["ade_mean"][-1]}
               for k in summary["methods"] if k in ft["position"]}
        for v in deg.values():
            v["ratio"] = v["ade_1s_active"] / v["ade_1s_frozen"]
        (out / "active_vs_frozen.json").write_text(json.dumps(deg, indent=1))
        print("pilota attivo / congelato (ADE 1 s, pesata):", {k: round(v["ratio"], 2) for k, v in deg.items()})

tables = {lv: root / f"val_L{lv}" / "table.json" for lv in (0, 1, 2)}
if args.split == "val" and all(p.exists() for p in tables.values()):
    curve = {}
    for lv, p in tables.items():
        s = json.loads(p.read_text())
        groups = {g["group"]: g for g in s["versions"]["weighted"]["groups"]}
        curve[lv] = {g: {"vs_best": groups[g]["skill_best"]["physics_acro_gen"]["ade_1s"],
                         "vs_cv": groups[g]["skill_cv"]["physics_acro_gen"]["ade_1s"]}
                     for g in ("tutti", "|ω_oriz| alto", "regime") if g in groups}
    (root / "mismatch.json").write_text(json.dumps(curve, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for g in ("tutti", "|ω_oriz| alto", "regime"):
            lv = [x for x in curve if g in curve[x]]
            y = [100 * curve[x][g]["vs_best"][0] for x in lv]
            lo = [100 * (curve[x][g]["vs_best"][0] - curve[x][g]["vs_best"][1]) for x in lv]
            hi = [100 * (curve[x][g]["vs_best"][2] - curve[x][g]["vs_best"][0]) for x in lv]
            ax.errorbar(lv, y, yerr=[lo, hi], marker="o", capsize=3, label=g)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks([0, 1, 2])
        ax.set_xlabel("livello di mismatch")
        ax.set_ylabel(f"skill ADE 1 s di physics_acro_gen vs {best} (%)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "mismatch.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib assente: figura saltata")
    print(f"curva di mismatch: {root / 'mismatch.png'}")
print(f"scritto {out} in {time.time() - t0:.0f} s")
