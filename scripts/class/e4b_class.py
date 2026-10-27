"""E4b on val with the class maps of the class grid: each cell against the base (C1+V) on the same windows.

Usage (project venv, from the repo root):
  python scripts/class/e4b_class.py --cells C1+W+g C1+L+g [--base C1+V] [--device auto] [--bootstrap-b 10000]
      [--out experiments/class/e4b/e4b_class.json]
Poses: experiments/class/e4b/<cell>/poses/poses_val_s{0,1,2}_causal.jsonl (scripts/export_poses.py --class-map); the
true-class row uses the base folder's _gt files. Parameters and nets are those of E4b (fold of each scene), unchanged.
ADE at 1 s on metric windows; differences paired on the same windows (pose seed, copy, GT window), bootstrap per
trajectory with scenes as strata. Val only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecast" / "e4"))
from dronepose.forecast.config import load_config  # noqa: E402
from dronepose.forecast.dataset import load_trajectories  # noqa: E402
from dronepose.forecast.e4_eval import align_window_rows, degradation, pipeline_errors  # noqa: E402
from dronepose.forecast.nets.device import pick_device  # noqa: E402
from dronepose.forecast.windows import make_windows  # noqa: E402
from e4_train import ARMS, e4_run_name  # noqa: E402

SEEDS = (0, 1, 2)
METHODS = ("R-CV-6D", "G-6D", "kf_ca", "kf_ca_los", "CV-2")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--base", default="C1+V")
    ap.add_argument("--root", default="experiments/class/e4b")
    ap.add_argument("--data", default="data/processed/forecast")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--params", default="data/processed/forecast_archive/e4")
    ap.add_argument("--runs", default="experiments/forecast/e4/runs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bootstrap-b", type=int, default=10000)
    ap.add_argument("--out", default="experiments/class/e4b/e4b_class.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg, device, runs, pdir = load_config(Path(args.config)), pick_device(args.device), Path(args.runs), Path(args.params)
    trajs = load_trajectories(Path(args.data), "val")
    gt = make_windows(trajs, cfg.obs_len, cfg.pred_len, cfg.fps)
    fold = {s: json.loads((pdir / f"params_fold{s}.json").read_text()) for s in ("03", "09", "14")}
    fold08 = {s: json.loads((pdir / f"params_08_fold{s}.json").read_text()) for s in ("03", "09", "14")}
    params_for = lambda scene, metric: fold[scene] if metric else fold08[scene]  # noqa: E731
    nets_for = lambda scene: {arm: [runs / e4_run_name(arm, f"fold{scene}", s) for s in SEEDS] for arm in ARMS}  # noqa: E731

    def errors(cell: str, cls: str):
        paths = [Path(args.root) / cell / "poses" / f"poses_val_s{k}_{cls}.jsonl" for k in SEEDS]
        err, _, idx, info, keys = pipeline_errors(trajs, paths, cfg, params_for, nets_for, device, with_keys=True)
        print(f"{cell} {cls}: {info} ({time.time() - t0:.0f} s)", flush=True)
        return err, idx, keys

    base = errors(args.base, "causal")
    rows = {f"{args.base} (classe vera)": errors(args.base, "gt")}
    rows.update({c: errors(c, "causal") for c in args.cells if c != args.base})
    out = {"base": args.base, "methods": METHODS, "cells": {}}
    for name, (err, idx, keys) in rows.items():
        ia, ib = align_window_rows(keys, base[2])
        w = gt.subset(base[1][ib])
        e_new = {m: err[m][ia] for m in METHODS}
        e_base = {m: base[0][m][ib] for m in METHODS}
        d = degradation(e_new, e_base, w, args.bootstrap_b)
        out["cells"][name] = {"n_paired": int(len(ia)), "n_rows": int(len(keys)), "n_base_rows": int(len(base[2])),
                              "vs_base": {m: {"ade_1s": d[m]["ade_1s_e4b"], "ade_1s_base": d[m]["ade_1s_e4a"],
                                              "diff": d[m]["mean"], "ci": d[m]["ci"]} for m in METHODS}}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=1))
    lines = ["# E4b sulla val per classe del drone (ADE 1 s, m, finestre in metri)", "",
             f"Differenze contro {args.base} sulle stesse finestre, IC 95% bootstrap per traiettoria.", "",
             "| Cella | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    first = next(iter(out["cells"].values()))["vs_base"]
    lines.append(f"| {args.base} | " + " | ".join(f"{first[m]['ade_1s_base']:.3f}" for m in METHODS) + " |")
    for name, c in out["cells"].items():
        v = c["vs_base"]
        lines.append(f"| {name} | " + " | ".join(f"{v[m]['ade_1s']:.3f} ({v[m]['diff']:+.3f} [{v[m]['ci'][0]:+.3f}, "
                                                f"{v[m]['ci'][1]:+.3f}])" for m in METHODS) + " |")
    outp.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"-> {outp} (+ .md) in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
