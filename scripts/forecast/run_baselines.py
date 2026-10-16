"""Evaluate the Phase 3 baselines (parameters tuned on train) on the val windows and write the report.

Usage: python scripts/forecast/run_baselines.py [--split val] [--params experiments/forecast/tuning/tuning.json]
                                                [--out experiments/forecast/baselines_val] [--allow-test]
"""
import argparse
import json
import time
from pathlib import Path

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.report import summarize, write_report
from dronepose.forecast.tuning import check_split, predict_all
from dronepose.forecast.windows import make_windows

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--trajectories", default="data/processed/forecast")
ap.add_argument("--params", default="experiments/forecast/tuning/tuning.json")
ap.add_argument("--config", default="configs/forecast.json")
ap.add_argument("--out", default=None, help="default experiments/forecast/baselines_<split>")
ap.add_argument("--allow-test", action="store_true")
args = ap.parse_args()

check_split(args.split, args.allow_test)
cfg = load_config(Path(args.config))
params = json.loads(Path(args.params).read_text())
if params.get("split") != "train":
    print(f"ATTENZIONE: parametri tarati su {params.get('split')!r}, non sul train")
t0 = time.time()
windows = make_windows(load_trajectories(Path(args.trajectories), args.split), cfg.obs_len, cfg.pred_len, cfg.fps)
print(f"{args.split}: {len(windows)} finestre ({int(windows.metric.sum())} in metri)", flush=True)
preds = predict_all(windows, params, cfg)
summary = summarize(windows, preds, cfg)
summary.update({"split": args.split, "params": {k: params[k] for k in ("cv_k", "ca_k", "kf_cv", "kf_ca")},
                "edge_warnings": params.get("edge_warnings", [])})
out = Path(args.out or f"experiments/forecast/baselines_{args.split}")
write_report(summary, windows, preds, out)
g = next(g for g in summary["groups"] if g["group"] == "metri: tutti")
print("metri: tutti — FDE media a 1 s e skill ADE 1 s vs cv:")
for k in summary["methods"]:
    sk = g["skill"][k]["ade_1s"]
    print(f"  {k:8s} FDE {g['position'][k]['fde_mean'][-1]:.3f} m | skill {100 * sk[0]:+.1f}% [{100 * sk[1]:+.1f}, {100 * sk[2]:+.1f}]")
print(f"scritto {out} in {time.time() - t0:.1f} s")
