"""Tune the Phase 3 baselines (CV/CA fit length k, Kalman q and r) on the TRAIN windows only (decision 2026-10-16).

Usage: python scripts/forecast/tune_baselines.py [--split train] [--out experiments/forecast/tuning]
"""
import argparse
import json
import time
from pathlib import Path

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.tuning import check_split, tune
from dronepose.forecast.windows import make_windows

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--trajectories", default="data/processed/forecast")
ap.add_argument("--config", default="configs/forecast.json")
ap.add_argument("--out", default="experiments/forecast/tuning")
ap.add_argument("--allow-nontrain", action="store_true", help="tune on another split (not for reported numbers)")
ap.add_argument("--allow-test", action="store_true")
args = ap.parse_args()

if args.split != "train" and not args.allow_nontrain:
    raise SystemExit(f"tuning is done on the train split only (got {args.split!r}); pass --allow-nontrain to override")
check_split(args.split, args.allow_test)
cfg = load_config(Path(args.config))
t0 = time.time()
windows = make_windows(load_trajectories(Path(args.trajectories), args.split), cfg.obs_len, cfg.pred_len, cfg.fps)
print(f"{args.split}: {len(windows)} finestre ({int(windows.metric.sum())} in metri)", flush=True)
params = tune(windows, cfg)
params.update({"split": args.split, "n_windows": len(windows), "n_metric_windows": int(windows.metric.sum())})
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "tuning.json").write_text(json.dumps(params, indent=1))
print(f"scelte: cv_k {params['cv_k']} | ca_k {params['ca_k']} | kf_cv {params['kf_cv']} | kf_ca {params['kf_ca']}")
print("ADE a 1 s (train, metri):", {m: round(s, 4) for m, s in params["scores"].items()})
print("avvisi di griglia:", params["edge_warnings"] or "nessuno")
print(f"scritto {out / 'tuning.json'} in {time.time() - t0:.1f} s")
