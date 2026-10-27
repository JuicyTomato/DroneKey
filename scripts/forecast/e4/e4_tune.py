"""E4b (spec 2026-10-27 §5.5): baselines re-tuned on the 6DroneSyn train with real pose errors.

Sets: fold03, fold09, fold14 (tracts from the two other archive scenes) and final (all three). One tract per train
window, same distance band as the last observation and line of sight aligned in azimuth, predicted-class scale;
wide grids (RETUNE_GRIDS) and the line-of-sight Kalman CA. Drone 08 (dataset units, separate row): tuned apart on its
own windows with tracts of drone 08 (marked metric only inside the tuning, values in dataset units). E4a keeps the
parameters tuned on the clean train (experiments/forecast/tuning/tuning.json).

Usage: python scripts/forecast/e4/e4_tune.py [--jobs 4] [--out data/processed/forecast_archive/e4]
"""
import argparse
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dronepose.forecast.archive_noise import (ARCHIVE_PATHS, ARCHIVE_SCENES, archive_noise_rng, fold_train,
                                              load_tracts, noisy_windows)
from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.noise import RETUNE_GRIDS, chosen_params, with_grids
from dronepose.forecast.tuning import tune
from dronepose.forecast.windows import make_windows

SETS = {"fold03": "03", "fold09": "09", "fold14": "14", "final": None}
TUNE_SEED = 4


def _job(name, metric, data, config):
    t0 = time.time()
    cfg = with_grids(load_config(Path(config)), RETUNE_GRIDS)
    w = make_windows(load_trajectories(Path(data), "train"), cfg.obs_len, cfg.pred_len, cfg.fps)
    w = w.subset(w.metric == metric)
    if not metric:
        w = dataclasses.replace(w, metric=np.ones(len(w), bool))       # tuning objective on the 08 windows
    scenes = fold_train(SETS[name])
    tracts = load_tracts(list(ARCHIVE_PATHS)).select(scenes, meters=metric)
    assert set(tracts.scene) == set(scenes), (name, set(tracts.scene))
    wn = noisy_windows(w, tracts, archive_noise_rng(TUNE_SEED, list(SETS).index(name), int(metric)),
                       distance_cond=True)
    params = tune(wn, cfg, los=True)
    out = {**chosen_params(params), "kf_ca_los": params["kf_ca_los"], "scores": params["scores"],
           "edge_warnings": params["edge_warnings"], "set": name, "tract_scenes": list(scenes),
           "group": "metri (01-06)" if metric else "drone 08 (unità del dataset)", "n_windows": len(w),
           "n_tracts": len(tracts), "lost_frame_fraction": float(1 - np.mean(wn.obs_valid)),
           "grids": {k: list(v) for k, v in RETUNE_GRIDS.items()}, "seconds": time.time() - t0}
    return name, metric, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/forecast")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--out", default="data/processed/forecast_archive/e4")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, m) for m in (True, False) for n in SETS]
    with ProcessPoolExecutor(args.jobs) as ex:
        for name, metric, res in ex.map(_job, *zip(*jobs), [args.data] * len(jobs), [args.config] * len(jobs)):
            f = out / (f"params_{name}.json" if metric else f"params_08_{name}.json")
            f.write_text(json.dumps(res, indent=1))
            s = res["scores"]
            print(f"{f.name}: {res['seconds']:.0f} s, cv_k {res['cv_k']}, ca_k {res['ca_k']}, kf_ca {res['kf_ca']}, "
                  f"kf_ca_los {res['kf_ca_los']}; ADE 1 s train cv {s['cv']:.3f} kf_ca {s['kf_ca']:.3f} "
                  f"kf_ca_los {s['kf_ca_los']:.3f}; avvisi {res['edge_warnings'] or 'nessuno'}", flush=True)


if __name__ == "__main__":
    main()
