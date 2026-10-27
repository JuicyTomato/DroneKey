"""D2b (spec 2026-10-27 §5.3): base-predictor parameters tuned on the generated train with real pose errors.

One parameter set per fold: the generated TRAIN (seed 1, as tune_noisy_bases.py) with one archive tract per window
from the two archive scenes other than the held-out one (uniform tract, uniform yaw, predicted-class scale), wide
grids (RETUNE_GRIDS). Used inside R-F (training pools) and as F / Kalman CA of the fold.

Usage: python scripts/forecast/archive/d2b_bases.py [--jobs 3] [--out data/processed/forecast_archive/d2b]
"""
import argparse
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
from dronepose.forecast.windows import freeze_windows

TUNE_SALT = 7


def _job(fold, data, config):
    t0 = time.time()
    cfg = with_grids(load_config(Path(config)), RETUNE_GRIDS)
    train = freeze_windows(load_trajectories(Path(data), "train"), cfg.obs_len, cfg.pred_len, cfg.fps)
    scenes = fold_train(fold)
    tracts = load_tracts(list(ARCHIVE_PATHS)).select(scenes, meters=True)
    assert set(tracts.scene) == set(scenes), (fold, set(tracts.scene))
    wn = noisy_windows(train, tracts, archive_noise_rng(1, TUNE_SALT, ARCHIVE_SCENES.index(fold)), distance_cond=False)
    params = tune(wn, cfg)
    return fold, {**chosen_params(params), "scores": params["scores"], "edge_warnings": params["edge_warnings"],
                  "split": "train generato (seme 1) con errori dell'archivio", "held_out": fold,
                  "tract_scenes": list(scenes), "n_tracts": len(tracts), "n_windows": len(train),
                  "lost_frame_fraction": float(1 - np.mean(wn.obs_valid)),
                  "grids": {k: list(v) for k, v in RETUNE_GRIDS.items()}, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/forecast_gen")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--out", default="data/processed/forecast_archive/d2b")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as ex:
        for fold, res in ex.map(_job, ARCHIVE_SCENES, [args.data] * 3, [args.config] * 3):
            (out / f"bases_fold{fold}.json").write_text(json.dumps(res, indent=1))
            print(f"fold {fold} (tratti {res['tract_scenes']}, {res['n_tracts']}): {res['seconds']:.0f} s; "
                  f"acro_gen {res['acro_gen']}, kf_ca {res['kf_ca']}, cv_k {res['cv_k']}; "
                  f"ADE 1 s train: acro gen {res['scores']['physics_acro_gen']:.3f}, kf_ca {res['scores']['kf_ca']:.3f}; "
                  f"avvisi {res['edge_warnings'] or 'nessuno'}", flush=True)


if __name__ == "__main__":
    main()
