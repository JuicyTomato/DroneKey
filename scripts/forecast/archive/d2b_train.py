"""D2b (spec 2026-10-27 §5.3): R-F trained with real pose errors, 3 folds x 3 seeds, resumable.

Runs: experiments/forecast/archive_d2b/runs/R-F_noisy_gru_fold{s}_s{seed}/ (10 input channels: mask included).

Usage: python scripts/forecast/archive/d2b_train.py [--folds 03 09 14] [--seeds 0 1 2] [--device auto]
"""
import argparse
import time
from pathlib import Path

from dronepose.forecast.archive_noise import ARCHIVE_SCENES
from dronepose.forecast.nets.device import load_nets_config, pick_device
from dronepose.forecast.nets.features import load_pool
from dronepose.forecast.nets.train import needed_bases, run_name, train_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=list(ARCHIVE_SCENES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pools", default="data/processed/forecast_archive/d2b")
    ap.add_argument("--nets-config", default="configs/nets.json")
    ap.add_argument("--out", default="experiments/forecast/archive_d2b/runs")
    args = ap.parse_args()
    cfg = load_nets_config(Path(args.nets_config))
    device = pick_device(args.device)
    names = needed_bases("R-F", False)
    for fold in args.folds:
        t0 = time.time()
        train = load_pool(Path(args.pools) / f"pool_train_fold{fold}.npz", bases=names)
        select = load_pool(Path(args.pools) / f"pool_select_fold{fold}.npz", bases=names)[:1]
        print(f"fold {fold}: pool caricati in {time.time() - t0:.0f} s", flush=True)
        for seed in args.seeds:
            name = run_name("R-F", "noisy", seed, tag=f"fold{fold}")
            res = train_run("R-F", "noisy", seed, cfg, train, select, Path(args.out) / name, device=device, n_in=10)
            print(f"{name}: {res['epochs']} epoche, migliore {res['best_epoch']} (ADE 1 s selezione "
                  f"{res['best_select_ade_1s']:.4f} m), {res['seconds_this_call']:.0f} s, finito {res['done']}",
                  flush=True)


if __name__ == "__main__":
    main()
