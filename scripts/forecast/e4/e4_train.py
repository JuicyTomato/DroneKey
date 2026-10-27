"""E4a / E4b (spec 2026-10-27 §5.4): the 6DroneSyn nets, 3 arms x 3 seeds x 5 sets = 45 runs, resumable.

G-6D (GRU alone) and R-CV-6D (residual on CV-2) from scratch; R-CV-FT = the clean generated R-CV of the same seed
(experiments/forecast/nets/runs/R-CV_clean_gru_s{seed}) fine-tuned with lr 1e-4 (mask-channel weights start at 0).
Sets: clean (E4a), fold03 / fold09 / fold14 (E4b val), final (E4b test). Runs in experiments/forecast/e4/runs/.

Usage: python scripts/forecast/e4/e4_train.py [--sets clean fold03 ...] [--arms G-6D ...] [--seeds 0 1 2]
"""
import argparse
import time
from pathlib import Path

from dronepose.forecast.nets.device import load_nets_config, pick_device
from dronepose.forecast.nets.features import load_pool
from dronepose.forecast.nets.train import needed_bases, run_name, train_run

SETS = ("clean", "fold03", "fold09", "fold14", "final")
ARMS = {"G-6D": ("G", False), "R-CV-6D": ("R-CV", False), "R-CV-FT": ("R-CV", True)}
FT_LR = 1e-4


def e4_run_name(arm: str, set_name: str, seed: int) -> str:
    base, ft = ARMS[arm]
    return run_name(base, "clean" if set_name == "clean" else "noisy", seed, tag=f"{'ft' if ft else '6d'}_{set_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pools", default="data/processed/forecast_archive/e4")
    ap.add_argument("--source-runs", default="experiments/forecast/nets/runs")
    ap.add_argument("--nets-config", default="configs/nets.json")
    ap.add_argument("--out", default="experiments/forecast/e4/runs")
    args = ap.parse_args()
    cfg = load_nets_config(Path(args.nets_config))
    device = pick_device(args.device)
    for set_name in args.sets:
        pools = {}
        for arm in args.arms:
            base, ft = ARMS[arm]
            names = needed_bases(base, False)
            if names not in pools:
                pools[names] = (load_pool(Path(args.pools) / f"pool_train_{set_name}.npz", bases=names),
                                load_pool(Path(args.pools) / f"pool_select_{set_name}.npz", bases=names)[:1])
            train, select = pools[names]
            regime = "clean" if set_name == "clean" else "noisy"
            for seed in args.seeds:
                name = e4_run_name(arm, set_name, seed)
                t0 = time.time()
                init = Path(args.source_runs) / run_name("R-CV", "clean", seed) / "best.pt" if ft else None
                res = train_run(base, regime, seed, cfg, train, select, Path(args.out) / name, device=device,
                                init_from=init, lr=FT_LR if ft else None, n_in=10)
                print(f"{name}: {res['epochs']} epoche, migliore {res['best_epoch']} (ADE 1 s selezione "
                      f"{res['best_select_ade_1s']:.4f} m), {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
