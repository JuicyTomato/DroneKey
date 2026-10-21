"""Train one run of the nets (plan step 3.3, Task 7); resumes from last.pt when the run directory has one.

Usage: python scripts/forecast/nets/train_nets.py --arm R-F --regime clean --seed 0 [--rnn gru] [--use-mode]
           [--n-train N] [--device auto] [--max-epochs E]
Output: experiments/forecast/nets/runs/{arm}_{regime}_{rnn}{_mode}{_nN}_s{seed}/ (history.json, best.pt, last.pt).
Pools: data/processed/forecast_nets/{regime}_{train_big,select}.npz (scripts/forecast/nets/build_pools.py).
"""
import argparse
import json
import time
from pathlib import Path

from dronepose.forecast.nets.device import load_nets_config, pick_device
from dronepose.forecast.nets.features import load_pool
from dronepose.forecast.nets.models import ARMS
from dronepose.forecast.nets.train import needed_bases, run_name, train_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--regime", default="clean", choices=("clean", "noisy"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rnn", default="gru", choices=("gru", "lstm"))
    ap.add_argument("--use-mode", action="store_true")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-epochs", type=int, default=None, help="override (trial runs only)")
    ap.add_argument("--pools", default="data/processed/forecast_nets")
    ap.add_argument("--nets-config", default="configs/nets.json")
    ap.add_argument("--out", default="experiments/forecast/nets/runs")
    args = ap.parse_args()

    cfg = load_nets_config(Path(args.nets_config))
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    name = run_name(args.arm, args.regime, args.seed, args.rnn, args.use_mode, args.n_train)
    t0 = time.time()
    names = needed_bases(args.arm, args.use_mode)
    train = load_pool(Path(args.pools) / f"{args.regime}_train_big.npz", bases=names)
    select = load_pool(Path(args.pools) / f"{args.regime}_select.npz", bases=names)[:1]
    print(f"{name}: pool caricati in {time.time() - t0:.0f} s ({len(train)} copie x {len(train[0])} campioni)",
          flush=True)
    res = train_run(args.arm, args.regime, args.seed, cfg, train, select, Path(args.out) / name, rnn=args.rnn,
                    use_mode=args.use_mode, n_train=args.n_train, device=pick_device(args.device))
    print(f"{name}: {res['epochs']} epoche, migliore {res['best_epoch']} (ADE 1 s selezione {res['best_select_ade_1s']:.4f} m),"
          f" {res['seconds_this_call']:.0f} s su {res['device']}, finito: {res['done']}", flush=True)
    hist = json.loads((Path(args.out) / name / "history.json").read_text())["history"][1:]
    if hist:
        print(f"{name}: {sum(h['seconds'] for h in hist) / len(hist):.1f} s per epoca", flush=True)


if __name__ == "__main__":
    main()
