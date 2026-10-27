"""D2b (spec 2026-10-27 §5.3): training pools with real pose errors, one per fold.

pool_train_fold{s}.npz: K copies (configs/nets.json) of train_big, pool_select_fold{s}.npz: 1 copy of select; tracts
only from the archive scenes other than s (asserted), uniform tract and yaw, mask channel, bases on the noisy poses
with bases_fold{s}.json. Stream of copy k: archive_noise_rng(set seed, k) (train_big 6, select 5).

Usage: python scripts/forecast/archive/d2b_pools.py [--jobs 4] [--out data/processed/forecast_archive/d2b]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dronepose.forecast.archive_noise import ARCHIVE_PATHS, ARCHIVE_SCENES, fold_train, load_tracts
from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.nets.device import load_nets_config
from dronepose.forecast.nets.features import archive_copy, save_pool
from dronepose.forecast.windows import freeze_windows

SETS = ("train_big", "select")
_STATE = {}


def _init(data, config, nets_config, out):
    cfg = load_config(Path(config))
    tracts = load_tracts(list(ARCHIVE_PATHS))
    _STATE.update(cfg=cfg, ncfg=load_nets_config(Path(nets_config)),
                  tracts={f: tracts.select(fold_train(f), meters=True) for f in ARCHIVE_SCENES},
                  params={f: json.loads((Path(out) / f"bases_fold{f}.json").read_text()) for f in ARCHIVE_SCENES},
                  windows={s: freeze_windows(load_trajectories(Path(data), s), cfg.obs_len, cfg.pred_len, cfg.fps)
                           for s in SETS})


def _job(fold, name, k):
    t0 = time.time()
    s = _STATE
    tr = s["tracts"][fold]
    assert set(tr.scene) == set(fold_train(fold)) and fold not in set(tr.scene)
    b = archive_copy(s["windows"][name], s["params"][fold], s["cfg"], tr, s["ncfg"]["set_seeds"][name], k,
                     distance_cond=False)
    return fold, name, k, b, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/forecast_gen")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--nets-config", default="configs/nets.json")
    ap.add_argument("--out", default="data/processed/forecast_archive/d2b")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    K = load_nets_config(Path(args.nets_config))["K"]
    t0 = time.time()
    jobs = [(f, "train_big", k) for f in ARCHIVE_SCENES for k in range(K)] + [(f, "select", 0) for f in ARCHIVE_SCENES]
    done = {}
    with ProcessPoolExecutor(args.jobs, initializer=_init,
                             initargs=(args.data, args.config, args.nets_config, args.out)) as ex:
        for fold, name, k, b, sec in ex.map(_job, *zip(*jobs)):
            done[(fold, name, k)] = b
            print(f"fold {fold} {name} copia {k}: {len(b)} finestre in {sec:.0f} s (totale {time.time() - t0:.0f} s)",
                  flush=True)
    out = Path(args.out)
    for f in ARCHIVE_SCENES:
        save_pool([done[(f, "train_big", k)] for k in range(K)], out / f"pool_train_fold{f}.npz")
        save_pool([done[(f, "select", 0)]], out / f"pool_select_fold{f}.npz")
    (out / "pools.json").write_text(json.dumps({"K": K, "folds": {f: list(fold_train(f)) for f in ARCHIVE_SCENES},
                                                "noise": "archive tracts, uniform, uniform yaw, predicted-class scale",
                                                "seconds": time.time() - t0}, indent=1))
    print(f"pool scritti in {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
