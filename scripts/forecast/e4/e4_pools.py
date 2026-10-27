"""E4a / E4b (spec 2026-10-27 §5.4): training pools of the 6DroneSyn nets.

Train windows in metres without scenes 02 / 05 (training) and with them (early stopping, never the val).
clean: K = 1, ground-truth poses (E4a), clean train parameters; fold03 / fold09 / fold14 / final: K copies with
real pose errors from the allowed archive scenes (distance band, line of sight, predicted-class scale), bases on the
noisy poses with the parameters of the same set (e4_tune.py). Every pool has the mask channel (10 inputs); the base
of R-CV is CV-2 (cv_k = 2) in every set. Streams: archive_noise_rng(11, k) train, archive_noise_rng(12, 0) select.

Usage: python scripts/forecast/e4/e4_pools.py [--jobs 4] [--out data/processed/forecast_archive/e4]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dronepose.forecast.archive_noise import ARCHIVE_PATHS, fold_train, load_tracts
from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.e4 import split_nets_scenes
from dronepose.forecast.nets.device import load_nets_config
from dronepose.forecast.nets.features import archive_copy, build_batch, save_pool
from dronepose.forecast.windows import make_windows

SETS = {"fold03": "03", "fold09": "09", "fold14": "14", "final": None}
TRAIN_SEED, SELECT_SEED = 11, 12
_STATE = {}


def _init(data, config, clean_params, params_dir):
    cfg = load_config(Path(config))
    w = make_windows(load_trajectories(Path(data), "train"), cfg.obs_len, cfg.pred_len, cfg.fps)
    tracts = load_tracts(list(ARCHIVE_PATHS))
    params = {"clean": json.loads(Path(clean_params).read_text())}
    params.update({s: json.loads((Path(params_dir) / f"params_{s}.json").read_text()) for s in SETS})
    _STATE.update(cfg=cfg, windows=dict(zip(("train", "select"), split_nets_scenes(w))),
                  tracts={s: tracts.select(fold_train(h), meters=True) for s, h in SETS.items()},
                  params={k: {**v, "cv_k": 2} for k, v in params.items()})


def _job(regime, part, k):
    t0 = time.time()
    s = _STATE
    w, p = s["windows"][part], s["params"][regime]
    if regime == "clean":
        b = build_batch(w, p, s["cfg"], mask_channel=True)
    else:
        tr = s["tracts"][regime]
        assert set(tr.scene) == set(fold_train(SETS[regime]))
        b = archive_copy(w, p, s["cfg"], tr, TRAIN_SEED if part == "train" else SELECT_SEED, k, distance_cond=True)
    return regime, part, k, b, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/forecast")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--nets-config", default="configs/nets.json")
    ap.add_argument("--clean-params", default="experiments/forecast/tuning/tuning.json")
    ap.add_argument("--out", default="data/processed/forecast_archive/e4")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    K = load_nets_config(Path(args.nets_config))["K"]
    t0 = time.time()
    jobs = [("clean", "train", 0), ("clean", "select", 0)]
    jobs += [(s, "train", k) for s in SETS for k in range(K)] + [(s, "select", 0) for s in SETS]
    done = {}
    with ProcessPoolExecutor(args.jobs, initializer=_init,
                             initargs=(args.data, args.config, args.clean_params, args.out)) as ex:
        for regime, part, k, b, sec in ex.map(_job, *zip(*jobs)):
            done[(regime, part, k)] = b
            print(f"{regime} {part} copia {k}: {len(b)} finestre in {sec:.0f} s (totale {time.time() - t0:.0f} s)",
                  flush=True)
    out = Path(args.out)
    for regime in ("clean", *SETS):
        n = 1 if regime == "clean" else K
        save_pool([done[(regime, "train", k)] for k in range(n)], out / f"pool_train_{regime}.npz")
        save_pool([done[(regime, "select", 0)]], out / f"pool_select_{regime}.npz")
    (out / "pools.json").write_text(json.dumps({"K": K, "sets": {s: list(fold_train(h)) for s, h in SETS.items()},
                                                "select_scenes": ["02", "05"], "seconds": time.time() - t0}, indent=1))
    print(f"pool scritti in {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
