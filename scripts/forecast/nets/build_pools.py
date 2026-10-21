"""Training pools of the nets (plan step 3.3, Task 5): features and base predictions, computed once.

clean_{train_big,select}.npz: K = 1, clean poses, base parameters tuned on the clean generated train;
noisy_{train_big,select}.npz: K copies (configs/nets.json), noise mixture per window sigma_p ~ U(0, 5) cm,
sigma_r ~ U(0, 3) deg (Gaussian, NOT the real PnP error), bases on the noisy poses with the mixed-noise parameters.
Noise stream of copy k: noise_rng(set seed, 0, salt=k) (train_big 6, select 5).

Usage: python scripts/forecast/nets/build_pools.py [--jobs 4] [--out data/processed/forecast_nets]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.nets.device import load_nets_config
from dronepose.forecast.nets.features import build_batch, noisy_copy, save_pool
from dronepose.forecast.windows import freeze_windows

SETS = ("train_big", "select")
_STATE = {}


def _init(data, config, nets_config, clean_params, noisy_params):
    cfg = load_config(Path(config))
    _STATE.update(cfg=cfg, ncfg=load_nets_config(Path(nets_config)),
                  clean=json.loads(Path(clean_params).read_text()), noisy=json.loads(Path(noisy_params).read_text()),
                  windows={s: freeze_windows(load_trajectories(Path(data), s), cfg.obs_len, cfg.pred_len, cfg.fps)
                           for s in SETS})


def _job(name, regime, k):
    """One copy of one pool (k = -1: the clean copy)."""
    t0 = time.time()
    s, cfg, ncfg = _STATE, _STATE["cfg"], _STATE["ncfg"]
    w = s["windows"][name]
    if regime == "clean":
        b = build_batch(w, s["clean"], cfg)
    else:
        b = noisy_copy(w, s["noisy"], cfg, ncfg["set_seeds"][name], k, ncfg["noise_max_p_cm"], ncfg["noise_max_r_deg"])
    return name, regime, k, b, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/forecast_gen")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--nets-config", default="configs/nets.json")
    ap.add_argument("--clean-params", default="experiments/forecast/generator/tuning.json")
    ap.add_argument("--noisy-params", default="experiments/forecast/nets/base_params_noisy.json")
    ap.add_argument("--out", default="data/processed/forecast_nets")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    ncfg = load_nets_config(Path(args.nets_config))
    t0 = time.time()
    jobs = [(s, "clean", 0) for s in SETS] + [(s, "noisy", k) for s in SETS for k in range(ncfg["K"])]
    done = {}
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(args.data, args.config, args.nets_config,
                                                                      args.clean_params, args.noisy_params)) as ex:
        for name, regime, k, b, sec in ex.map(_job, *zip(*jobs)):
            done[(name, regime, k)] = b
            print(f"{regime}_{name} copia {k}: {len(b)} finestre in {sec:.0f} s (totale {time.time() - t0:.0f} s)",
                  flush=True)
    out = Path(args.out)
    for name in SETS:
        save_pool([done[(name, "clean", 0)]], out / f"clean_{name}.npz")
        save_pool([done[(name, "noisy", k)] for k in range(ncfg["K"])], out / f"noisy_{name}.npz")
    (out / "pools.json").write_text(json.dumps({
        "K": ncfg["K"], "set_seeds": ncfg["set_seeds"], "clean_params": args.clean_params,
        "noisy_params": args.noisy_params, "noise": "sigma_p ~ U(0, %g) cm, sigma_r ~ U(0, %g) deg per window, "
        "gaussian (not the real PnP error), noise_rng(set seed, 0, salt=k)" % (ncfg["noise_max_p_cm"],
                                                                            ncfg["noise_max_r_deg"]),
        "seconds": time.time() - t0}, indent=1))
    print(f"scritti 4 pool in {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
