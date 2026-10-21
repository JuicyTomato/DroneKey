"""Base-predictor parameters for the noisy training regime of the nets (spec §5.3, plan step 3.3, Task 4).

The real system does not know the noise level, so inside R-F / R-CV / R-F-P one parameter set is used, tuned on the
generated TRAIN (seed 1) with a noise mixture per window: sigma_p ~ U(0, 5) cm, sigma_r ~ U(0, 3) deg (Gaussian,
NOT the real PnP error). Same objective and wide grids as noise_sensitivity.py --retune (RETUNE_GRIDS).

Usage: python scripts/forecast/nets/tune_noisy_bases.py [--out experiments/forecast/nets/base_params_noisy.json]
"""
import argparse
import json
import time
from pathlib import Path

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.nets.device import load_nets_config
from dronepose.forecast.noise import (RETUNE_GRIDS, add_pose_noise_per_window, chosen_params, noise_rng,
                                      sample_mixture_sigmas, with_grids)
from dronepose.forecast.tuning import tune
from dronepose.forecast.windows import freeze_windows

NOISE_SEED, NOISE_SALT = 0, 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/forecast_gen")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--nets-config", default="configs/nets.json")
    ap.add_argument("--clean-params", default="experiments/forecast/generator/tuning.json")
    ap.add_argument("--out", default="experiments/forecast/nets/base_params_noisy.json")
    args = ap.parse_args()

    ncfg = load_nets_config(Path(args.nets_config))
    cfg = with_grids(load_config(Path(args.config)), RETUNE_GRIDS)
    t0 = time.time()
    train = freeze_windows(load_trajectories(Path(args.data), "train"), cfg.obs_len, cfg.pred_len, cfg.fps)
    rng = noise_rng(NOISE_SEED, 0, salt=NOISE_SALT)
    sp, sr = sample_mixture_sigmas(len(train), rng, ncfg["noise_max_p_cm"], ncfg["noise_max_r_deg"])
    params = tune(add_pose_noise_per_window(train, sp, sr, rng), cfg)
    out = {**chosen_params(params), "scores": params["scores"], "edge_warnings": params["edge_warnings"],
           "split": "train (generato, seme 1) con rumore misto per finestra",
           "noise": {"sigma_p_cm": f"U(0, {ncfg['noise_max_p_cm']})", "sigma_r_deg": f"U(0, {ncfg['noise_max_r_deg']})",
                     "rng": f"noise_rng({NOISE_SEED}, 0, salt={NOISE_SALT})", "gaussian_not_real": True},
           "grids": {k: list(v) for k, v in RETUNE_GRIDS.items()}, "n_windows": len(train),
           "seconds": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    clean = json.loads(Path(args.clean_params).read_text())
    print(f"taratura con rumore misto su {len(train)} finestre in {out['seconds']:.0f} s -> {args.out}")
    for k in ("cv_k", "ca_k", "kf_ca", "ctr_k", "acro_gen", "acro_mode"):
        print(f"  {k}: pulito {clean.get(k)} | rumore misto {out.get(k)}")
    print("  ADE 1 s sul train rumoroso (pesata):", {m: round(s, 4) for m, s in out["scores"].items()})
    print("  avvisi di bordo:", out["edge_warnings"] or "nessuno")


if __name__ == "__main__":
    main()
