"""D2b (spec 2026-10-27 §5.3, §6): criterion on the generated val (level 1, weighted) with real pose errors.

Fold s: val windows with tracts ONLY from archive scene s (uniform tract and yaw, 3 draws archive_noise_rng(2, i_s, d)),
R-F of fold s (3 seeds) against F and Kalman CA with the parameters of fold s (bases of the same noisy batch).
Diagnostic, not a criterion: the R-F trained with Gaussian noise (3.3, 9 channels, lost frames filled, no mask) on the
same noisy windows, with its own base parameters (base_params_noisy.json). Never touches a test split.

Usage: python scripts/forecast/archive/d2b_evaluate.py [--device auto] [--bootstrap-b 10000]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from dronepose.forecast.archive_noise import (ARCHIVE_PATHS, ARCHIVE_SCENES, archive_noise_rng, load_tracts,
                                              noisy_windows)
from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.metrics import bootstrap_counts
from dronepose.forecast.nets.device import pick_device
from dronepose.forecast.nets.evaluate import contrast, criterion_d2b, predict_run, window_errors
from dronepose.forecast.nets.features import build_batch
from dronepose.forecast.nets.train import run_name
from dronepose.forecast.windows import freeze_windows

SEEDS, DRAWS, VAL_SEED = (0, 1, 2), 3, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/forecast_gen")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--bases", default="data/processed/forecast_archive/d2b")
    ap.add_argument("--runs", default="experiments/forecast/archive_d2b/runs")
    ap.add_argument("--gauss-runs", default="experiments/forecast/nets/runs")
    ap.add_argument("--gauss-params", default="experiments/forecast/nets/base_params_noisy.json")
    ap.add_argument("--out", default="experiments/forecast/archive_d2b")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bootstrap-b", type=int, default=10000)
    args = ap.parse_args()
    t0 = time.time()
    cfg, device = load_config(Path(args.config)), pick_device(args.device)
    val = freeze_windows(load_trajectories(Path(args.data), "val"), cfg.obs_len, cfg.pred_len, cfg.fps)
    wt, traj, strata = np.asarray(val.meta["weight"], float), val.traj, val.meta["mode"]
    tracts = load_tracts(list(ARCHIVE_PATHS))
    gauss_params = json.loads(Path(args.gauss_params).read_text())
    names = ("R-F archivio", "F archivio", "Kalman CA archivio", "R-F gaussiano (diagnostica)",
             "F parametri gaussiani")
    errs = {n: {} for n in names}
    rf_seeds = {}
    info = {}
    for i, fold in enumerate(ARCHIVE_SCENES):
        params = json.loads((Path(args.bases) / f"bases_fold{fold}.json").read_text())
        tr = tracts.select((fold,), meters=True)
        assert set(tr.scene) == {fold}
        acc = {n: [] for n in names}
        per_seed = {s: [] for s in SEEDS}
        lost = []
        for d in range(DRAWS):
            wn = noisy_windows(val, tr, archive_noise_rng(VAL_SEED, i, d), distance_cond=False)
            lost.append(float(1 - np.mean(wn.obs_valid)))
            b = build_batch(wn, params, cfg, mask_channel=True)
            for s in SEEDS:
                run = Path(args.runs) / run_name("R-F", "noisy", s, tag=f"fold{fold}")
                per_seed[s].append(window_errors(predict_run(run, b, device), b.target))
            acc["F archivio"].append(window_errors(b.bases["phys"], b.target))
            acc["Kalman CA archivio"].append(window_errors(b.bases["kf_ca"], b.target))
            bg = build_batch(wn, gauss_params, cfg)
            acc["R-F gaussiano (diagnostica)"].append(np.mean(
                [window_errors(predict_run(Path(args.gauss_runs) / run_name("R-F", "noisy", s), bg, device), bg.target)
                 for s in SEEDS], 0))
            acc["F parametri gaussiani"].append(window_errors(bg.bases["phys"], bg.target))
        rf_seeds[fold] = [np.mean(per_seed[s], 0) for s in SEEDS]
        errs["R-F archivio"][fold] = np.mean(rf_seeds[fold], 0)
        for n in names[1:]:
            errs[n][fold] = np.mean(acc[n], 0)
        info[fold] = {"n_tracts": len(tr), "lost_frame_fraction": float(np.mean(lost)), "params": {
            k: params[k] for k in ("cv_k", "kf_ca", "acro_gen")}}
        print(f"fold {fold}: " + ", ".join(f"{n} {np.average(errs[n][fold], weights=wt):.3f}" for n in names)
              + f" ({time.time() - t0:.0f} s)", flush=True)
    d2b = criterion_d2b(rf_seeds, errs["F archivio"], wt, traj, strata, args.bootstrap_b, cfg.bootstrap_seed)
    counts = bootstrap_counts(traj, strata, args.bootstrap_b, cfg.bootstrap_seed)
    total = {n: np.mean([errs[n][f] for f in ARCHIVE_SCENES], 0) for n in names}
    table = {n: {**{f: float(np.average(errs[n][f], weights=wt)) for f in ARCHIVE_SCENES},
                 "totale": float(np.average(total[n], weights=wt))} for n in names}
    extra = {f"{a} vs {r}": contrast([total[a]], total[r], wt, traj, strata, args.bootstrap_b, 0, counts)
             for a, r in (("R-F archivio", "Kalman CA archivio"), ("F archivio", "Kalman CA archivio"),
                          ("R-F gaussiano (diagnostica)", "F archivio"), ("R-F archivio", "R-F gaussiano (diagnostica)"))}
    out = Path(args.out)
    res = {"D2b": d2b, "ade_1s": table, "contrasts": extra, "folds": info,
           "n_rescaled_archive_frames": tracts.n_rescaled, "n_lost_pred08": tracts.n_lost_pred08,
           "seconds": time.time() - t0}
    (out / "criteria.json").write_text(json.dumps(res, indent=1))
    pct = lambda c: f"{100 * c['value']:+.1f}% [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}]"
    lines = ["# D2b — rumore dall'archivio degli errori reali (val generata L1, pesata)", "",
             f"**D2b: {'passa' if d2b['pass'] else 'non passa'}** — totale {pct(d2b)}, semi "
             + " / ".join(f"{100 * v:+.1f}%" for v in d2b["per_seed"]) + "; per fold (scena tenuta fuori): "
             + "; ".join(f"{f} {pct(c)}" for f, c in d2b["folds"].items()), "",
             "| ADE 1 s (m) | " + " | ".join(ARCHIVE_SCENES) + " | totale |", "|---|" + "---|" * 4]
    lines += [f"| {n} | " + " | ".join(f"{table[n][f]:.3f}" for f in ARCHIVE_SCENES) + f" | {table[n]['totale']:.3f} |"
              for n in names]
    lines += ["", "Altri contrasti (skill ADE 1 s, totale):"] + [f"- {k}: {pct(c)}" for k, c in extra.items()]
    lines += ["", "Frame persi per fold: "
              + ", ".join(f"{f} {100 * v['lost_frame_fraction']:.2f}%" for f, v in info.items())]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
