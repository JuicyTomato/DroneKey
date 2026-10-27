"""E4a / E4b on the 6DroneSyn VAL (spec 2026-10-27 §5.5-§6): tables, selection of the method, oracle -> pipeline.

E4a: GT poses, baselines tuned on the clean train (experiments/forecast/tuning/tuning.json), nets of the set 'clean'.
E4b: estimated poses (classe causale; row with the true class apart), for scene s every method of fold s (parameters
tuned and nets trained with archive tracts of the OTHER two scenes only), 3 pose seeds x 3 copies. Drone 08 in
dataset units with its own parameters, no nets, outside the selection. The differences E4b - E4a are paired on the
same windows (E4a errors taken at the GT window of every E4b row). Never touches a test split.

Usage: python scripts/forecast/e4/e4_val.py [--device auto] [--bootstrap-b 10000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.e4_eval import (BASELINES, REFERENCE, degradation, method_errors, pipeline_errors, select_on,
                                        summarize_e4)
from dronepose.forecast.nets.device import pick_device
from dronepose.forecast.windows import make_windows

sys.path.insert(0, str(Path(__file__).parent))
from e4_train import ARMS, e4_run_name  # noqa: E402

SEEDS = (0, 1, 2)


def _nets(runs: Path, set_name: str) -> dict:
    return {arm: [runs / e4_run_name(arm, set_name, s) for s in SEEDS] for arm in ARMS}


def _md_group(summary: dict, title: str, methods) -> list[str]:
    lines = [f"### {title}", "", "| metodo | " + " | ".join(g["group"] for g in summary["groups"][:3]) + " |",
             "|---|" + "---|" * 3]
    for m in methods:
        cells = []
        for g in summary["groups"][:3]:
            a = g["position"][m]["ade"][-1]
            s = g["skill_vs_CV-2"].get(m, {}).get("ade_1s")
            cells.append(f"{a:.3f}" + (f" ({100 * s[0]:+.1f}% [{100 * s[1]:+.1f}, {100 * s[2]:+.1f}])"
                                       if s and len(s) == 3 else ""))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/forecast")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--clean-params", default="experiments/forecast/tuning/tuning.json")
    ap.add_argument("--params", default="data/processed/forecast_archive/e4")
    ap.add_argument("--poses", default="experiments/poses")
    ap.add_argument("--runs", default="experiments/forecast/e4/runs")
    ap.add_argument("--out", default="experiments/forecast/e4/val")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bootstrap-b", type=int, default=10000)
    args = ap.parse_args()
    t0 = time.time()
    cfg, device, b = load_config(Path(args.config)), pick_device(args.device), args.bootstrap_b
    runs, pdir = Path(args.runs), Path(args.params)
    trajs = load_trajectories(Path(args.data), "val")
    gt = make_windows(trajs, cfg.obs_len, cfg.pred_len, cfg.fps)
    clean = json.loads(Path(args.clean_params).read_text())
    err_a, rot_a = method_errors(gt, clean, cfg, _nets(runs, "clean"), device)
    cands_a = BASELINES + tuple(ARMS)
    res = {"E4a": {"summary": summarize_e4(err_a, rot_a, gt, cfg, b), "selection": select_on(err_a, gt, cands_a, b)}}
    print(f"E4a: scelto {res['E4a']['selection']['chosen']} (migliore {res['E4a']['selection']['best']}, "
          f"pareggi {res['E4a']['selection']['ties']}) in {time.time() - t0:.0f} s", flush=True)

    fold = {s: json.loads((pdir / f"params_fold{s}.json").read_text()) for s in ("03", "09", "14")}
    fold08 = {s: json.loads((pdir / f"params_08_fold{s}.json").read_text()) for s in ("03", "09", "14")}
    params_for = lambda scene, metric: fold[scene] if metric else fold08[scene]
    nets_for = lambda scene: _nets(runs, f"fold{scene}")
    cands_b = cands_a + ("kf_ca_los",)
    for cls in ("causal", "gt"):
        paths = [Path(args.poses) / f"poses_val_s{k}_{cls}.jsonl" for k in SEEDS]
        err_b, rot_b, idx, info = pipeline_errors(trajs, paths, cfg, params_for, nets_for, device)
        wb = gt.subset(idx)
        err_a_same = {k: v[idx] for k, v in err_a.items()}
        key = f"E4b_{cls}"
        res[key] = {"summary": summarize_e4(err_b, rot_b, wb, cfg, b), "counts": info,
                    "degradation_vs_E4a": degradation(err_b, err_a_same, wb, b)}
        if cls == "causal":
            res[key]["selection"] = select_on(err_b, wb, cands_b, b)
            sel = res[key]["selection"]
            print(f"E4b: scelto {sel['chosen']} (migliore {sel['best']}, pareggi {sel['ties']}); {info} "
                  f"in {time.time() - t0:.0f} s", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "selection.json").write_text(json.dumps({"E4a": res["E4a"]["selection"],
                                                    "E4b": res["E4b_causal"]["selection"]}, indent=1))
    lines = ["# E4a / E4b sulla val del 6DroneSyn", "",
             "ADE a 1 s (m) e skill contro CV-2 (IC 95%, bootstrap per traiettoria, strati = scena).", ""]
    lines += _md_group(res["E4a"]["summary"], "E4a — pose vere", (REFERENCE,) + cands_a)
    lines += _md_group(res["E4b_causal"]["summary"], "E4b — pose stimate, classe causale", (REFERENCE,) + cands_b)
    lines += _md_group(res["E4b_gt"]["summary"], "E4b — pose stimate, classe vera", (REFERENCE,) + cands_b)
    for k in ("E4a", "E4b_causal"):
        s = res[k]["selection"]
        lines.append(f"- **Selezione {k}:** {s['chosen']} (migliore {s['best']}; pareggi con il migliore: "
                     f"{', '.join(s['ties']) or 'nessuno'})")
    lines += ["", "Degrado oracle → pipeline (classe causale), ADE 1 s E4b − E4a sulle stesse finestre:", ""]
    lines += [f"- {m}: {d['ade_1s_e4a']:.3f} → {d['ade_1s_e4b']:.3f} m ({d['mean']:+.3f} [{d['ci'][0]:+.3f}, "
              f"{d['ci'][1]:+.3f}])" for m, d in res["E4b_causal"]["degradation_vs_E4a"].items()]
    lines += ["", f"Conteggi E4b (classe causale): {res['E4b_causal']['counts']}"]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
