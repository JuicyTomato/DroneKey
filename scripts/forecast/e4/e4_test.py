"""E4a / E4b on the 6DroneSyn TEST, once (spec 2026-10-27 §5.6, criteria T1-T3 of §6). No new choice is made here.

Refuses without --allow-test, and refuses if the output directory already exists (the test is used once). The methods
are the ones chosen on the val (experiments/forecast/e4/val/selection.json). E4a: GT poses, parameters of the clean
train, nets 'clean'. E4b: estimated test poses (classe causale; classe vera in a separate row), parameters and nets
'final' (archive tracts of all three val scenes; the test scenes are not in the archive). Drone 08 in a separate row.
T1 = E4a chosen vs CV-2, T2 = E4b chosen vs CV-2 on the same estimated poses, T3 = E4b - E4a of the E4b method on the
same windows (skill ADE at 1 s / difference, 95% interval, bootstrap per trajectory, strata = scene).

Usage: python scripts/forecast/e4/e4_test.py --allow-test   (only after the user's explicit go-ahead)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from dronepose.forecast.e4_eval import guard_test_run

sys.path.insert(0, str(Path(__file__).parent))

SEEDS = (0, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--smoke-val", action="store_true",
                    help="run the same code on the VAL split (code check before the single test run; not a result)")
    ap.add_argument("--data", default="data/processed/forecast")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--clean-params", default="experiments/forecast/tuning/tuning.json")
    ap.add_argument("--params", default="data/processed/forecast_archive/e4")
    ap.add_argument("--selection", default="experiments/forecast/e4/val/selection.json")
    ap.add_argument("--poses", default="experiments/poses")
    ap.add_argument("--runs", default="experiments/forecast/e4/runs")
    ap.add_argument("--out", default="experiments/forecast/e4/test")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bootstrap-b", type=int, default=10000)
    args = ap.parse_args()
    split = "val" if args.smoke_val else "test"
    try:
        guard_test_run(args.allow_test, Path(args.out), smoke=args.smoke_val)   # before reading anything of the test
    except ValueError as e:
        sys.exit(f"e4_test.py: {e}")

    from dronepose.forecast.config import load_config
    from dronepose.forecast.dataset import load_trajectories
    from dronepose.forecast.e4_eval import (BASELINES, REFERENCE, degradation, method_errors, pipeline_errors,
                                            summarize_e4, t3_e4a_key, validate_choices)
    from dronepose.forecast.nets.device import pick_device
    from dronepose.forecast.windows import make_windows
    from e4_train import ARMS, e4_run_name

    t0 = time.time()
    cfg, device, b = load_config(Path(args.config)), pick_device(args.device), args.bootstrap_b
    chosen = json.loads(Path(args.selection).read_text())
    pick = {"E4a": chosen["E4a"]["chosen"], "E4b": chosen["E4b"]["chosen"]}
    runs, pdir = Path(args.runs), Path(args.params)
    nets = lambda set_name: {arm: [runs / e4_run_name(arm, set_name, s) for s in SEEDS] for arm in ARMS}
    validate_choices(pick, BASELINES + tuple(ARMS), BASELINES + tuple(ARMS) + ("kf_ca_los",))
    out = Path(args.out)
    out.mkdir(parents=True)                                   # marks the single use from here on

    trajs = load_trajectories(Path(args.data), split)
    gt = make_windows(trajs, cfg.obs_len, cfg.pred_len, cfg.fps)
    err_a, rot_a = method_errors(gt, json.loads(Path(args.clean_params).read_text()), cfg, nets("clean"), device)
    res = {"split": split, "chosen_on_val": pick, "E4a": {"summary": summarize_e4(err_a, rot_a, gt, cfg, b)}}
    final, final08 = (json.loads((pdir / f"{n}.json").read_text()) for n in ("params_final", "params_08_final"))
    for cls in ("causal", "gt"):
        paths = [Path(args.poses) / f"poses_{split}_s{k}_{cls}.jsonl" for k in SEEDS]
        err_b, rot_b, idx, info = pipeline_errors(trajs, paths, cfg, lambda s, m: final if m else final08,
                                                  lambda s: nets("final"), device, allow_test=split == "test")
        wb = gt.subset(idx)
        res[f"E4b_{cls}"] = {"summary": summarize_e4(err_b, rot_b, wb, cfg, b), "counts": info,
                             "degradation_vs_E4a": degradation(err_b, {k: v[idx] for k, v in err_a.items()}, wb, b)}
        t3 = {pick["E4b"]: err_b[pick["E4b"]]}
        res[f"E4b_{cls}"]["degradation_T3"] = degradation(t3, {pick["E4b"]: err_a[t3_e4a_key(pick["E4b"])][idx]}, wb, b)
    top = lambda key, m: next(g for g in res[key]["summary"]["groups"] if g["group"] == "metri: tutti")[
        "skill_vs_CV-2"][m]["ade_1s"]
    res["criteria"] = {
        "T1": {"method": pick["E4a"], "skill_ade_1s_vs_CV-2": top("E4a", pick["E4a"])},
        "T2": {"method": pick["E4b"], "skill_ade_1s_vs_CV-2": top("E4b_causal", pick["E4b"])},
        "T3": {"method": pick["E4b"], "e4a_counterpart": t3_e4a_key(pick["E4b"]),
               **res["E4b_causal"]["degradation_T3"][pick["E4b"]]}}
    res["seconds"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "criteria.json").write_text(json.dumps(res["criteria"], indent=1))
    pct = lambda v: f"{100 * v[0]:+.1f}% [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"
    c = res["criteria"]
    lines = ["# E4a / E4b sul test del 6DroneSyn (scene 06, 07) — passaggio unico", "",
             f"- **T1 (E4a, {c['T1']['method']} vs CV-2):** {pct(c['T1']['skill_ade_1s_vs_CV-2'])}",
             f"- **T2 (E4b, {c['T2']['method']} vs CV-2 sulle stesse pose):** {pct(c['T2']['skill_ade_1s_vs_CV-2'])}",
             f"- **T3 (degrado oracle → pipeline, {c['T3']['method']}):** {c['T3']['ade_1s_e4a']:.3f} → "
             f"{c['T3']['ade_1s_e4b']:.3f} m ({c['T3']['mean']:+.3f} [{c['T3']['ci'][0]:+.3f}, {c['T3']['ci'][1]:+.3f}])",
             "", "| ADE 1 s (m), metri | E4a | E4b classe causale | E4b classe vera |", "|---|---|---|---|"]
    g = {k: next(x for x in res[k]["summary"]["groups"] if x["group"] == "metri: tutti")
         for k in ("E4a", "E4b_causal", "E4b_gt")}
    for m in g["E4b_causal"]["position"]:
        cell = lambda k: f"{g[k]['position'][m]['ade'][-1]:.3f}" if m in g[k]["position"] else "—"
        lines.append(f"| {m}{' (riferimento)' if m == REFERENCE else ''} | {cell('E4a')} | {cell('E4b_causal')} | "
                     f"{cell('E4b_gt')} |")
    lines += ["", f"Conteggi E4b (classe causale): {res['E4b_causal']['counts']}"]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
