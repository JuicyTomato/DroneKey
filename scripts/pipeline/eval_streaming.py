"""Offline vs streaming on the 6DroneSyn val, seed 0 (spec 2026-11-14 §7), from the live perception cache.

Usage (venv ~/.venvs/dronepose-yolo, from the repo root):
  python scripts/pipeline/eval_streaming.py [--config configs/pipeline_sN.json] [--perception PERC.jsonl]
      [--out OUT_DIR] [--W 60 --obs e] [--bootstrap-b 10000] [--device auto] [--extras]
Defaults follow the config's seed: experiments/pipeline/perception_val_sN.jsonl, experiments/pipeline/streaming_val_sN.
--W / --obs freeze the choices (seeds 1, 2 reuse those of seed 0) instead of applying the selection rules.
Rows: offline (DroneKey); streaming W = 1, 15, 30, 60, 120 (emitted poses); chosen W with re-decided poses;
context: offline with the YOLO26s seed-0 keypoints through the same stages. Never touches the test split.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "forecast" / "e4"))

import numpy as np  # noqa: E402

from dronepose.eval.bootstrap import by_trajectory, paired_bootstrap  # noqa: E402
from dronepose.forecast.config import load_config  # noqa: E402
from dronepose.forecast.dataset import load_trajectories  # noqa: E402
from dronepose.forecast.e4 import load_pose_windows  # noqa: E402
from dronepose.forecast.e4_eval import REFERENCE, align_window_rows, degradation  # noqa: E402
from dronepose.forecast.metrics import ade_at  # noqa: E402
from dronepose.forecast.nets.device import pick_device  # noqa: E402
from dronepose.forecast.windows import make_windows  # noqa: E402
from dronepose.keypoints.class_scores import reproj_per_class  # noqa: E402
from dronepose.pipeline.decide import decide_sequence, redecided_windows, write_pose_file  # noqa: E402
from dronepose.pipeline.klass import load_klg_params  # noqa: E402
from dronepose.pipeline.perception import Perception  # noqa: E402
from dronepose.pipeline.sources import SixDroneSynSource  # noqa: E402
from dronepose.pipeline.streaming_eval import (NET, pose_summary, select_obs_mode, select_window,  # noqa: E402
                                               select_window_flips, window_errors)
from dronepose.pose.replay import load_keypoints  # noqa: E402
from e4_train import e4_run_name  # noqa: E402

WS = (1, 15, 30, 60, 120)
E4B_THESIS_KLG = 0.153


def run_variant(src, percs, params, mode, W=None, gravity=False):
    res, red = {}, {}
    for seq in src.sequences():
        recs = src.records(seq)
        res[seq], red[seq] = decide_sequence([percs[r["frame_id"]] for r in recs], recs, src.info(seq), params,
                                             mode, W, gravity)
    return res, red


def flat(res):
    return [r for rs in res.values() for r in rs]


def flips(results):
    return by_trajectory([{**r, "flip": None if r["rot_err_deg"] is None else float(r["rot_err_deg"] > 90)}
                          for r in results if r["frame_id"].split("_")[1] != "08"], "flip")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", default=None)
    ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--W", type=int, default=None, help="frozen window (skips the selection)")
    ap.add_argument("--obs", choices=["e", "r"], default=None, help="frozen observation mode")
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--bootstrap-b", type=int, default=10000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--extras", action="store_true",
                    help="post hoc analyses (gravity prior, flip-aware W rule), written to extras.json / extras.md; "
                         "results.json and table.md are not changed")
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg_p = json.loads(Path(args.config).read_text())
    seed = int(cfg_p["seed"])
    perc_path = args.perception or f"experiments/pipeline/perception_val_s{seed}.jsonl"
    out = Path(args.out or f"experiments/pipeline/streaming_val_s{seed}")
    (out / "poses").mkdir(parents=True, exist_ok=True)
    src = SixDroneSynSource(Path(args.canonical))
    with open(perc_path) as f:
        percs = {p.frame_id: p for p in (Perception.from_row(json.loads(l)) for l in f if l.strip())}
    by_id = {r["frame_id"]: r for s in src.sequences() for r in src.records(s)}
    if missing := [i for i in by_id if i not in percs]:
        sys.exit(f"perception cache misses {len(missing)} frames, e.g. {missing[0]}")
    params = load_klg_params(Path(cfg_p["class_params"]))
    b = args.bootstrap_b

    # --- poses -----------------------------------------------------------------------------------------------
    variants = {"offline": run_variant(src, percs, params, "offline")}
    for W in WS:
        variants[f"W{W}"] = run_variant(src, percs, params, "streaming", W)
        print(f"W={W} done {time.time() - t0:.0f}s", flush=True)
    yolo_kp = load_keypoints(Path(cfg_p["yolo_keypoints"]))
    models = src.info(src.sequences()[0]).models
    percs_y = {}
    for fid, p in percs.items():
        k = np.asarray(yolo_kp.get(fid, np.full((4, 3), np.nan)), float)
        rp = reproj_per_class({"K": by_id[fid]["K"]}, k, models) if p.probs is not None else None
        percs_y[fid] = Perception(fid, p.frame_idx, p.box, p.box_conf, k, p.probs, rp, {})
    variants["offline_yolo"] = run_variant(src, percs_y, params, "offline")

    res = {"summary": {k: pose_summary(flat(v[0])) for k, v in variants.items()}}
    rot = {k: by_trajectory(flat(v[0]), "rot_err_deg") for k, v in variants.items()}
    res["rot_vs_offline"] = paired_bootstrap(rot, ref="offline", n_boot=b)
    res["flip_no08_vs_offline"] = paired_bootstrap({k: flips(flat(v[0])) for k, v in variants.items()},
                                                   ref="offline", n_boot=b, stat="mean")
    sel = select_window({W: res["summary"][f"W{W}"] for W in WS if W > 1},
                        {W: rot[f"W{W}"] for W in WS if W > 1}, n_boot=b)
    if args.W is not None:
        sel = {**sel, "chosen": args.W, "frozen": True}
    res["window_selection"] = sel
    Wc = sel["chosen"]
    print(f"poses done, W chosen {Wc} (best {sel['best']}, tied {sel['tied']}) {time.time() - t0:.0f}s", flush=True)

    # --- forecast ------------------------------------------------------------------------------------------
    fcfg = load_config(Path("configs/forecast.json"))
    trajs = load_trajectories(Path("data/processed/forecast"), "val")
    device = pick_device(args.device)
    pdir, runs = Path(cfg_p["forecast_params"]), Path(cfg_p["forecast_runs"])
    fold = {s: json.loads((pdir / f"params_fold{s}.json").read_text()) for s in ("03", "09", "14")}
    params_for = lambda scene: fold[scene]
    nets_for = lambda scene: {NET: [runs / e4_run_name(NET, f"fold{scene}", int(cfg_p["seed"]))]}
    errs = {}
    for name in ("offline", f"W{Wc}", "offline_yolo"):
        path = out / "poses" / f"poses_val_s{seed}_{name}.jsonl"
        write_pose_file(variants[name][0], by_id, "val", seed, f"pipeline_{name}", path)
        w, _ = load_pose_windows(path, trajs, fcfg)
        errs[name] = window_errors(w, fcfg, params_for, nets_for, device)
    w_r = redecided_windows(variants[f"W{Wc}"][1], variants[f"W{Wc}"][0], by_id, trajs, fcfg, seed)
    errs[f"W{Wc}_r"] = window_errors(w_r, fcfg, params_for, nets_for, device)
    print(f"forecast errors done {time.time() - t0:.0f}s", flush=True)

    gt_all = make_windows(trajs, fcfg.obs_len, fcfg.pred_len, fcfg.fps)
    H = fcfg.pred_len

    def paired(a, bname):
        ia, ib = align_window_rows(errs[a][2], errs[bname][2])
        ea = {k: v[ia] for k, v in errs[a][0].items()}
        eb = {k: v[ib] for k, v in errs[bname][0].items()}
        return ea, eb, gt_all.subset(errs[a][1][ia])

    ee, er, wer = paired(f"W{Wc}", f"W{Wc}_r")
    res["obs_mode_selection"] = select_obs_mode(ee, er, wer, b)
    if args.obs is not None:
        res["obs_mode_selection"] = {**res["obs_mode_selection"], "chosen": args.obs, "frozen": True}
    res["seed"] = seed
    stream = f"W{Wc}" if res["obs_mode_selection"]["chosen"] == "e" else f"W{Wc}_r"
    res["forecast"] = {}
    for name in ("offline", f"W{Wc}", f"W{Wc}_r", "offline_yolo"):
        e = errs[name][0]
        res["forecast"][name] = {k: {"ade_1s": float(ade_at(e[k], H).mean()), "fde_1s": float(e[k][:, -1].mean()),
                                     "n_windows": int(len(e[k]))} for k in (NET, REFERENCE)}
    eo, es, wos = paired("offline", stream)
    res["forecast_streaming_minus_offline"] = degradation(es, eo, wos, b)
    res["streaming_row"] = stream
    res["e4b_thesis_klg_ade_1s"] = E4B_THESIS_KLG
    (out / "results.json").write_text(json.dumps(res, indent=1, default=lambda x: x.tolist()
                                                 if isinstance(x, np.ndarray) else float(x)))
    write_table(out / "table.md", res, Wc, stream, seed)
    print((out / "table.md").read_text())
    if args.extras:
        extras = run_extras(args, src, percs, params, variants, res, rot, by_id, trajs, fcfg, errs, paired, params_for,
                            nets_for, device, Wc, seed, b, t0)
        (out / "extras.json").write_text(json.dumps(extras, indent=1, default=lambda x: x.tolist()
                                                    if isinstance(x, np.ndarray) else float(x)))
        write_extras_md(out / "extras.md", extras, seed)
        print((out / "extras.md").read_text())
    print(f"total {time.time() - t0:.0f}s")


GRID = (15, 30, 60, 120)      # eligible W of the selection rules (W = 1 is excluded as in the official rule)
GWS = (1, 15, 30, 60, 120)


def run_extras(args, src, percs, params, variants, res, rot, by_id, trajs, fcfg, errs, paired, params_for, nets_for,
               device, Wc, seed, b, t0) -> dict:
    """Post hoc analyses declared as added after the results: gravity prior (`up` in `global`) and flip-aware W."""
    out = Path(args.out or f"experiments/pipeline/streaming_val_s{seed}")
    H = fcfg.pred_len
    variants["offline_g"] = run_variant(src, percs, params, "offline", gravity=True)
    for W in GWS:
        variants[f"W{W}_g"] = run_variant(src, percs, params, "streaming", W, gravity=True)
        print(f"extras W={W}_g done {time.time() - t0:.0f}s", flush=True)
    names = ["offline"] + [f"W{W}" for W in GWS] + ["offline_g"] + [f"W{W}_g" for W in GWS]
    ex = {"seed": seed, "W_official": Wc, "grid": list(GRID)}
    ex["summary"] = {k: pose_summary(flat(variants[k][0])) for k in names}
    rotx = {k: by_trajectory(flat(variants[k][0]), "rot_err_deg") for k in names}
    flp = {k: flips(flat(variants[k][0])) for k in names}
    ex["rot_vs"], ex["flip_vs"] = {}, {}
    for ref in ("offline", "offline_g"):
        ex["rot_vs"][ref] = paired_bootstrap(rotx, ref=ref, n_boot=b)
        ex["flip_vs"][ref] = paired_bootstrap(flp, ref=ref, n_boot=b, stat="mean")
    ex["flip_aware"] = {}
    for tag, suf in (("nogravity", ""), ("gravity", "_g")):
        ex["flip_aware"][tag] = select_window_flips({W: ex["summary"][f"W{W}{suf}"] for W in GRID},
                                                    {W: flp[f"W{W}{suf}"] for W in GRID},
                                                    {W: rotx[f"W{W}{suf}"] for W in GRID}, n_boot=b)
    # forecast errors: emitted poses (mode e) for the rows below
    todo = ["offline_g", f"W{Wc}_g"]
    for tag, suf in (("nogravity", ""), ("gravity", "_g")):
        todo.append(f"W{ex['flip_aware'][tag]['chosen']}{suf}")
    for name in dict.fromkeys(todo):
        if name in errs:
            continue
        path = out / "poses" / f"poses_val_s{seed}_{name}.jsonl"
        write_pose_file(variants[name][0], by_id, "val", seed, f"pipeline_{name}", path)
        w, _ = load_pose_windows(path, trajs, fcfg)
        errs[name] = window_errors(w, fcfg, params_for, nets_for, device)
        print(f"extras forecast {name} done {time.time() - t0:.0f}s", flush=True)
    ex["forecast"] = {}
    for name in dict.fromkeys(["offline"] + todo):
        e = errs[name][0]
        ex["forecast"][name] = {k: {"ade_1s": float(ade_at(e[k], H).mean()), "fde_1s": float(e[k][:, -1].mean()),
                                    "n_windows": int(len(e[k]))} for k in (NET, REFERENCE)}
    pairs = [("offline_g", "offline"), (f"W{Wc}_g", "offline_g"), (f"W{Wc}_g", "offline")]
    for tag, suf in (("nogravity", ""), ("gravity", "_g")):
        c = f"W{ex['flip_aware'][tag]['chosen']}{suf}"
        pairs += [(c, "offline_g" if suf else "offline"), (c, "offline")]
    ex["forecast_minus"] = {}
    for a, r in dict.fromkeys(pairs):
        if a == r:
            continue
        ea, er, wr = paired(a, r)
        ex["forecast_minus"][f"{a} - {r}"] = degradation(ea, er, wr, b)
    return ex


def write_extras_md(path: Path, ex: dict, seed: int) -> None:
    S = ex["summary"]
    L = [f"# Extra post hoc — val 6DroneSyn, seme {seed}", "",
         "Analisi aggiunte DOPO i risultati (prior di gravita' su `global`, regola di scelta di W per flip); "
         "non sostituiscono le righe ufficiali. Differenze: bootstrap appaiato per traiettoria.", "",
         "| riga | copertura | rot mediana (°) | Δ rot vs offline [IC] | Δ rot vs offline_g [IC] | flip senza 08 | Δ flip vs offline [IC] | Δ flip vs offline_g [IC] |",
         "|---|---|---|---|---|---|---|---|"]
    ci = lambda d, k=1.0, n=3: f"{k * d[0]:+.{n}f} [{k * d[1]:+.{n}f}, {k * d[2]:+.{n}f}]"
    for k, s in S.items():
        L.append(f"| {k} | {100 * s['coverage']:.1f}% | {s['median_rot_err_deg']:.3f} | {ci(ex['rot_vs']['offline'][k])} "
                 f"| {ci(ex['rot_vs']['offline_g'][k])} | {100 * s['flip_rate_no08']:.2f}% "
                 f"| {ci(ex['flip_vs']['offline'][k], 100, 2)} | {ci(ex['flip_vs']['offline_g'][k], 100, 2)} |")
    L += ["", "Scelta di W per flip (copertura >= 95%, meno flip senza 08, pareggio = IC della differenza include 0, "
          "poi rot mediana minore, poi W minore):", ""]
    for tag, fa in ex["flip_aware"].items():
        L.append(f"- {tag}: scelto W = {fa['chosen']} (migliore sui flip {fa['best']}, in pareggio {fa['tied']}, ammessi {fa['eligible']}).")
    L += ["", f"Previsione, emesse (droni 01–06, metri): ", "",
          f"| riga | {NET} ADE 1 s | {NET} FDE 1 s | {REFERENCE} ADE 1 s | finestre |", "|---|---|---|---|---|"]
    for k, v in ex["forecast"].items():
        L.append(f"| {k} | {v[NET]['ade_1s']:.4f} | {v[NET]['fde_1s']:.4f} | {v[REFERENCE]['ade_1s']:.4f} | {v[NET]['n_windows']} |")
    L += ["", f"{NET}, differenza ADE 1 s (prima − seconda):", ""]
    for k, d in ex["forecast_minus"].items():
        x = d[NET]
        L.append(f"- {k}: {x['mean']:+.4f} m [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}]")
    path.write_text("\n".join(L) + "\n")


def write_table(path: Path, res: dict, Wc: int, stream: str, seed: int) -> None:
    S, rb, fb = res["summary"], res["rot_vs_offline"], res["flip_no08_vs_offline"]
    L = [f"# Offline vs streaming — val 6DroneSyn, seme {seed}", "",
         "Posa (classe K+L+g, keypoint DroneKey, IPPE+LM). Differenze contro offline: bootstrap appaiato per traiettoria.", "",
         "| riga | copertura | rot mediana (°) | Δ vs offline [IC 95%] | p90 (°) | flip senza 08 | Δ flip [IC] | t mediana |",
         "|---|---|---|---|---|---|---|---|"]
    for k in ["offline"] + [f"W{W}" for W in WS] + ["offline_yolo"]:
        s, d, f = S[k], rb[k], fb[k]
        L.append(f"| {k} | {100 * s['coverage']:.1f}% | {s['median_rot_err_deg']:.3f} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] "
                 f"| {s['p90_rot_err_deg']:.2f} | {100 * s['flip_rate_no08']:.2f}% | {100 * f[0]:+.2f} [{100 * f[1]:+.2f}, {100 * f[2]:+.2f}] "
                 f"| {s['median_t_err']:.4f} |")
    sel, om = res["window_selection"], res["obs_mode_selection"]
    L += ["", f"- **W {'fissato dal seme 0' if sel.get('frozen') else 'scelto'}:** {Wc} (migliore {sel['best']}, in pareggio {sel['tied']}, ammessi {sel['eligible']}).",
          f"- **Osservazioni:** ({om['chosen']}{', fissato dal seme 0' if om.get('frozen') else ''}); ridecise − emesse ADE 1 s {om['r_minus_e']['mean']:+.4f} m "
          f"[{om['r_minus_e']['ci'][0]:+.4f}, {om['r_minus_e']['ci'][1]:+.4f}].", "",
          "Previsione (droni 01–06, metri, media sulle finestre):", "",
          f"| riga | {NET} ADE 1 s | {NET} FDE 1 s | {REFERENCE} ADE 1 s | finestre |", "|---|---|---|---|---|"]
    for k, v in res["forecast"].items():
        L.append(f"| {k} | {v[NET]['ade_1s']:.4f} | {v[NET]['fde_1s']:.4f} | {v[REFERENCE]['ade_1s']:.4f} | {v[NET]['n_windows']} |")
    d = res["forecast_streaming_minus_offline"]
    L += ["", f"- **Streaming ({stream}) − offline, ADE 1 s:** "
          + "; ".join(f"{m} {x['mean']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}]" for m, x in d.items() if m in (NET, REFERENCE)),
          f"- Contesto: E4b della tesi con classe K+L+g e keypoint YOLO26s (3 semi di posa × 3 della rete): {res['e4b_thesis_klg_ade_1s']} m.",
          "- W e (e)/(r) sono scelti sulla stessa val su cui si misura la perdita: lo streaming è leggermente favorito."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
