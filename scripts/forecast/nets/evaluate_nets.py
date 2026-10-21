"""Evaluate the nets of plan step 3.3 (Task 8): criteria D1-D4 (spec §6), tables, diagnostics, learning curve.

Writes experiments/forecast/nets/{gen_val, noise, dronesyn_val}/, learning_curve.json, criteria.json, table.md.
Bases inside the nets: always the parameters they were trained with (clean: generator tuning.json; noisy:
base_params_noisy.json), also on the 6DroneSyn. Comparison rows on the 6DroneSyn (CV-2, Kalman CA) use the parameters
tuned on the 6DroneSyn train; F on the 6DroneSyn is the physics inside R-F (generated parameters). D2 compares R-F
trained with noise against F re-tuned for each level and noise seed (noise_retuned.json, same val noise draws).
Gaussian noise is NOT the real PnP error. Never touches a test split.

Usage: python scripts/forecast/nets/evaluate_nets.py [--device auto] [--bootstrap-b 10000]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.metrics import ade_at, bootstrap_counts, fde_at, position_errors, skill
from dronepose.forecast.nets.device import pick_device
from dronepose.forecast.nets.evaluate import (DEFINITIONS, contrast, criterion_d1, criterion_d2, criterion_d3,
                                              criterion_d4, predict_run, refuse_test_split, window_errors)
from dronepose.forecast.nets.features import base_predictions, build_batch
from dronepose.forecast.nets.train import run_name
from dronepose.forecast.noise import SIGMA_P_CM, SIGMA_R_DEG, add_pose_noise, level_key, noise_rng
from dronepose.forecast.report import _generated_groups
from dronepose.forecast.windows import freeze_windows, make_windows

SEEDS = (0, 1, 2)
# name in the tables -> (arm, regime, rnn, use_mode); the four arms first, then the secondary variants
ARMS = {"G": ("G", "clean", "gru", False), "R-CV": ("R-CV", "clean", "gru", False),
        "R-F": ("R-F", "clean", "gru", False), "R-F-P": ("R-F-P", "clean", "gru", False),
        "R-F-LSTM": ("R-F", "clean", "lstm", False), "R-F con modo": ("R-F", "clean", "gru", True),
        "R-F rumore": ("R-F", "noisy", "gru", False)}
LEARNING = (2500, 10000, None)
NOISE_LEVELS = [(sp, sr) for sp, sr in zip(SIGMA_P_CM, SIGMA_R_DEG) if sp or sr]


def run_dirs(root: Path, name: str) -> list[Path]:
    arm, regime, rnn, use_mode = ARMS[name]
    return [root / run_name(arm, regime, s, rnn, use_mode) for s in SEEDS]


class Evaluator:
    def __init__(self, args):
        self.cfg = load_config(Path(args.config))
        self.b = args.bootstrap_b
        self.device = pick_device(args.device)
        self.runs = Path(args.runs)
        self.params = {"clean": json.loads(Path(args.clean_params).read_text()),
                       "noisy": json.loads(Path(args.noisy_params).read_text())}
        self.dronesyn_params = json.loads(Path(args.dronesyn_params).read_text())
        self.retuned = json.loads(Path(args.noise_retuned).read_text())
        self.data, self.dronesyn_dir = Path(args.data), Path(args.dronesyn)

    def windows(self, name):
        refuse_test_split(name)
        c = self.cfg
        return freeze_windows(load_trajectories(self.data, name), c.obs_len, c.pred_len, c.fps)

    def net_preds(self, name, batches: dict, return_w=False):
        """Predictions of every seed of an arm; batches = regime -> Batch (the arm reads its regime's batch)."""
        regime = ARMS[name][1]
        return [predict_run(d, batches[regime], self.device, return_w) for d in run_dirs(self.runs, name)]

    # ----- generated sets ---------------------------------------------------------------------------------------
    def generated(self, split: str) -> dict:
        w = self.windows(split)
        batches = {"clean": build_batch(w, self.params["clean"], self.cfg)}
        batches["noisy"] = build_batch(w, self.params["noisy"], self.cfg)      # R-F rumore on clean poses: observation
        target = batches["clean"].target
        preds = {n: self.net_preds(n, batches) for n in ARMS}
        bc = batches["clean"].bases
        refs = {"F": bc["phys"], "F modo noto": bc["phys_mode"], "Kalman CA": bc["kf_ca"], "CV-2": bc["cv"]}
        wt = np.asarray(w.meta["weight"], float)
        err = {n: [window_errors(p, target) for p in ps] for n, ps in preds.items()}
        err.update({n: [window_errors(p, target)] for n, p in refs.items()})
        rows = self._rows({**preds, **{n: [p] for n, p in refs.items()}}, target, wt, w.traj, w.meta["mode"], err)
        groups = []
        for gname, mask in _generated_groups(w):
            counts = bootstrap_counts(w.traj[mask], w.meta["mode"][mask], self.b, self.cfg.bootstrap_seed)
            g = {"group": gname, "n": int(mask.sum()), "ade_1s": {}, "skill_vs_F": {}}
            for n, es in err.items():
                g["ade_1s"][n] = float(np.average(np.mean(es, 0)[mask], weights=wt[mask])) if wt[mask].sum() > 0 \
                    else float("nan")
                if n in ARMS:
                    g["skill_vs_F"][n] = contrast([e[mask] for e in es], err["F"][0][mask], wt[mask], None, None,
                                                  self.b, 0, counts)
            groups.append(g)
        out = {"split": split, "n": len(w), "rows": rows, "groups": groups}
        if "R-F-P" in preds:
            ws = [predict_run(d, batches["clean"], self.device, True)[1] for d in run_dirs(self.runs, "R-F-P")]
            out["rfp_w"] = {"mean": float(np.average(np.mean(ws, 0), weights=wt)),
                            "per_mode": {m: float(np.average(np.mean(ws, 0)[w.meta["mode"] == m],
                                                             weights=wt[w.meta["mode"] == m]))
                                         for m in sorted(set(w.meta["mode"]))}}
        return out, err, w

    def _rows(self, preds: dict, target, wt, traj, strata, err) -> dict:
        hs, H = self.cfg.horizons_frames, target.shape[1]
        counts = bootstrap_counts(traj, strata, self.b, self.cfg.bootstrap_seed)
        rows = {}
        for n, ps in preds.items():
            e = [position_errors(p, target) for p in ps]
            r = {"ade": [float(np.average(np.mean([ade_at(x, h) for x in e], 0), weights=wt)) for h in hs],
                 "fde": [float(np.average(np.mean([fde_at(x, h) for x in e], 0), weights=wt)) for h in hs],
                 "n_seeds": len(ps)}
            for ref in ("F", "Kalman CA", "CV-2"):
                if ref in err and n != ref:
                    r[f"skill_vs_{ref}"] = contrast(err[n], err[ref][0], wt, None, None, self.b, 0, counts)
            rows[n] = r
        return rows

    # ----- noise (D2) -------------------------------------------------------------------------------------------
    def noise(self) -> dict:
        w = self.windows("val")
        wt = np.asarray(w.meta["weight"], float)
        by = {(r["sigma_p_cm"], r["sigma_r_deg"], r["seed"]): r for r in self.retuned["runs"]}
        levels, table = {}, {}
        for sp, sr in NOISE_LEVELS:
            rf_net = {s: [] for s in SEEDS}
            acc = {"F ritarato": [], "Kalman CA ritarato": [], "F parametri misti": [], "R-F pulito (osservazione)": []}
            for ns in SEEDS:
                wn = add_pose_noise(w, sp, sr, noise_rng(ns, sr))
                bn = build_batch(wn, self.params["noisy"], self.cfg)
                for s, p in zip(SEEDS, self.net_preds("R-F rumore", {"noisy": bn})):
                    rf_net[s].append(window_errors(p, bn.target))
                rt = base_predictions(wn, by[(sp, sr, ns)]["params"], self.cfg)
                acc["F ritarato"].append(window_errors(rt["phys"], w.fut_p))
                acc["Kalman CA ritarato"].append(window_errors(rt["kf_ca"], w.fut_p))
                acc["F parametri misti"].append(window_errors(bn.bases["phys"], bn.target))
                bc = build_batch(wn, self.params["clean"], self.cfg)
                acc["R-F pulito (osservazione)"].append(np.mean(
                    [window_errors(p, bc.target) for p in self.net_preds("R-F", {"clean": bc})], 0))
            key = level_key(sp, sr)
            rf = [np.mean(rf_net[s], 0) for s in SEEDS]                    # per net seed, mean over the noise seeds
            f = np.mean(acc["F ritarato"], 0)
            levels[key] = (rf, f)
            ref_ade = self.retuned["results"][key]["physics_acro_gen"]["ade_1s"]
            table[key] = {"R-F rumore": float(np.average(np.mean(rf, 0), weights=wt)),
                          **{n: float(np.average(np.mean(v, 0), weights=wt)) for n, v in acc.items()},
                          "check_F_ritarato_vs_noise_retuned_json": ref_ade}
            print(f"  rumore {key}: R-F rumore {table[key]['R-F rumore']:.3f}, F ritarato {table[key]['F ritarato']:.3f} "
                  f"(noise_retuned.json {ref_ade:.3f}), F misti {table[key]['F parametri misti']:.3f}", flush=True)
        d2 = criterion_d2(levels, wt, w.traj, w.meta["mode"], self.b, self.cfg.bootstrap_seed)
        return {"table": table, "D2": d2}

    # ----- 6DroneSyn val (D3) -----------------------------------------------------------------------------------
    def dronesyn(self) -> dict:
        refuse_test_split("val")
        c = self.cfg
        w = make_windows(load_trajectories(self.dronesyn_dir, "val"), c.obs_len, c.pred_len, c.fps)
        batches = {r: build_batch(w, self.params[r], c) for r in ("clean", "noisy")}
        own = base_predictions(w, self.dronesyn_params, c)
        target = batches["clean"].target
        names = [n for n in ARMS if not ARMS[n][3]]                        # the mode is unknown on the 6DroneSyn
        err = {n: [window_errors(p, target) for p in self.net_preds(n, batches)] for n in names}
        err["F"] = [window_errors(batches["clean"].bases["phys"], target)]
        err["CV-2"] = [window_errors(own["cv"], w.fut_p)]
        err["Kalman CA"] = [window_errors(own["kf_ca"], w.fut_p)]
        m = np.asarray(w.metric, bool)
        counts = bootstrap_counts(w.traj[m], w.scene[m], self.b, c.bootstrap_seed)
        rows = {}
        for n, es in err.items():
            r = {"ade_1s": float(np.mean(np.mean(es, 0)[m])), "n_seeds": len(es)}
            for ref in ("CV-2", "Kalman CA"):
                if n != ref:
                    r[f"skill_vs_{ref}"] = contrast([e[m] for e in es], err[ref][0][m], None, None, None, self.b, 0,
                                                    counts)
            r["non_metric_skill_vs_CV-2"] = {f"drone {d}": skill(np.mean(es, 0)[~m & (w.drone == d)],
                                                                 err["CV-2"][0][~m & (w.drone == d)])
                                             for d in sorted(set(w.drone[~m]))}
            rows[n] = r
        d3 = criterion_d3(err["R-F"], err["CV-2"][0], m, w.traj, w.scene, w.drone, self.b, c.bootstrap_seed)
        ws = [predict_run(d, batches["clean"], self.device, True)[1] for d in run_dirs(self.runs, "R-F-P")]
        return {"n_windows": len(w), "n_metric": int(m.sum()), "rows": rows, "D3": d3,
                "rfp_w_mean_metric": float(np.mean(np.mean(ws, 0)[m]))}

    def learning_curve(self, err_val_target, w, batch) -> dict:
        wt = np.asarray(w.meta["weight"], float)
        out = {}
        for n in LEARNING:
            d = self.runs / run_name("R-F", "clean", 0, n_train=n)
            e = window_errors(predict_run(d, batch, self.device), batch.target)
            out[str(n or 50000)] = {"ade_1s": float(np.average(e, weights=wt)),
                                    "skill_vs_F": skill(wt * e, wt * err_val_target)}
        return out


def _fmt(c) -> str:
    return f"{100 * c['value']:+.1f}% [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}]"


def _markdown(res: dict) -> str:
    L = ["# Reti 3.3 — criteri D1–D4 e tabelle principali", "",
         "Rumore gaussiano i.i.d. sulle pose osservate: **non** è l'errore reale del PnP. Test generato e test del "
         "6DroneSyn mai usati. IC 95% bootstrap appaiato per traiettoria (generato: strati = modo, pesato; 6DroneSyn: "
         f"strati = scena), {res['bootstrap_b']} repliche; errore per traiettoria = media sui 3 semi.", "", "## Criteri", ""]
    for k, v in res["criteria"].items():
        esito = "**passa**" if v["pass"] else "**non passa**"
        if k == "D2":
            val = "; ".join(f"{lv} {_fmt(x)}" for lv, x in v["levels"].items()) + f" ({v['n_levels_pass']}/4 livelli)"
        elif k == "D4":
            val = "; ".join(f"vs {a} {_fmt(x)}" for a, x in v["contrasts"].items())
        else:
            val = _fmt(v) + ("" if v.get("per_seed") is None else
                             " — semi " + ", ".join(f"{100 * s:+.1f}%" for s in v["per_seed"]))
        L.append(f"- **{k}** — {v['definition']}: {val} → {esito}")
    hs = " / ".join(f"{h:.2f}" for h in res["horizons_s"])
    for split in ("val", "val_L0", "val_L2", "active"):
        g = res["gen"].get(split)
        if g is None:
            continue
        L += ["", f"## Generato `{split}` ({g['n']} campioni, pesata) — orizzonti {hs} s", "",
              "| braccio | ADE | FDE | skill ADE 1 s vs F | vs Kalman CA | vs CV-2 | semi vs F |", "|---|---|---|---|---|---|---|"]
        for n, r in g["rows"].items():
            sk = lambda ref: _fmt(r[f"skill_vs_{ref}"]) if f"skill_vs_{ref}" in r else "—"
            seeds = ", ".join(f"{100 * s:+.1f}" for s in r["skill_vs_F"]["per_seed"]) if "skill_vs_F" in r and \
                r["n_seeds"] > 1 else ""
            L.append(f"| {n} | {' / '.join(f'{x:.3f}' for x in r['ade'])} | {' / '.join(f'{x:.3f}' for x in r['fde'])} "
                     f"| {sk('F')} | {sk('Kalman CA')} | {sk('CV-2')} | {seeds} |")
    g = res["gen"]["val"]
    L += ["", "## Diagnostica sulla val generata (ADE 1 s pesata; skill di R-F vs F)", "",
          "| gruppo | n | F | R-F | G | R-CV | R-F-P | skill R-F vs F |", "|---|---|---|---|---|---|---|---|"]
    for gr in g["groups"]:
        a = gr["ade_1s"]
        L.append(f"| {gr['group']} | {gr['n']} | " + " | ".join(f"{a.get(n, float('nan')):.3f}" for n in
                                                                 ("F", "R-F", "G", "R-CV", "R-F-P"))
                 + f" | {_fmt(gr['skill_vs_F']['R-F'])} |")
    L += ["", "## Rumore (ADE 1 s pesata, media su 3 semi di rumore)", "",
          "| livello | R-F rumore | F ritarato | Kalman CA ritarato | F parametri misti | R-F pulito |", "|---|---|---|---|---|---|"]
    for lv, t in res["noise"]["table"].items():
        L.append(f"| {lv} | {t['R-F rumore']:.3f} | {t['F ritarato']:.3f} | {t['Kalman CA ritarato']:.3f} | "
                 f"{t['F parametri misti']:.3f} | {t['R-F pulito (osservazione)']:.3f} |")
    ds = res["dronesyn"]
    L += ["", f"## 6DroneSyn val, zero-shot ({ds['n_metric']} finestre in metri, droni 01–06)", "",
          "| metodo | ADE 1 s (m) | skill vs CV-2 | skill vs Kalman CA | drone 08 vs CV-2 (unità dataset) |",
          "|---|---|---|---|---|"]
    for n, r in ds["rows"].items():
        sk = lambda ref: _fmt(r[f"skill_vs_{ref}"]) if f"skill_vs_{ref}" in r else "—"
        nm = ", ".join(f"{100 * v:+.1f}%" for v in r["non_metric_skill_vs_CV-2"].values())
        L.append(f"| {n} | {r['ade_1s']:.3f} | {sk('CV-2')} | {sk('Kalman CA')} | {nm} |")
    L += ["", f"R-F-P: peso medio della fisica w = {res['gen']['val'].get('rfp_w', {}).get('mean', float('nan')):.3f} "
              f"sul generato, {ds['rfp_w_mean_metric']:.3f} sul 6DroneSyn (osservazione).", "",
          "## Curva di apprendimento (R-F pulito, seme 0, val generata)", "",
          "| campioni di train | ADE 1 s | skill vs F |", "|---|---|---|"]
    for n, v in res["learning_curve"].items():
        L.append(f"| {n} | {v['ade_1s']:.3f} | {100 * v['skill_vs_F']:+.1f}% |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/forecast_gen")
    ap.add_argument("--dronesyn", default="data/processed/forecast")
    ap.add_argument("--config", default="configs/forecast.json")
    ap.add_argument("--clean-params", default="experiments/forecast/generator/tuning.json")
    ap.add_argument("--noisy-params", default="experiments/forecast/nets/base_params_noisy.json")
    ap.add_argument("--dronesyn-params", default="experiments/forecast/tuning/tuning.json")
    ap.add_argument("--noise-retuned", default="experiments/forecast/generator/noise_retuned.json")
    ap.add_argument("--runs", default="experiments/forecast/nets/runs")
    ap.add_argument("--out", default="experiments/forecast/nets")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bootstrap-b", type=int, default=10000)
    args = ap.parse_args()

    t0 = time.time()
    ev = Evaluator(args)
    out = Path(args.out)
    res = {"bootstrap_b": args.bootstrap_b, "horizons_s": [h / ev.cfg.fps for h in ev.cfg.horizons_frames], "gen": {}}
    for split in ("val", "val_L0", "val_L2", "active"):
        g, err, w = ev.generated(split)
        res["gen"][split] = g
        print(f"{split}: F {g['rows']['F']['ade'][-1]:.3f}, R-F {g['rows']['R-F']['ade'][-1]:.3f} m "
              f"({time.time() - t0:.0f} s)", flush=True)
        if split == "val":
            val_err, val_w = err, w
    wt = np.asarray(val_w.meta["weight"], float)
    tr, st = val_w.traj, val_w.meta["mode"]
    crit = {"D1": criterion_d1(val_err["R-F"], val_err["F"][0], wt, tr, st, args.bootstrap_b, ev.cfg.bootstrap_seed)}
    res["noise"] = ev.noise()
    crit["D2"] = res["noise"]["D2"]
    res["dronesyn"] = ev.dronesyn()
    crit["D3"] = res["dronesyn"]["D3"]
    crit["D4"] = criterion_d4(val_err["R-F"], val_err["G"], val_err["R-CV"], wt, tr, st, args.bootstrap_b,
                              ev.cfg.bootstrap_seed)
    mode_variant = contrast(val_err["R-F con modo"], val_err["F modo noto"][0], wt, tr, st, args.bootstrap_b,
                            ev.cfg.bootstrap_seed)
    res["gen"]["val"]["mode_variant_vs_F_mode"] = mode_variant
    res["learning_curve"] = ev.learning_curve(val_err["F"][0], val_w, build_batch(val_w, ev.params["clean"], ev.cfg))
    res["criteria"] = crit
    for sub, key in (("gen_val", "gen"), ("noise", "noise"), ("dronesyn_val", "dronesyn")):
        (out / sub).mkdir(parents=True, exist_ok=True)
        (out / sub / "results.json").write_text(json.dumps(res[key], indent=1, default=float))
    (out / "learning_curve.json").write_text(json.dumps(res["learning_curve"], indent=1))
    hist = {d.name: json.loads((d / "history.json").read_text()) for d in sorted(Path(args.runs).iterdir())
            if (d / "history.json").exists()}
    (out / "histories.json").write_text(json.dumps(hist, indent=1))
    (out / "criteria.json").write_text(json.dumps({**crit, "definitions": DEFINITIONS}, indent=1, default=float))
    (out / "table.md").write_text(_markdown(res))
    for k, v in crit.items():
        print(f"{k}: pass={v['pass']}  value={v['value']}  ci={v['ci']}")
    print(f"variante con modo vs fisica a modo noto: {_fmt(mode_variant)}")
    print(f"scritto {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
