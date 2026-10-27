"""Evaluation of the nets (plan step 3.3, Task 8): predictions per seed, mean over seeds, criteria D1-D4 (spec §6).

Error of a window = ADE at 1 s averaged over the 3 seeds (and over the noise seeds for D2). On the generated sets
every mean, skill and bootstrap is weighted by the importance weight (errors multiplied by the weight before the sums
per trajectory, as report.summarize_generated); on the 6DroneSyn the weights are 1. Paired bootstrap per trajectory,
strata = command mode (generated) or scene (6DroneSyn). The test splits are refused.
"""
from pathlib import Path

import numpy as np
import torch

from dronepose.forecast.metrics import ade_at, bootstrap_counts, position_errors, skill, skill_ci
from dronepose.forecast.nets.features import Batch, Normaliser
from dronepose.forecast.nets.train import EVAL_CHUNK, build_model, load_checkpoint

DEFINITIONS = {
    "D1": "contrasto primario (unico): val generata, livello 1, pesata: R-F contro F (fisica acro generalizzata), "
          "skill ADE a 1 s > 0 con IC 95% che esclude lo zero, e skill > 0 in ciascuno dei 3 semi",
    "D2": "rumore: R-F allenato con rumore contro F ritarato per livello, sulla curva 1 cm-0.5° / 2 cm-1° / 3 cm-2° / "
          "5 cm-3° (stessi campioni e semi di rumore di noise_retuned.json): skill ADE a 1 s > 0 con IC che esclude lo "
          "zero in almeno 3 livelli su 4. Rumore gaussiano i.i.d., NON l'errore del PnP (correlato nel tempo, "
          "anisotropo, con flip)",
    "D3": "fuori dominio: val del 6DroneSyn, droni 01-06 (finestre in metri), zero-shot: R-F contro CV-2, skill ADE a "
          "1 s ≥ 0 (IC riportato). Atteso dichiarato in anticipo: probabile fallimento (la fisica parte da −390%)",
    "D2b": "rumore reale (spec 2026-10-27 §6): R-F allenato con l'archivio degli errori contro F ritarato con lo stesso "
           "rumore, val generata L1 pesata, leave-one-scene-out sulle scene 03 / 09 / 14 dell'archivio: skill ADE a 1 s "
           "> 0 con IC 95% che esclude lo zero sul totale (per campione media dei 3 fold, un solo bootstrap, strati = "
           "modo) e stima puntuale > 0 in ciascun fold",
    "D4": "contributi: val generata: R-F batte G e batte R-CV (due skill ADE a 1 s > 0 con IC che esclude lo zero)",
}


def refuse_test_split(name: str) -> None:
    if "test" in name.lower():
        raise ValueError(f"{name!r}: the test splits (generated and 6DroneSyn) are never used in step 3.3")


@torch.no_grad()
def predict_run(run_dir: Path, batch: Batch, device, return_w: bool = False):
    """Predictions in metres (frame of the batch) of the best checkpoint of a run; with return_w also R-F-P's w."""
    ck = load_checkpoint(Path(run_dir) / "best.pt")
    model = build_model(ck["arm"], ck["cfg"], ck["rnn"], ck["use_mode"], ck.get("n_in", 9)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    norm = Normaliser.from_json(ck["normaliser"])
    nb = norm.apply(batch)
    to = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    mc = np.concatenate([nb.mode_onehot, nb.class_onehot], 1)
    preds, ws = [], []
    for s in range(0, len(nb), EVAL_CHUNK):
        sl = slice(s, s + EVAL_CHUNK)
        out = model(to(nb.x[sl]), {k: to(v[sl]) for k, v in nb.bases.items()}, to(mc[sl]) if ck["use_mode"] else None)
        preds.append(out["pred"].cpu().numpy())
        if out["w"] is not None:
            ws.append(out["w"].cpu().numpy())
    pred = norm.pos_scale * np.concatenate(preds).astype(np.float64)
    if return_w:
        return pred, (np.concatenate(ws) if ws else None)
    return pred


def window_errors(pred, target) -> np.ndarray:
    """ADE at the full horizon (1 s) per window."""
    target = np.asarray(target)
    return ade_at(position_errors(pred, target), target.shape[1])


def seed_mean_errors(preds_per_seed: list, target) -> np.ndarray:
    return np.mean([window_errors(p, target) for p in preds_per_seed], axis=0)


def _skill(err_m, err_r, weight, inv, counts) -> list[float]:
    return list(skill_ci(weight * np.asarray(err_m), weight * np.asarray(err_r), inv, counts))


def contrast(seed_errs: list, ref, weight, traj, strata, b: int, seed: int, counts=None) -> dict:
    """Skill (mean over the seeds) with CI and the skill of every seed, weighted."""
    weight = np.ones(len(ref)) if weight is None else np.asarray(weight, float)
    inv, cts = counts if counts is not None else bootstrap_counts(traj, strata, b, seed)
    v = _skill(np.mean(seed_errs, 0), ref, weight, inv, cts)
    return {"value": v[0], "ci": v[1:], "per_seed": [skill(weight * e, weight * np.asarray(ref)) for e in seed_errs]}


def criterion_d1(rf_seed_errs, f_err, weight, traj, strata, b: int = 10000, seed: int = 0) -> dict:
    c = contrast(rf_seed_errs, f_err, weight, traj, strata, b, seed)
    ok = c["value"] > 0 and c["ci"][0] > 0 and all(s > 0 for s in c["per_seed"])
    return {"definition": DEFINITIONS["D1"], **c, "pass": bool(ok)}


def criterion_d2(levels: dict, weight, traj, strata, b: int = 10000, seed: int = 0) -> dict:
    """levels: name -> (R-F error per net seed (each averaged over the noise seeds), F error averaged likewise)."""
    counts = bootstrap_counts(traj, strata, b, seed)
    out = {}
    for name, (rf, f) in levels.items():
        c = contrast(rf, f, weight, traj, strata, b, seed, counts)
        out[name] = {**c, "pass": bool(c["value"] > 0 and c["ci"][0] > 0)}
    n = sum(v["pass"] for v in out.values())
    return {"definition": DEFINITIONS["D2"], "levels": out, "n_levels_pass": n,
            "value": n, "ci": [None, None], "per_seed": None, "pass": bool(n >= 3)}


def criterion_d3(rf_seed_errs, cv_err, metric, traj, scene, drone, b: int = 10000, seed: int = 0) -> dict:
    """Metric windows only (drones 01-06); the drones without a verified size (08) in a separate row."""
    m = np.asarray(metric, bool)
    rf = [np.asarray(e)[m] for e in rf_seed_errs]
    c = contrast(rf, np.asarray(cv_err)[m], None, np.asarray(traj)[m], np.asarray(scene)[m], b, seed)
    rows = {}
    for d in sorted(set(np.asarray(drone)[~m])):
        sel = ~m & (np.asarray(drone) == d)
        rows[f"drone {d}"] = {"skill": skill(np.mean([np.asarray(e)[sel] for e in rf_seed_errs], 0),
                                             np.asarray(cv_err)[sel]),
                              "n_windows": int(sel.sum()), "units": "dataset (nominale, fuori dal criterio)"}
    return {"definition": DEFINITIONS["D3"], **c, "pass": bool(c["value"] >= 0), "n_windows": int(m.sum()),
            "non_metric": rows}


def criterion_d4(rf_seed_errs, g_seed_errs, rcv_seed_errs, weight, traj, strata, b: int = 10000,
                 seed: int = 0) -> dict:
    counts = bootstrap_counts(traj, strata, b, seed)
    rf = np.mean(rf_seed_errs, 0)
    out = {}
    for name, other in (("G", g_seed_errs), ("R-CV", rcv_seed_errs)):
        v = _skill(rf, np.mean(other, 0), np.asarray(weight, float), *counts)
        out[name] = {"value": v[0], "ci": v[1:], "pass": bool(v[0] > 0 and v[1] > 0)}
    return {"definition": DEFINITIONS["D4"], "contrasts": out, "value": None, "ci": [None, None], "per_seed": None,
            "pass": bool(all(v["pass"] for v in out.values()))}


def criterion_d2b(per_fold_rf: dict, per_fold_f: dict, weight, traj, strata, b: int = 10000, seed: int = 0) -> dict:
    """per_fold_rf: held-out scene -> R-F error per net seed (each averaged over the noise draws); per_fold_f: scene ->
    F error averaged likewise. Total: per sample the mean over the folds; one bootstrap (paired, strata = mode)."""
    counts = bootstrap_counts(traj, strata, b, seed)
    folds = {s: contrast(per_fold_rf[s], per_fold_f[s], weight, traj, strata, b, seed, counts)
             for s in sorted(per_fold_f)}
    n_seeds = len(next(iter(per_fold_rf.values())))
    rf_tot = [np.mean([per_fold_rf[s][i] for s in folds], 0) for i in range(n_seeds)]
    f_tot = np.mean([per_fold_f[s] for s in folds], 0)
    c = contrast(rf_tot, f_tot, weight, traj, strata, b, seed, counts)
    ok = c["value"] > 0 and c["ci"][0] > 0 and all(v["value"] > 0 for v in folds.values())
    return {"definition": DEFINITIONS["D2b"], **c, "folds": folds, "pass": bool(ok)}
