"""Drone-class grid on val (spec 2026-10-27-classe-drone-design.md): appearance x time x geometry, leave-one-scene-out
tuning, class accuracy with paired bootstrap, decomposition, selection; optionally the pose level (Task 8).

Usage (project venv, from the repo root):
  python scripts/class/eval_class.py --out experiments/class/eval/class_val.json [--no-pose] [--workers N]
Inputs (fixed paths): C detections, all-boxes export of C (CM; cells skipped if missing), classifier probabilities
experiments/class/k/probs_val_s{0,1,2}.jsonl (K; skipped if missing), geometry experiments/class/geom/geom_val_s{N}.jsonl.
Writes OUT (.json + .md), class maps experiments/class/maps/map_val_s{N}_{cell}.json (each scene with the parameters of
its own fold) and experiments/class/eval/final_params.json (chosen cell, tuned on all val: only written, unused here).
Val only.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
from dronepose.data.canonical import read_jsonl  # noqa: E402
from dronepose.eval.bootstrap import by_trajectory, paired_bootstrap  # noqa: E402
from dronepose.eval.class_eval import class_metrics, correct_rows  # noqa: E402
from dronepose.eval.class_select import GRID, Cell, choose_cell, loso_maps, param_grid, tune_all, tune_loso  # noqa: E402
from dronepose.keypoints.class_scores import (combine, fuse_causal, load_boxes, load_probs, log_scores, scores_c1,  # noqa: E402
                                              scores_cm, scores_geom, scores_k)
from dronepose.keypoints.detections import class_causal_vote, load_detections  # noqa: E402
from dronepose.pose.replay import check_no_test  # noqa: E402

SEEDS = (0, 1, 2)
KP = "experiments/results/keypoints"
KPTS = {0: f"{KP}/keypoints_val_b_cbest_yolo26s.jsonl", 1: f"{KP}/keypoints_val_b_cbest_crop_s_w1_s1.jsonl",
        2: f"{KP}/keypoints_val_b_cbest_crop_s_w1_s2.jsonl"}
DET = f"{KP}/keypoints_val_c_yolo26s_best.jsonl"
CM = "experiments/class/cm/keypoints_val_c_allboxes.jsonl"
K = "experiments/class/k/probs_val_s{}.jsonl"
GEOM = "experiments/class/geom/geom_val_s{}.jsonl"
BASE = "C1+V"
N_BOOT = 2000

_S: dict = {}   # sources, filled once per process (main and every worker, see _init)


def load_sources(records: list[dict]) -> dict:
    dets = load_detections(Path(DET))
    src = {"records": records, "dets": dets, "c1_w": scores_c1(dets, weighted=True), "boxes": None, "k": None,
           "geom": {s: {r["frame_id"]: r["reproj_px"] for r in read_jsonl(Path(GEOM.format(s)))} for s in SEEDS}}
    if Path(CM).exists():
        src["boxes"] = load_boxes(Path(CM))
    if all(Path(K.format(s)).exists() for s in SEEDS):
        src["k"] = {s: scores_k(load_probs(Path(K.format(s)))) for s in SEEDS}
    return src


def _init() -> None:
    records = read_jsonl(Path("data/processed/canonical_val.jsonl"))
    check_no_test(records, allow_test=False)
    _S.update(load_sources(records))


def available(cell: Cell, src: dict) -> bool:
    return ("CM" not in cell.aspect or src["boxes"] is not None) and ("K" not in cell.aspect or src["k"] is not None)


@lru_cache(maxsize=None)
def _cm(iou_min: float):
    return scores_cm(_S["boxes"], iou_min)


@lru_cache(maxsize=None)
def _geom(seed: int, form: str, sigma: float):
    return scores_geom(_S["geom"][seed], form, sigma=sigma)


def _aspect(cell: Cell, params: dict, seed: int):
    if cell.aspect == "none":
        return None
    parts = []
    if cell.aspect == "C1":
        parts.append((_S["c1_w"], 1.0))
    if "K" in cell.aspect:
        parts.append((_S["k"][seed], 1.0))
    if "CM" in cell.aspect:
        parts.append((_cm(params["iou_min"]), params.get("w_cm", 1.0)))
    if cell.temporal == "L":
        parts = [(log_scores(s, params["a_min"]), w) for s, w in parts]
    return combine(parts)


def build(cell: Cell, params: dict) -> dict[int, dict[str, str]]:
    """Class map per seed of one cell with one set of parameters."""
    recs = _S["records"]
    if cell.temporal == "V":
        m = class_causal_vote(recs, _S["dets"])
        return {s: m for s in SEEDS}
    out = {}
    for s in SEEDS:
        parts = []
        a = _aspect(cell, params, s)
        if a is not None:
            parts.append((a, 1.0))
        if cell.geom:
            parts.append((_geom(s, params["form"], params.get("sigma", 1.0)), params.get("w_g", 1.0)))
        out[s] = fuse_causal(recs, combine(parts), "F" if cell.temporal == "F" else "S")
    return out


def run_cell(cell: Cell) -> dict:
    recs = _S["records"]
    grid = param_grid(cell)
    fn = lambda p: build(cell, p)  # noqa: E731
    folds = tune_loso(fn, recs, grid)
    maps = loso_maps(fn, recs, folds)
    return {"cell": cell, "folds": folds, "final_params": tune_all(fn, recs, grid), "maps": maps,
            "metrics": {s: class_metrics(recs, maps[s]) for s in SEEDS}}


def det_accuracy(src: dict) -> dict:
    """Per-detection accuracy of each appearance source (frame argmax, detected frames only)."""
    truth = {r["frame_id"]: r["drone"] for r in src["records"]}

    def acc(scores):
        ok = [max(sc, key=sc.get) == truth[f] for f, sc in scores.items() if sc and f in truth]
        by: dict[str, list[bool]] = {}
        for f, sc in scores.items():
            if sc and f in truth:
                by.setdefault(truth[f], []).append(max(sc, key=sc.get) == truth[f])
        return {"acc": float(np.mean(ok)), "n": len(ok), "by_drone": {d: float(np.mean(v)) for d, v in sorted(by.items())}}
    out = {"C1": acc(src["c1_w"])}
    if src["boxes"] is not None:
        out["CM_iou0.5"] = acc(_cm(0.5))
        multi = [len(sc) > 1 for sc in _cm(0.5).values() if sc]
        out["CM_multi_class_frames"] = float(np.mean(multi))
    if src["k"] is not None:
        out["K"] = {s: acc(src["k"][s]) for s in SEEDS}
    return out


def boots_factory(results: dict, records: list[dict]):
    cache: dict[str, dict] = {}

    def boots_vs(ref: str) -> dict[str, list[tuple[float, float, float]]]:
        if ref not in cache:
            per_seed = []
            for s in SEEDS:
                groups = {n: by_trajectory(correct_rows(records, r["maps"][s]), "correct") for n, r in results.items()}
                per_seed.append(paired_bootstrap(groups, ref, N_BOOT, seed=s, stat="mean"))
            cache[ref] = {n: [b[n] for b in per_seed] for n in results}
        return cache[ref]
    return boots_vs


def decomposition(acc: dict[str, list[float]], boots_vs) -> list[dict]:
    """Contribution of one axis at a time (spec 5.5), each with its per-seed paired interval."""
    mean = {n: float(np.mean(v)) for n, v in acc.items()}
    rows = []

    def add(what, a, b):
        if a in acc and b in acc:
            rows.append({"what": what, "cell": a, "vs": b, "diff": mean[a] - mean[b], "ci": boots_vs(b)[a]})
    best_c1_time = max((n for n in ("C1+W", "C1+L") if n in acc), key=mean.get)
    add("tempo (C1, miglior somma contro frame)", best_c1_time, "C1+F")
    add("tempo (C1, somma lineare contro maggioranza)", "C1+W", "C1+V")
    t = best_c1_time.split("+")[1]
    for asp in ("CM", "K", "K+CM"):
        add(f"aspetto ({asp} contro C1, tempo {t})", f"{asp}+{t}", best_c1_time)
    no_g = [n for n in acc if not n.endswith("+g") and not n.startswith("G+") and n != BASE]
    best_no_g = max(no_g, key=mean.get)
    add("geometria (aggiunta alla migliore senza)", best_no_g + "+g", best_no_g)
    add("geometria da sola contro C1 oggi", "G+L", BASE)
    return rows


def _fmt_ci(ci) -> str:
    return " ".join(f"[{lo:+.3f}, {hi:+.3f}]" for _, lo, hi in ci)


def write_md(path: Path, out: dict) -> None:
    lines = ["# Classe del drone sulla val — griglia", "", "| Cella | Classe per frame (s0 / s1 / s2) | Media | "
             "Sequenze giuste | Costo | vs base (IC per seme) |", "|---|---|---|---|---|---|"]
    for n, c in sorted(out["cells"].items(), key=lambda kv: -kv[1]["acc_mean"]):
        lines.append(f"| {n} | {' / '.join(f'{a:.3f}' for a in c['acc'])} | {c['acc_mean']:.3f} | "
                     f"{np.mean(c['seq_final_acc']):.3f} | {c['cost']} | {_fmt_ci(c['vs_base'])} |")
    lines += ["", "## Scomposizione", "", "| Asse | Cella | Contro | Differenza | IC per seme |", "|---|---|---|---|---|"]
    lines += [f"| {r['what']} | {r['cell']} | {r['vs']} | {r['diff']:+.3f} | {_fmt_ci(r['ci'])} |"
              for r in out["decomposition"]]
    lines += ["", f"**Scelta:** {out['choice']} (pose_ok da {'Task 8' if out['pose'] else '--no-pose: tutte vere'})",
              "", "## Accuratezza per detection delle fonti", "", "```", json.dumps(out["det_accuracy"], indent=1), "```"]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pose", action="store_true", help="skip the pose level (pose_ok true for every cell)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--maps-dir", default="experiments/class/maps")
    args = ap.parse_args(argv)

    _init()
    records = _S["records"]
    cells = [c for c in GRID if available(c, _S)]
    skipped = [c.name for c in GRID if not available(c, _S)]
    print(f"{len(cells)} cells ({len(skipped)} skipped: missing inputs {skipped})", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        results = {r["cell"].name: r for r in ex.map(run_cell, cells)}
    acc = {n: [r["metrics"][s]["frame_acc"] for s in SEEDS] for n, r in results.items()}
    boots_vs = boots_factory(results, records)

    pose = None
    pose_ok = {n: True for n in results}
    if not args.no_pose:
        raise SystemExit("pose level: Task 8")

    choice = choose_cell(acc, boots_vs, {n: r["cell"] for n, r in results.items()}, pose_ok)
    maps_dir = Path(args.maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)
    for n, r in results.items():
        for s in SEEDS:
            (maps_dir / f"map_val_s{s}_{n}.json").write_text(json.dumps(r["maps"][s]))
    base_ci = boots_vs(BASE)
    out = {"split": "val", "skipped_cells": skipped, "choice": choice, "pose": pose,
           "det_accuracy": det_accuracy(_S), "decomposition": decomposition(acc, boots_vs),
           "cells": {n: {"acc": acc[n], "acc_mean": float(np.mean(acc[n])), "cost": r["cell"].cost,
                         "n_params": r["cell"].n_params, "folds": r["folds"], "final_params": r["final_params"],
                         "seq_final_acc": [r["metrics"][s]["seq_final_acc"] for s in SEEDS],
                         "by_drone": {s: r["metrics"][s]["by_drone"] for s in SEEDS},
                         "by_scene": {s: r["metrics"][s]["by_scene"] for s in SEEDS},
                         "vs_base": base_ci[n], "vs_choice": boots_vs(choice)[n], "pose_ok": pose_ok[n]}
                     for n, r in results.items()}}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=1))
    write_md(outp.with_suffix(".md"), out)
    (outp.parent / "final_params.json").write_text(json.dumps(
        {"cell": choice, "params": results[choice]["final_params"], "note": "tuned on all val scenes; frozen"}, indent=1))
    print(f"choice {choice}; -> {outp} (+ .md), maps in {maps_dir}", flush=True)


if __name__ == "__main__":
    main()
