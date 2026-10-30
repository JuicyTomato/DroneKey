"""Drone class on the 6DroneSyn TEST, once: the two kept variants with the frozen parameters, against today's class
(C1+V) and the true class (protocol in PROJECT_MEMORY §6septdecies). No choice is made here.

Usage (project venv, from the repo root):
  python scripts/class/test_class.py --allow-test               (only after the user's go-ahead; refuses if OUT exists)
  python scripts/class/test_class.py --smoke-val --out DIR      (same code on the val: code check, not a result)
Inputs per split: C detections, YOLO26s B keypoints (3 seeds), classifier probabilities and geometry per seed
(experiments/class/{k,geom}/..._{split}_s{N}.jsonl). Parameters: experiments/class/eval/final_params.json.
Three levels, as on the val: class per frame / sequence (paired bootstrap per trajectory against C1+V), pose
(ippe+lm+global, rotation bootstrap against C1+V, true class as a row), E4b (parameters and nets 'final' of
e4_test.py, ADE at 1 s against C1+V on the same windows). Writes OUT/class_test.json (+ .md), OUT/maps/ and the pose
files OUT/e4b/<cell>/poses/.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))                       # dronepose without pip install
sys.path.insert(0, str(ROOT / "scripts"))                   # export_poses
sys.path.insert(0, str(Path(__file__).resolve().parent))    # eval_class
sys.path.insert(0, str(ROOT / "scripts" / "forecast" / "e4"))   # e4_train

import eval_class as ec  # noqa: E402
from dronepose.data.canonical import read_jsonl  # noqa: E402
from dronepose.eval.bootstrap import by_trajectory, paired_bootstrap  # noqa: E402
from dronepose.eval.class_eval import class_metrics, correct_rows  # noqa: E402
from dronepose.eval.class_select import GRID  # noqa: E402
from dronepose.pose.replay import check_no_test  # noqa: E402

SEEDS = ec.SEEDS
BASE, GT = ec.BASE, ec.GT
TEST_OUT = Path("experiments/class/test")
PARAMS = Path("experiments/class/eval/final_params.json")
COMBO = "ippe+lm+global"
METHODS = ("R-CV-6D", "G-6D", "kf_ca", "kf_ca_los", "CV-2")
KP = "experiments/results/keypoints"
INPUTS = {
    "test": {"kpts": {s: f"{KP}/test/keypoints_test_b_s{s}.jsonl" for s in SEEDS},
             "det": f"{KP}/test/keypoints_test_c_best.jsonl",
             "k": "experiments/class/k/probs_test_s{}.jsonl", "geom": "experiments/class/geom/geom_test_s{}.jsonl"},
    "val": {"kpts": ec.KPTS, "det": ec.DET, "k": ec.K, "geom": ec.GEOM},
}


def guard(allow_test: bool, out: Path, smoke: bool) -> None:
    """The test is used once: explicit --allow-test, and never a second run into an existing directory."""
    if smoke:
        if out.resolve() == TEST_OUT.resolve():
            raise ValueError(f"smoke run on the val: pass an --out other than {TEST_OUT}")
    elif not allow_test:
        raise ValueError("the test split (scenes 06, 07) is used once: pass --allow-test explicitly")
    if out.exists():
        raise ValueError(f"{out} exists: il test si usa una volta sola (nessuna seconda esecuzione)")


def input_paths(split: str) -> list[Path]:
    i = INPUTS[split]
    return ([Path(p) for p in i["kpts"].values()] + [Path(i["det"])]
            + [Path(i[k].format(s)) for k in ("k", "geom") for s in SEEDS])


def setup(split: str) -> list[dict]:
    """Point eval_class at the inputs of `split` and load its sources (main process and every worker)."""
    i = INPUTS[split]
    ec.KPTS, ec.DET, ec.K, ec.GEOM, ec.CM = i["kpts"], i["det"], i["k"], i["geom"], "experiments/class/cm/none"
    records = read_jsonl(Path(f"data/processed/canonical_{split}.jsonl"))
    check_no_test(records, allow_test=split == "test")
    ec._S.clear()
    ec._S.update(ec.load_sources(records))
    ec._cm.cache_clear()
    ec._geom.cache_clear()
    return records


def frozen_cells() -> dict[str, tuple]:
    """name -> (Cell, params): today's class and the two kept variants with their frozen parameters."""
    fp = json.loads(PARAMS.read_text())
    cells = {c.name: c for c in GRID}
    out = {BASE: (cells[BASE], {})}
    for key in ("rule_choice", "best_measured"):
        out[fp[key]["cell"]] = (cells[fp[key]["cell"]], fp[key]["params"])
    return out


def class_level(records: list[dict], maps: dict[str, dict[int, dict]]) -> dict:
    out = {}
    for n, m in maps.items():
        met = [class_metrics(records, m[s]) for s in SEEDS]
        out[n] = {"frame_acc": [x["frame_acc"] for x in met], "seq_final_acc": [x["seq_final_acc"] for x in met],
                  "by_drone": [x["by_drone"] for x in met], "by_scene": [x["by_scene"] for x in met]}
    for s in SEEDS:
        groups = {n: by_trajectory(correct_rows(records, m[s]), "correct") for n, m in maps.items()}
        b = paired_bootstrap(groups, BASE, ec.N_BOOT, seed=s, stat="mean")
        for n in maps:
            out[n].setdefault("vs_base", []).append(b[n])
    return out


def pose_level(split: str, maps: dict[str, dict[int, dict]], workers: int) -> dict:
    jobs = [(n, s, m[s]) for n, m in maps.items() for s in SEEDS] + [(GT, s, None) for s in SEEDS]
    with ProcessPoolExecutor(workers, initializer=setup, initargs=(split,)) as ex:
        done = list(ex.map(ec.run_pose, jobs))
    by = {(d["name"], d["seed"]): d for d in done}
    names = list(maps) + [GT]
    boots = []
    for s in SEEDS:
        groups = {n: by_trajectory(by[(n, s)]["rot"], "rot_err_sym_deg") for n in names}
        boots.append(paired_bootstrap(groups, BASE, ec.N_BOOT, seed=s))
    pose = {}
    for n in names:
        tables = [by[(n, s)]["table"] for s in SEEDS]
        mean = {row: {k: float(np.mean([t[row][k] for t in tables if t[row][k] is not None])) for k in ec.POSE_KEYS}
                for row in ("all", "no08")}
        pose[n] = {"per_seed": tables, "mean": mean, "rot_vs_base": [b[n] for b in boots]}
    return pose


def export_cell_poses(split: str, records: list[dict], cell: str, map_paths: dict[int, Path] | None, out: Path) -> None:
    import export_poses as ep
    i = INPUTS[split]
    for s in SEEDS:
        ep.export(COMBO, i["det"], split, True, s, i["kpts"][s], str(out / "e4b" / cell), records=records,
                  class_map=None if map_paths is None else str(map_paths[s]))


def e4b_level(split: str, cells: list[str], out: Path, device: str, b: int) -> dict:
    from dronepose.forecast.config import load_config
    from dronepose.forecast.dataset import load_trajectories
    from dronepose.forecast.e4_eval import align_window_rows, degradation, pipeline_errors
    from dronepose.forecast.nets.device import pick_device
    from dronepose.forecast.windows import make_windows
    from e4_train import ARMS, e4_run_name

    cfg, dev = load_config(Path("configs/forecast.json")), pick_device(device)
    trajs = load_trajectories(Path("data/processed/forecast"), split)
    gt = make_windows(trajs, cfg.obs_len, cfg.pred_len, cfg.fps)
    pdir, runs = Path("data/processed/forecast_archive/e4"), Path("experiments/forecast/e4/runs")
    final, final08 = (json.loads((pdir / f"{n}.json").read_text()) for n in ("params_final", "params_08_final"))
    nets = {arm: [runs / e4_run_name(arm, "final", s) for s in SEEDS] for arm in ARMS}

    def errors(cell: str, cls: str):
        paths = [out / "e4b" / cell / "poses" / f"poses_{split}_s{k}_{cls}.jsonl" for k in SEEDS]
        err, _, idx, info, keys = pipeline_errors(trajs, paths, cfg, lambda s, m: final if m else final08,
                                                  lambda s: nets, dev, allow_test=split == "test", with_keys=True)
        print(f"E4b {cell} {cls}: {info}", flush=True)
        return err, idx, keys

    base = errors(BASE, "causal")
    rows = {f"{BASE} (classe vera)": errors(BASE, "gt")}
    rows.update({c: errors(c, "causal") for c in cells if c != BASE})
    res = {"base": BASE, "methods": METHODS, "cells": {}}
    for name, (err, idx, keys) in rows.items():
        ia, ib = align_window_rows(keys, base[2])
        w = gt.subset(base[1][ib])
        d = degradation({m: err[m][ia] for m in METHODS}, {m: base[0][m][ib] for m in METHODS}, w, b)
        res["cells"][name] = {"n_paired": int(len(ia)), "n_rows": int(len(keys)), "n_base_rows": int(len(base[2])),
                              "vs_base": {m: {"ade_1s": d[m]["ade_1s_e4b"], "ade_1s_base": d[m]["ade_1s_e4a"],
                                              "diff": d[m]["mean"], "ci": d[m]["ci"]} for m in METHODS}}
    return res


def _ci(v) -> str:
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"


def write_md(path: Path, res: dict) -> None:
    L = [f"# Classe del drone sul {res['split']} (una volta; parametri congelati)", "",
         "## Classe", "", "| Riga | Classe per frame (3 semi) | Sequenze giuste | Δ contro C1+V per seme |", "|---|---|---|---|"]
    for n, c in res["class"].items():
        L.append(f"| {n} | {np.mean(c['frame_acc']):.3f} ({', '.join(f'{x:.3f}' for x in c['frame_acc'])}) | "
                 f"{np.mean(c['seq_final_acc']):.3f} | {'; '.join(_ci(v) for v in c['vs_base'])} |")
    L += ["", "## Posa (ippe+lm+global, media dei 3 semi)", "",
          "| Riga | Rot. mediana | p90 | t camera | copertura | flip senza 08 | Δrot contro C1+V per seme |",
          "|---|---|---|---|---|---|---|"]
    for n, p in res["pose"].items():
        a, n8 = p["mean"]["all"], p["mean"]["no08"]
        L.append(f"| {n} | {a['median_rot_sym']:.3f}° | {a['p90_rot_sym']:.2f}° | {a['median_t_camera']:.4f} | "
                 f"{a['coverage']:.3f} | {n8['flip_rate']:.4f} | {'; '.join(_ci(v) for v in p['rot_vs_base'])} |")
    e = res["e4b"]
    L += ["", f"## E4b (ADE 1 s, m; differenze contro {e['base']} sulle stesse finestre)", "",
          "| Riga | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    first = next(iter(e["cells"].values()))["vs_base"]
    L.append(f"| {e['base']} | " + " | ".join(f"{first[m]['ade_1s_base']:.3f}" for m in METHODS) + " |")
    for n, c in e["cells"].items():
        v = c["vs_base"]
        L.append(f"| {n} | " + " | ".join(f"{v[m]['ade_1s']:.3f} ({v[m]['diff']:+.3f} [{v[m]['ci'][0]:+.3f}, "
                                          f"{v[m]['ci'][1]:+.3f}])" for m in METHODS) + " |")
    path.write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--smoke-val", action="store_true", help="run the same code on the VAL split (code check only)")
    ap.add_argument("--out", default=str(TEST_OUT))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bootstrap-b", type=int, default=10000)
    args = ap.parse_args(argv)
    split, out = ("val" if args.smoke_val else "test"), Path(args.out)
    try:
        guard(args.allow_test, out, args.smoke_val)          # before reading anything of the test
    except ValueError as e:
        sys.exit(f"test_class.py: {e}")
    missing = [str(p) for p in input_paths(split) if not p.exists()]
    if missing:
        sys.exit(f"test_class.py: missing inputs {missing}")
    out.mkdir(parents=True)                                   # marks the single use from here on

    t0 = time.time()
    records = setup(split)
    cells = frozen_cells()
    maps = {n: ec.build(c, p) for n, (c, p) in cells.items()}
    map_paths = {}
    for n in cells:
        map_paths[n] = {s: out / "maps" / f"map_{split}_s{s}_{n}.json" for s in SEEDS}
        for s, p in map_paths[n].items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(maps[n][s]))
    res = {"split": split, "params": {n: p for n, (_, p) in cells.items()}, "class": class_level(records, maps),
           "det_accuracy": ec.det_accuracy(ec._S)}
    print(f"class level done ({time.time() - t0:.0f} s)", flush=True)
    res["pose"] = pose_level(split, maps, args.workers)
    print(f"pose level done ({time.time() - t0:.0f} s)", flush=True)
    for n in cells:
        export_cell_poses(split, records, n, None if n == BASE else map_paths[n], out)
    res["e4b"] = e4b_level(split, list(cells), out, args.device, args.bootstrap_b)
    (out / "class_test.json").write_text(json.dumps(res, indent=1))
    write_md(out / "class_test.md", res)
    print(f"-> {out}/class_test.json (+ .md) in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
