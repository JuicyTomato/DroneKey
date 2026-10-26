"""E3 grid: solver x refinement x strategy x drone class on saved keypoints; selection on val, report on test.

Usage:
  python scripts/eval_e3.py --split val|test [--allow-test] --det DETS.jsonl
      --seed s0=KPTS.jsonl [--seed s1=KPTS.jsonl ...] [--combos NAME,...] [--chosen NAME]
      [--compare-dronekeypp] [--workers N] --out OUT.json
NAME = solver[+lm]+strategy, e.g. ippe+global, sqpnp+lm+reproj (both classes, gt and causal, are run).
Without --combos all 64 combinations per seed are run; on val the selection rule (classe causal, mean over seeds)
then picks one (`choose_combo`). With --combos nothing is selected (report mode: pass the chosen one, the reference
ippe+global is always added). The test split needs --allow-test and never selects.
--compare-dronekeypp: MAE/MedAE of rotation and translation for --chosen and iterative+reproj (class causal).
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.eval.bootstrap import by_trajectory, paired_bootstrap
from dronepose.eval.e1 import e1_table, mean_std
from dronepose.eval.e3 import CLASSES, Combo, all_combos, choose_combo, euler_axis_errors, mae_medae, \
    reference, solver_timing
from dronepose.eval.units import WORLD_PER_CAMERA
from dronepose.keypoints.detections import class_causal_vote, load_detections
from dronepose.pose.replay import check_no_test, load_keypoints, replay
from dronepose.pose.solvers import SOLVERS

ROWS = ("all", "no08", "08")
SHOWN_STRATEGIES = ("reproj", "global")
AXES = ("yaw", "pitch", "roll")
REFERENCE_NAME = "ippe+global"
DK_COMPARE = "iterative+reproj"
# Numbers of the DroneKey++ paper (spec 2026-10-26-fase2-e3-archivio-design.md, section 4.6): (R MAE deg,
# R MedAE deg, t MAE m, t MedAE m). Their GT rotation is probably affected by the Euler bug, t units uncertain.
PAPER = {
    "their_yolov8_keypoint_pnp": {"rot_mae_deg": 79.25, "rot_medae_deg": 87.55, "t_mae_m": 0.391, "t_medae_m": 0.105},
    "dronekey": {"rot_mae_deg": 20.17, "rot_medae_deg": 21.28, "t_mae_m": 0.533, "t_medae_m": 0.370},
    "dronekey_pp": {"rot_mae_deg": 17.34, "rot_medae_deg": 17.10, "t_mae_m": 0.135, "t_medae_m": 0.242},
}


def parse_seed(spec: str) -> tuple[str, str]:
    """Parse `NAME=KPTS.jsonl` into (name, path)."""
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise ValueError(f"bad seed {spec!r}: expected NAME=KPTS.jsonl")
    return name, path


def select_combos(names: list[str] | None) -> list[Combo]:
    """All 64 combinations, or those whose name is in `names` (both classes); the reference is always kept."""
    allc = all_combos()
    if names is None:
        return allc
    known = {c.name for c in allc}
    bad = [n for n in names if n not in known]
    if bad:
        raise ValueError(f"unknown combos {bad}; expected solver[+lm]+strategy, e.g. ippe+global")
    keep = set(names) | {REFERENCE_NAME}
    return [c for c in allc if c.name in keep]


class Context:
    """Records, models, detections and per-seed keypoints, loaded once per process."""

    def __init__(self, seeds: list[tuple[str, str]], det: str, split: str, allow_test: bool,
                 records: list[dict] | None, canonical: str | None, models_path: str):
        if records is None:
            records = read_jsonl(Path(canonical or f"data/processed/canonical_{split}.jsonl"))
        check_no_test(records, allow_test)
        self.records = records
        self.models = load_drone_models(Path(models_path))
        dets = load_detections(Path(det))
        self.scores = {fid: d.conf for fid, d in dets.items()}
        self.model_drone = {"gt": None, "causal": class_causal_vote(records, dets)}
        self.kpts = {name: load_keypoints(Path(path)) for name, path in seeds}

    def replay_flat(self, seed: str, c: Combo) -> list[dict]:
        res = replay(self.records, self.kpts[seed], self.models, c.strategy, model_drone=self.model_drone[c.cls],
                     solver=c.solver, refine=c.refine)
        return [r for rs in res.values() for r in rs]

    def evaluate(self, seed: str, c: Combo) -> dict:
        flat = self.replay_flat(seed, c)
        table = e1_table(flat, self.records, self.kpts[seed], self.scores)
        return {"table": table, "rot": by_trajectory(flat, "rot_err_sym_deg"), "t": by_trajectory(flat, "t_err")}

    def timing(self, seed: str, solver: str, refine: bool) -> dict:
        return solver_timing(self.records, self.kpts[seed], self.models, solver, refine)


_CTX: Context | None = None


def _init_worker(*args) -> None:
    global _CTX
    _CTX = Context(*args)


def _work(seed: str, c: Combo) -> tuple[str, str, dict, float]:
    t0 = time.time()
    return seed, c.key, _CTX.evaluate(seed, c), time.time() - t0


def _boot_vs(groups: dict[str, dict[str, list[float]]], ref: str) -> dict[str, dict]:
    """Paired bootstrap of every name with poses against `ref`; names without poses are left out."""
    g = {n: v for n, v in groups.items() if v}
    if ref not in g:
        return {}
    return {n: {"diff": d, "lo": lo, "hi": hi} for n, (d, lo, hi) in paired_bootstrap(g, ref).items()}


def _flip_no08(table: dict) -> float | None:
    return table["no08"]["flip_rate"]


def select_on_val(seed_names: list[str], data: dict[str, dict[str, dict]], min_coverage: float = 0.95) -> dict:
    """Selection rule on class causal. `data[seed][combo name]` = {"table", "rot"} (by_trajectory groups of
    rot_err_sym_deg). Returns {"chosen", "best", "means", "boots"} or {"error": ...} if no combo has enough coverage.
    The bootstrap against the best combination is run per seed, only among combinations with enough coverage."""
    names = [n for n in data[seed_names[0]] if all(n in data[s] for s in seed_names)]
    means: dict[str, dict] = {}
    for n in names:
        rows = [data[s][n]["table"] for s in seed_names]
        vals = {"coverage": [t["all"]["coverage"] for t in rows],
                "median_rot_sym": [t["all"]["median_rot_sym"] for t in rows],
                "flip_no08": [_flip_no08(t) for t in rows]}
        if any(v is None for vs in vals.values() for v in vs):
            continue                                  # no pose in some seed or no frames without drone 08
        means[n] = {k: float(np.mean(vs)) for k, vs in vals.items()}
    ok = [n for n, m in means.items() if m["coverage"] >= min_coverage]
    if not ok:
        best_cov = max((m["coverage"] for m in means.values()), default=None)
        return {"error": f"no combination reaches coverage >= {min_coverage} (best {best_cov})", "means": means}
    best = min(ok, key=lambda n: means[n]["median_rot_sym"])
    boots: dict[str, list[tuple[float, float, float]]] = {n: [] for n in ok if n != best}
    boots_full: dict[str, dict[str, dict]] = {}
    for s in seed_names:
        bs = _boot_vs({n: data[s][n]["rot"] for n in ok}, best)
        boots_full[s] = bs
        for n in boots:
            b = bs[n]
            boots[n].append((b["diff"], b["lo"], b["hi"]))
    chosen = choose_combo(means, boots, min_coverage)
    return {"chosen": chosen, "best": best, "means": means, "boots_vs_best": boots_full}


def _fmt(v) -> str:
    return "  -  " if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def _ms(ms: dict, row: str, metric: str) -> str:
    m = ms[row][metric]
    return "-" if m["mean"] is None else f"{m['mean']:.4g}" + ("" if m["std"] is None else f"±{m['std']:.2g}")


def print_table(combos: list[Combo], summary: dict[str, dict], chosen: str | None) -> None:
    """Readable table: every ippe combination, and reproj/global of the other solvers; then the chosen one."""
    cols = (("coverage", "all", "coverage"), ("median_rot_sym", "all", "med_rot"), ("p90_rot_sym", "all", "p90_rot"),
            ("median_t_camera", "all", "med_t_cam"), ("flip_rate", "no08", "flip_no08"))
    print("\nmean ± std over seeds")
    for cls in CLASSES:
        shown = [c for c in combos if c.cls == cls and (c.solver == "ippe" or c.strategy in SHOWN_STRATEGIES)]
        if not shown:
            continue
        print(f"\nclass {cls}\n{'combination':24s} " + " ".join(f"{h:>14s}" for _, _, h in cols))
        for c in shown:
            ms = summary[c.key]
            print(f"{c.name:24s} " + " ".join(f"{_ms(ms, row, m):>14s}" for m, row, _ in cols), flush=True)
    if chosen:
        print(f"\nchosen: {chosen}")
        for cls in CLASSES:
            key = f"{chosen}|{cls}"
            if key in summary:
                ms = summary[key]
                print(f"  {cls:7s} " + "  ".join(f"{h}={_ms(ms, row, m)}" for m, row, h in cols))


def combo_by_name(name: str, cls: str) -> Combo:
    return next(c for c in all_combos() if c.name == name and c.cls == cls)


def compare_paper(ctx: Context, seed_names: list[str], chosen: str) -> dict:
    """MAE/MedAE of rotation (deg) and translation (camera units and x1.718 world) on posed frames, class causal."""
    out: dict = {"paper": PAPER, "world_per_camera": WORLD_PER_CAMERA, "ours": {}}
    for name in (chosen, DK_COMPARE):
        c = combo_by_name(name, "causal")
        per_seed = {}
        for s in seed_names:
            posed = [r for r in ctx.replay_flat(s, c) if r["rot_err_sym_deg"] is not None]
            rot = [r["rot_err_sym_deg"] for r in posed]
            t = [r["t_err"] for r in posed]
            tw = [x * WORLD_PER_CAMERA for x in t]
            rot_mae, rot_med = mae_medae(rot)
            t_mae, t_med = mae_medae(t)
            tw_mae, tw_med = mae_medae(tw)
            per_seed[s] = {"n_posed": len(posed), "rot_mae_deg": rot_mae, "rot_medae_deg": rot_med,
                           "t_camera_mae": t_mae, "t_camera_medae": t_med,
                           "t_world_mae": tw_mae, "t_world_medae": tw_med}
        keys = next(iter(per_seed.values())).keys()
        out["ours"][name] = {"per_seed": per_seed,
                             "mean": {k: float(np.mean([v[k] for v in per_seed.values()])) for k in keys}}
    print("\nDroneKey++ comparison (class causal, posed frames, mean over seeds)")
    print("  paper: " + "; ".join(f"{k} R {v['rot_mae_deg']}/{v['rot_medae_deg']} deg, t {v['t_mae_m']}/{v['t_medae_m']} m"
                                  for k, v in PAPER.items()))
    for name, v in out["ours"].items():
        m = v["mean"]
        print(f"  {name:20s} R MAE/MedAE {m['rot_mae_deg']:.2f}/{m['rot_medae_deg']:.2f} deg  "
              f"t camera {m['t_camera_mae']:.3f}/{m['t_camera_medae']:.3f}  "
              f"t world {m['t_world_mae']:.3f}/{m['t_world_medae']:.3f}")
    return out


def run(seeds: list[tuple[str, str]], det: str, split: str = "val", allow_test: bool = False,
        combos: list[str] | None = None, compare_dronekeypp: bool = False, out: str | None = None,
        chosen: str | None = None, workers: int = 1, records: list[dict] | None = None,
        canonical: str | None = None, models_path: str = "configs/drone_models.json") -> dict:
    """Run the grid on `split` and return the full result dict (also written to `out` if given).
    Test records without allow_test raise ValueError before anything is loaded."""
    if records is not None:
        check_no_test(records, allow_test)
    if not seeds or len({n for n, _ in seeds}) != len(seeds):
        raise ValueError(f"need at least one seed and unique seed names, got {[n for n, _ in seeds]}")
    if compare_dronekeypp and not chosen:
        raise ValueError("--compare-dronekeypp needs --chosen NAME")
    if chosen is not None and chosen not in {c.name for c in all_combos()}:
        raise ValueError(f"unknown --chosen {chosen!r}")
    names = None if combos is None else sorted(set(combos) | ({chosen} if chosen else set())
                                               | ({DK_COMPARE} if compare_dronekeypp else set()))
    grid = select_combos(names)
    if workers > 1 and records is not None:
        raise ValueError("--workers needs records read from the canonical file")
    ctx_args = (seeds, det, split, allow_test, records, canonical, models_path)
    seed_names = [n for n, _ in seeds]
    jobs = [(s, c) for s in seed_names for c in grid]
    t0 = time.time()
    data: dict[str, dict[str, dict]] = {s: {} for s in seed_names}     # seed -> combo key -> evaluation

    def done(i: int, seed: str, key: str, res: dict, dt: float) -> None:
        data[seed][key] = res
        print(f"[{i}/{len(jobs)}] {seed} {key:34s} {dt:6.1f}s  elapsed {time.time() - t0:7.0f}s", flush=True)

    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=ctx_args) as pool:
            ctx = Context(*ctx_args)                                     # main process: extras and timing
            futs = [pool.submit(_work, s, c) for s, c in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                done(i, *f.result())
    else:
        ctx = Context(*ctx_args)
        for i, (s, c) in enumerate(jobs, 1):
            t1 = time.time()
            done(i, s, c.key, ctx.evaluate(s, c), time.time() - t1)

    result: dict = {"split": split, "det": det, "seeds": dict(seeds), "combos": [c.key for c in grid],
                    "grid": {}, "mean_std": {}, "timing": {}}
    summary = {}
    for c in grid:
        ref = reference(c.cls)
        per_seed = {}
        for s in seed_names:
            groups = {k: data[s][k] for k in (c.key, ref.key)}
            entry = {"table": data[s][c.key]["table"], "bootstrap_vs_reference": {}}
            for label in ("rot", "t"):
                b = _boot_vs({k: v[label] for k, v in groups.items()}, ref.key)
                entry["bootstrap_vs_reference"][label] = b.get(c.key)
            per_seed[s] = entry
        result["grid"][c.key] = per_seed
        summary[c.key] = mean_std([per_seed[s]["table"] for s in seed_names])
    result["mean_std"] = summary
    selecting = split == "val" and combos is None
    if not selecting:
        print_table(grid, summary, chosen)

    print("\nsolver timing (class gt, ms per solve call)", flush=True)
    for s in seed_names:
        result["timing"][s] = {}
        for solver, refine in sorted({(c.solver, c.refine) for c in grid},
                                     key=lambda x: (SOLVERS.index(x[0]), x[1])):
            tm = ctx.timing(s, solver, refine)
            result["timing"][s][f"{solver}{'+lm' if refine else ''}"] = tm
            print(f"  {s} {solver}{'+lm' if refine else '':3s} n={tm['n']} median {tm['median_ms']:.3f}  p90 {tm['p90_ms']:.3f}")

    if selecting:
        causal = {s: {c.name: data[s][c.key] for c in grid if c.cls == "causal"} for s in seed_names}
        sel = select_on_val(seed_names, causal)
        result["selection"] = sel
        if "error" in sel:
            print(f"\nselection failed: {sel['error']}")
            print_table(grid, summary, None)
        else:
            picked = sel["chosen"]
            print(f"\nselection (val, class causal, mean over {len(seed_names)} seeds): best {sel['best']}, chosen {picked}")
            c = combo_by_name(picked, "causal")
            eulers, tables = [], {}
            for s in seed_names:
                flat = ctx.replay_flat(s, c)
                tables[s] = e1_table(flat, ctx.records, ctx.kpts[s], ctx.scores, breakdown=True)
                by_id = {r["frame_id"]: r for r in ctx.records}
                eulers += [euler_axis_errors(np.array(r["R"]), np.array(by_id[r["frame_id"]]["R_gt"]))
                           for r in flat if r["R"] is not None]
            med = np.median(np.array(eulers), axis=0)
            sel["chosen_extras"] = {"euler_median_deg": dict(zip(AXES, map(float, med))), "n_posed": len(eulers),
                                    "breakdown_tables": tables}
            print(f"  chosen per-axis median Euler error (deg, posed frames of all seeds): "
                  + "  ".join(f"{a}={v:.3f}" for a, v in zip(AXES, med)))
            print_table(grid, summary, picked)
    if compare_dronekeypp:
        result["compare_dronekeypp"] = compare_paper(ctx, seed_names, chosen)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--det", required=True, help="detections JSONL (class vote and AP score)")
    ap.add_argument("--seed", action="append", required=True, metavar="NAME=KPTS.jsonl")
    ap.add_argument("--combos", default=None, help="comma-separated names, e.g. ippe+global,epnp+reproj")
    ap.add_argument("--chosen", default=None, help="chosen combination name (needed by --compare-dronekeypp)")
    ap.add_argument("--compare-dronekeypp", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="processes over (seed, combination) jobs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run([parse_seed(s) for s in args.seed], args.det, args.split, args.allow_test,
        args.combos.split(",") if args.combos else None, args.compare_dronekeypp, args.out, args.chosen, args.workers)


if __name__ == "__main__":
    main()
