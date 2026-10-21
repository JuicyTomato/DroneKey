"""Generate the physically simulated trajectory sets (plan step 3.1b): stratified at the freeze, importance-weighted.

Usage: python scripts/forecast/generate_trajectories.py [--out data/processed/forecast_gen] [--level 1]
           [--sets train val test active] [--aux-n 20000] [--aux-seed 0] [--workers 4]

Level 1 (default): an auxiliary natural set gives the cell edges and natural probabilities (edges.json), then each
set is generated with its own seed (train 1, val 2, test 3, active 4 = pilot keeps flying). Other levels
(--level 0 / 2, normally with --sets val): the same candidates accepted at level 1 are regenerated at that level
(same seed, same ids, same weights), so the mismatch curve compares the same samples; files get a _L<level> suffix.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from dronepose.forecast.config import load_generator_config
from dronepose.forecast.dataset import load_trajectories, save_trajectories
from dronepose.forecast.generator import generate, simulate_candidates
from dronepose.forecast.stratify import cell_edges, cell_key, natural_cell_probs, stratified_generate

SEEDS = {"train": 1, "val": 2, "test": 3, "active": 4, "select": 5, "train_big": 6}   # select, train_big: nets (3.3)
SIZES = {"train": 10000, "val": 2000, "test": 2000, "active": 2000, "select": 2000, "train_big": 50000}

def rejection_summary(rejected_by: dict, n_candidates: int) -> None:
    by_reason, by_mode = Counter(), Counter()
    for key, v in rejected_by.items():
        reason, mode, _ = key.split("/")
        by_reason[reason] += v["count"]
        by_mode[f"{reason}/{mode}"] += v["count"]
    print(f"  scartati {sum(by_reason.values())} su {n_candidates} candidati: {dict(by_reason)}; per modo: {dict(by_mode)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/forecast_gen")
    ap.add_argument("--config", default="configs/generator.json")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--sets", nargs="+", default=["train", "val", "test", "active"])
    ap.add_argument("--aux-n", type=int, default=20000)
    ap.add_argument("--aux-seed", type=int, default=0)
    ap.add_argument("--min-per-group", type=int, default=200)
    ap.add_argument("--max-attempts-factor", type=float, default=15.0, help="candidate budget = factor x set size")
    ap.add_argument("--workers", type=int, default=4, help="processes; the output does not depend on it")
    args = ap.parse_args()

    cfg = load_generator_config(Path(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    if args.level == 1:
        edges_path = out / "edges.json"
        if edges_path.exists():
            e = json.loads(edges_path.read_text())
            edges = {k: tuple(v) for k, v in e["edges"].items()}
            probs = {tuple(int(c) for c in k): p for k, p in e["probs"].items()}
            print(f"edges da {edges_path} (aux {e['aux_n']} campioni, seme {e['aux_seed']})")
        else:
            t0 = time.time()
            aux = generate(args.aux_n, seed=args.aux_seed, level=1, cfg=cfg, split="aux", workers=args.workers)
            meta = [t.meta for t in aux.trajs]
            edges = cell_edges(meta)
            probs = natural_cell_probs(meta, edges)
            edges_path.write_text(json.dumps({"edges": edges, "probs": {cell_key(c): p for c, p in probs.items()},
                                              "aux_n": args.aux_n, "aux_seed": args.aux_seed, "level": 1,
                                              "n_candidates": aux.n_candidates, "n_rejected": aux.n_rejected}, indent=1))
            print(f"aux: {args.aux_n} campioni naturali in {time.time() - t0:.0f} s, {len(probs)} celle occupate su 243")
            rejection_summary(aux.rejected_by, aux.n_candidates)
        for name in args.sets:
            t0 = time.time()
            n = SIZES[name]
            batch, report = stratified_generate(n, SEEDS[name], 1, cfg, edges, probs, args.min_per_group,
                                                int(args.max_attempts_factor * n), freeze=name != "active",
                                                workers=args.workers, split=name)
            save_trajectories(batch.trajs, out, name)
            report.update({"seed": SEEDS[name], "level": 1, "seconds": time.time() - t0,
                           "modes": dict(Counter(t.meta["mode"] for t in batch.trajs)),
                           "styles": dict(Counter(t.meta["style"] for t in batch.trajs)),
                           "flip": int(sum(t.meta["flip"] for t in batch.trajs))})
            (out / f"strat_report_{name}.json").write_text(json.dumps(report, indent=1))
            print(f"{name}: {report['n']} campioni ({report['attempts']} candidati, {report['seconds']:.0f} s); "
                  f"celle piene {report['filled']}/{report['occupiable']}, scoperte {len(report['uncovered'])}, "
                  f"rabbocco {report['topped_up']}; quota per livello ok: {report['quota_ok']} "
                  f"(min {min(min(x) for x in report['per_level'].values())}); flip {report['flip']}")
            rejection_summary(report["rejected_by"], report["attempts"])
            print("  quota di scarto per gruppo di ω: " + ", ".join(
                f"{g} {sum(v['rejected'].values()) / max(v['candidates'], 1):.1%}" for g, v in report["by_omega"].items()))
    else:
        for name in args.sets:
            t0 = time.time()
            base = load_trajectories(out, name)
            ids = np.array([t.meta["sample_id"] for t in base])
            by_id = {t.meta["sample_id"]: t.meta for t in base}
            trajs, would_reject = [], Counter()
            chunk = cfg["generator"]["chunk"]
            for k in range(0, len(ids), chunk):
                r = simulate_candidates(ids[k:k + chunk], SEEDS[name], args.level, cfg, freeze=name != "active",
                                        split=f"{name}_L{args.level}")
                for t, rej, why in zip(r["trajs"], r["reject"], r["reason"]):
                    t.meta["weight"] = by_id[t.meta["sample_id"]]["weight"]
                    t.meta["cell"] = by_id[t.meta["sample_id"]]["cell"]     # cell and weight of the level-1 sample
                    if rej:
                        would_reject[why] += 1
                    trajs.append(t)
            save_trajectories(trajs, out, f"{name}_L{args.level}")
            report = {"n": len(trajs), "level": args.level, "seed": SEEDS[name], "same_ids_as": f"{name} (level 1)",
                      "kept_although_implausible_at_this_level": dict(would_reject), "seconds": time.time() - t0}
            (out / f"strat_report_{name}_L{args.level}.json").write_text(json.dumps(report, indent=1))
            print(f"{name}_L{args.level}: {len(trajs)} campioni (stessi id del livello 1) in {report['seconds']:.0f} s; "
                  f"non plausibili a questo livello ma tenuti: {dict(would_reject)}")
    print(f"totale {time.time() - t_all:.0f} s")


if __name__ == "__main__":      # the worker processes re-import this file (spawn)
    main()
