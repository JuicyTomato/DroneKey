"""Tune the two free parameters of the global disambiguation (reject_cost, z_cap) on the val split only,
with leave-one-scene-out: choose on two scenes, measure on the third. Needs saved keypoints (no GPU).

Usage: python scripts/tune_global.py --keypoints experiments/keypoints/keypoints_val.jsonl [--out experiments/tune_global]
Grid extended to reject_cost 48 after the final review. Rule (plan): among combinations with coverage >= 95% (drone 08 excluded from the choice), the lowest flip rate.
"""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.pose.global_disambiguation import GlobalParams
from dronepose.pose.replay import check_no_test, load_keypoints, replay
from dronepose.pose.tuning import add_counts, counts_from_results, scene_of, select_best

REJECT_COSTS = (3.0, 6.0, 12.0, 24.0, 48.0)
Z_CAPS = (4.0, 6.0, 8.0, 12.0)

ap = argparse.ArgumentParser()
ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
ap.add_argument("--keypoints", required=True)
ap.add_argument("--models", default="configs/drone_models.json")
ap.add_argument("--min-coverage", type=float, default=0.95)
ap.add_argument("--allow-test", action="store_true")
ap.add_argument("--out", default="experiments/tune_global")
args = ap.parse_args()

records = read_jsonl(Path(args.canonical))
check_no_test(records, args.allow_test)
kpts = load_keypoints(Path(args.keypoints))
models = load_drone_models(Path(args.models))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

by_scene = defaultdict(list)
for r in records:
    by_scene[scene_of(r["frame_id"])].append(r)
scenes = sorted(by_scene)
print("scene di val:", scenes)

counts = {}                                  # (reject_cost, z_cap, scene) -> {'all': Counts, 'no08': Counts}
for rc, zc in itertools.product(REJECT_COSTS, Z_CAPS):
    for sc in scenes:
        res = replay(by_scene[sc], kpts, models, "global", GlobalParams(reject_cost=rc, z_cap=zc))
        counts[(rc, zc, sc)] = counts_from_results(res)
    print(f"  reject_cost={rc:4g} z_cap={zc:4g} fatto")

with open(out / "grid_counts.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["reject_cost", "z_cap", "scene", "subset", "flips", "posed", "frames"])
    for (rc, zc, sc), c in sorted(counts.items()):
        for subset, cc in c.items():
            w.writerow([rc, zc, sc, subset, cc.flips, cc.posed, cc.frames])


def rows_for(train_scenes):
    return [{"params": {"reject_cost": rc, "z_cap": zc},
             "counts": add_counts([counts[(rc, zc, sc)]["no08"] for sc in train_scenes])}
            for rc, zc in itertools.product(REJECT_COSTS, Z_CAPS)]


folds = []
print("\nleave-one-scene-out (scelta su due scene, misura sulla terza; flip/copertura senza il drone 08 e con):")
for held in scenes:
    best, ok = select_best(rows_for([s for s in scenes if s != held]), args.min_coverage)
    p = best["params"]
    h = counts[(p["reject_cost"], p["z_cap"], held)]
    folds.append({"held_out": held, **p, "coverage_floor_met": ok})
    print(f"  fuori {held}: scelti reject_cost={p['reject_cost']:g} z_cap={p['z_cap']:g}  ->  "
          f"senza 08 flip {100 * h['no08'].flip_rate:5.2f}% cop {100 * h['no08'].coverage:5.1f}%  |  "
          f"con 08 flip {100 * h['all'].flip_rate:5.2f}% cop {100 * h['all'].coverage:5.1f}%")

final, ok = select_best(rows_for(scenes), args.min_coverage)
p = final["params"]
tot = {k: add_counts([counts[(p["reject_cost"], p["z_cap"], sc)][k] for sc in scenes]) for k in ("all", "no08")}
stable = len({(f["reject_cost"], f["z_cap"]) for f in folds}) == 1
print(f"\nscelta finale su tutta la val: reject_cost={p['reject_cost']:g} z_cap={p['z_cap']:g} "
      f"(copertura minima rispettata: {ok}; stessa scelta nei {len(folds)} giri: {stable})")
print(f"  senza 08: flip {100 * tot['no08'].flip_rate:.2f}%  copertura {100 * tot['no08'].coverage:.1f}%")
print(f"  con 08:   flip {100 * tot['all'].flip_rate:.2f}%  copertura {100 * tot['all'].coverage:.1f}%")
(out / "final.json").write_text(json.dumps({**p, "coverage_floor_met": ok, "stable_across_folds": stable,
                                           "folds": folds}, indent=2))
