"""Replay the IPPE strategies on saved keypoints (val by default) and optionally scan a threshold grid.

Usage:
  python scripts/replay_disambiguation.py --keypoints experiments/keypoints/keypoints_val.jsonl
  python scripts/replay_disambiguation.py --keypoints ... --grid
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.pose.disambiguation import STRATEGIES, TemporalParams
from dronepose.pipeline.skeleton import summarize
from dronepose.pose.replay import check_no_test, gt_rotation_steps_deg, load_keypoints, replay, strategy_table

ap = argparse.ArgumentParser()
ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
ap.add_argument("--keypoints", required=True)
ap.add_argument("--models", default="configs/drone_models.json")
ap.add_argument("--sequence", default=None, help="optional frame_id prefix to replay a single sequence")
ap.add_argument("--grid", action="store_true", help="scan temporal_reject thresholds")
ap.add_argument("--up", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                help="camera-frame vertical for the global strategy gravity gate (default: off)")
ap.add_argument("--allow-test", action="store_true")
ap.add_argument("--out", default="experiments/replay")
args = ap.parse_args()

records = read_jsonl(Path(args.canonical))
if args.sequence:
    records = [r for r in records if r["frame_id"].startswith(args.sequence)]
check_no_test(records, args.allow_test)
kpts = load_keypoints(Path(args.keypoints))
models = load_drone_models(Path(args.models))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

steps = gt_rotation_steps_deg(records)
print(f"passo di rotazione GT per frame: mediana {np.median(steps):.2f}°, p99 {np.percentile(steps, 99):.2f}°, "
      f"max {steps.max():.2f}°")

rows = []
per_seq: dict[str, dict] = {}
for strategy in STRATEGIES:
    res = replay(records, kpts, models, strategy, up=args.up)
    table = strategy_table(res)
    a = table["all"]
    no08 = summarize([r for key, rs in res.items() if key.split("_")[1] != "08" for r in rs])
    for key, rs in res.items():
        s = summarize(rs)
        per_seq.setdefault(key, {"sequence": key, "drone": key.split("_")[1]}).update(
            {f"flip_{strategy}": s["flip_rate"], f"cov_{strategy}": s["coverage"]})
    rows.append({"strategy": strategy, **{k: a[k] for k in ("n_frames", "n_pose", "coverage", "flip_rate",
                 "oracle_flip_rate", "median_rot_err_deg", "p90_rot_err_deg", "median_t_err", "median_fde")},
                 "n_missing_kpts": table["n_missing_kpts"]})
    (out / f"table_{strategy}.json").write_text(json.dumps(table, indent=2))
    print(f"{strategy:16s} flip {100 * a['flip_rate']:5.2f}%  copertura {100 * a['coverage']:5.1f}%  "
          f"oracolo {100 * a['oracle_flip_rate']:5.2f}%  rot mediana {a['median_rot_err_deg']:.2f}°  "
          f"p90 {a['p90_rot_err_deg']:.2f}°  | senza drone 08: flip {100 * no08['flip_rate']:5.2f}%  "
          f"copertura {100 * no08['coverage']:5.1f}%")
with open(out / "per_sequence.csv", "w", newline="") as f:
    seq_rows = [per_seq[k] for k in sorted(per_seq)]
    w = csv.DictWriter(f, fieldnames=list(seq_rows[0]))
    w.writeheader()
    w.writerows(seq_rows)
if "global" in STRATEGIES:
    worse = [(r["sequence"], r["drone"], 100 * (r["flip_global"] - min(r["flip_reproj"], r["flip_temporal"])))
             for r in seq_rows if None not in (r["flip_global"], r["flip_reproj"], r["flip_temporal"])
             and r["flip_global"] - min(r["flip_reproj"], r["flip_temporal"]) > 0.02]
    for label, group in (("droni 01-06 (criterio)", [w for w in worse if w[1] != "08"]),
                         ("drone 08 (solo informativo: confusione di identità)", [w for w in worse if w[1] == "08"])):
        print(f"sequenze con global > min(a, b) + 2 punti di flip, {label}: {len(group)}"
              + ("  " + ", ".join(f"{k} (+{d:.1f})" for k, _, d in group) if group else ""))
with open(out / "strategies.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

if args.grid:
    grid = []
    for ratio, jump, rep in itertools.product((2.0, 3.0, 5.0), (10.0, 20.0, 30.0, 45.0), (10.0, 20.0, 40.0)):
        p = TemporalParams(reanchor_ratio=ratio, max_jump_deg=jump, max_reproj_px=rep)
        a = strategy_table(replay(records, kpts, models, "temporal_reject", p))["all"]
        grid.append({"reanchor_ratio": ratio, "max_jump_deg": jump, "max_reproj_px": rep,
                     "flip_rate": a["flip_rate"], "coverage": a["coverage"], "p90_rot_err_deg": a["p90_rot_err_deg"]})
    grid.sort(key=lambda g: (g["flip_rate"], -g["coverage"]))
    with open(out / "grid_temporal_reject.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0]))
        w.writeheader()
        w.writerows(grid)
    print("migliori 5 (flip più basso, poi copertura più alta):")
    for g in grid[:5]:
        print("  ", g)
