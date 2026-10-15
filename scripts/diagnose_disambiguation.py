"""Diagnostics behind the temporal_reject investigation (2026-10-15). Reads val keypoints, no GPU.

  python scripts/diagnose_disambiguation.py trace   --sequence 14_02_07_000 --frames 262 300   # per-frame decisions
  python scripts/diagnose_disambiguation.py maxgap                                            # TemporalParams.max_gap grid
  python scripts/diagnose_disambiguation.py gravity                                           # gravity prior + gate alone

Gravity assumption (checked on GT only, val): camera-frame up is -y, GT body-up tilt from it has p50 11 deg,
p99 49 deg, max 54 deg. NOT guaranteed for real videos (camera may not be level).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from dronepose.common.geometry import geodesic_deg
from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.pipeline.skeleton import ippe_candidates
from dronepose.pose.disambiguation import TemporalParams
from dronepose.pose.replay import _by_sequence, load_keypoints, replay, strategy_table

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["trace", "maxgap", "gravity"])
ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
ap.add_argument("--keypoints", default="experiments/keypoints/keypoints_val.jsonl")
ap.add_argument("--models", default="configs/drone_models.json")
ap.add_argument("--sequence", default="14_02_07_000")
ap.add_argument("--frames", type=int, nargs=2, default=[262, 300])
ap.add_argument("--max-tilt", type=float, default=70.0)
args = ap.parse_args()

records = read_jsonl(Path(args.canonical))
assert all(r["split"] != "test" for r in records), "val only"
kp = load_keypoints(Path(args.keypoints))
models = load_drone_models(Path(args.models))
UP = np.array([0.0, -1.0, 0.0])                       # camera-frame up (y points down)
tilt = lambda R: float(np.degrees(np.arccos(np.clip(R[:, 2] @ UP, -1, 1))))   # R[:, 2] = body up (FLU z)

if args.mode == "trace":
    recs = [r for r in records if r["frame_id"].startswith(args.sequence + "_")]
    c = replay(recs, kp, models, "temporal_reject")[args.sequence]
    b = replay(recs, kp, models, "temporal")[args.sequence]
    print("idx rej reason sol ratio rot_err(c) oracle rot_err(b)")
    for r, rb in zip(c, b):
        if args.frames[0] <= r["frame_idx"] <= args.frames[1]:
            ratio = "-" if r["reproj_ratio"] is None else f"{r['reproj_ratio']:.2f}"
            err = "-" if r["rot_err_deg"] is None else f"{r['rot_err_deg']:.0f}"
            orc = "-" if r["oracle_rot_err_deg"] is None else f"{r['oracle_rot_err_deg']:.0f}"
            eb = "-" if rb["rot_err_deg"] is None else f"{rb['rot_err_deg']:.0f}"
            print(f"{r['frame_idx']:3d} {int(r['rejected'])}   {str(r['reject_reason']):8s} {r['solution_index']}  {ratio:>5} {err:>8} {orc:>6} {eb:>8}")

elif args.mode == "maxgap":
    print("max_gap  flip%  coverage%  flip% without drone 08  (temporal_reject, val)")
    for g in (5, 15, 30, 60, 1000):
        t = strategy_table(replay(records, kp, models, "temporal_reject", TemporalParams(max_gap=g)))
        no8 = np.mean([100 * v["flip_rate"] for d, v in t["by_drone"].items() if d != "08"])
        print(f"{g:7d}  {100 * t['all']['flip_rate']:5.2f}  {100 * t['all']['coverage']:8.1f}  {no8:8.2f}")

else:
    tilts = np.array([tilt(np.array(r["R_gt"])) for r in records[::10]])
    print("GT body-up tilt from camera up, percentiles 50/90/99/max:", np.round(np.percentile(tilts, [50, 90, 99, 100]), 1))
    wrong = implausible = 0
    stats = defaultdict(lambda: [0, 0, 0])             # drone -> [posed, flips, frames]
    for rs in _by_sequence(records).values():
        for r in rs:
            sols = ippe_candidates(r, kp[r["frame_id"]], models)
            R_gt = np.array(r["R_gt"])
            for s in sols:
                if geodesic_deg(s.R, R_gt) > 90:
                    wrong += 1
                    implausible += tilt(s.R) > 75
            stats[r["drone"]][2] += 1
            ok = [s for s in sols if tilt(s.R) <= args.max_tilt and s.reproj_px <= 20.0]
            if ok:
                stats[r["drone"]][0] += 1
                stats[r["drone"]][1] += geodesic_deg(ok[0].R, R_gt) > 90
    print(f"IPPE candidates with error > 90 deg: {wrong}; with tilt > 75 deg: {implausible} ({100 * implausible / max(wrong, 1):.0f}%)")
    print(f"gravity gate (tilt <= {args.max_tilt:g}, reproj <= 20 px) + lowest reprojection, no temporal info:")
    for d in sorted(stats):
        p, f, n = stats[d]
        print(f"  drone {d}: flip {100 * f / max(p, 1):5.2f}%  coverage {100 * p / n:5.1f}%")
