"""Why drone 08 (Tello) keeps flipping with YOLO keypoints, and what survives (val only, no GPU).

Three analyses, all printed:
  1. keypoint identity: share of frames where the best permutation of the 4 predicted keypoints is the identity,
     per drone and, for the 08, per view angle (0 = nose towards the camera) and per drone size in pixels;
  2. does the drone fly nose-first? angle between the body front axis and the velocity (train + val, moving frames);
  3. with the 'global' strategy, with and without the gravity gate: plain rotation error, rotation error up to the
     body symmetry of a square, up-axis error and translation error, for drone 08 and drone 01 (control).

Usage: python scripts/audit/drone08_probe.py [--keypoints experiments/keypoints/keypoints_val.jsonl]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.eval.metrics import symmetric_rotation_error_deg
from dronepose.pose.replay import load_keypoints, replay

ap = argparse.ArgumentParser()
ap.add_argument("--keypoints", default="experiments/keypoints/keypoints_val.jsonl")
args = ap.parse_args()
val = read_jsonl(Path("data/processed/canonical_val.jsonl"))
assert all(r["split"] == "val" for r in val)
train = read_jsonl(Path("data/processed/canonical_train.jsonl"))
by_id = {r["frame_id"]: r for r in val}

# ---- 1. identity of the keypoints -------------------------------------------------------------------------------
PERMS = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2), (1, 0, 3, 2), (0, 3, 2, 1), (3, 2, 1, 0), (2, 1, 0, 3)]
rows = []
for line in open(args.keypoints):
    r = json.loads(line)
    rec = by_id[r["frame_id"]]
    p, g = np.array(r["kpts_pred"])[:, :2], np.array(rec["kpts2d"])
    best = int(np.argmin([np.linalg.norm(p[list(pm)] - g, axis=1).mean() for pm in PERMS]))
    R, t = np.array(rec["R_gt"]), np.array(rec["t_gt"])
    view = np.degrees(np.arccos(np.clip(R[:, 0] @ (-t / np.linalg.norm(t)), -1, 1)))
    rows.append((rec["drone"], view, max(np.ptp(g[:, 0]), np.ptp(g[:, 1])), best == 0))
print("1. identity of the keypoints (share of frames with the identity as best permutation)")
for d in sorted({r[0] for r in rows}):
    x = [r for r in rows if r[0] == d]
    print(f"   drone {d}: extent {np.median([r[2] for r in x]):4.0f} px   identity ok {100 * np.mean([r[3] for r in x]):5.1f}%")
x = [r for r in rows if r[0] == "08"]
print("   drone 08 by view angle (0 = nose to the camera, 180 = away):")
for lo in range(0, 180, 30):
    s = [r for r in x if lo <= r[1] < lo + 30]
    if s:
        print(f"     {lo:3d}-{lo + 30:3d} deg: n={len(s):4d}  identity ok {100 * np.mean([r[3] for r in s]):5.1f}%")

# ---- 2. nose-first flight? ---------------------------------------------------------------------------------------
by_seq = collections.defaultdict(list)
for r in train + val:
    by_seq[r["frame_id"].rsplit("_", 1)[0]].append(r)
ang = collections.defaultdict(list)
for rs in by_seq.values():
    rs.sort(key=lambda r: r["frame_idx"])
    T, R = np.array([r["t_gt"] for r in rs]), np.array([r["R_gt"] for r in rs])
    for i in range(2, len(rs) - 2):
        v = T[i + 2] - T[i - 2]
        if np.linalg.norm(v) >= 0.02:
            ang[rs[0]["drone"]].append(np.degrees(np.arccos(np.clip(R[i][:, 0] @ v / np.linalg.norm(v), -1, 1))))
print("\n2. angle between the body front axis and the velocity (moving frames)")
for d in sorted(ang):
    a = np.array(ang[d])
    print(f"   drone {d}: median {np.median(a):5.1f} deg   within 45 deg {100 * np.mean(a < 45):5.1f}%")

# ---- 3. global strategy: plain vs symmetric error, up axis, translation ---------------------------------------------
kp = load_keypoints(Path(args.keypoints))
models = load_drone_models(Path("configs/drone_models.json"))
print("\n3. 'global' strategy on val (drone 08 vs control drone 01)")
for drone in ("08", "01"):
    sub = [r for r in val if r["drone"] == drone]
    for label, up in (("no gravity", None), ("gravity", [0.0, -1.0, 0.0])):
        res = replay(sub, kp, models, "global", up=up)
        e = []
        for key, rs in res.items():
            for r in rs:
                if r["R"] is None:
                    continue
                rec, R = by_id[r["frame_id"]], np.array(r["R"])
                Rg = np.array(rec["R_gt"])
                e.append((r["rot_err_deg"], symmetric_rotation_error_deg(R, Rg),
                          np.degrees(np.arccos(np.clip(R[:, 2] @ Rg[:, 2], -1, 1))),
                          abs(r["t"][2] - rec["t_gt"][2]) / rec["t_gt"][2]))
        a = np.array(e)
        print(f"   drone {drone} {label:10s}: poses {len(a)}/{len(sub)}  flip {100 * np.mean(a[:, 0] > 90):5.1f}%  "
              f"flip up to symmetry {100 * np.mean(a[:, 1] > 90):4.1f}%  median sym err {np.median(a[:, 1]):4.1f} deg  "
              f"up-axis median {np.median(a[:, 2]):4.1f} deg  depth err median {100 * np.median(a[:, 3]):4.2f}%")
