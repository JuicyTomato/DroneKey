import json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path("/Volumes/MakeMePasta/6DroneSyn"); KEYS = ["key1","key2","key3","key4"]
seqs = defaultdict(list)
for p in sorted(ROOT.glob("scene_*/*/*/*_anno.json")):
    if p.name.startswith("._"): continue
    a = json.load(open(p))
    for fid, e in a.items():
        if fid == "camera_parameters": continue
        seqs[(e["seq_type"], e["drone_type"], e["background"])].append(
            (int(e["filename"][:4]), np.array([e["2DKey"][k] for k in KEYS]), np.array(e["drone"]["3Dtranslation(camera)"]), fid))
gaps = 0; j2 = []; j3 = []; acc = []; worst = []; per_scene = defaultdict(list); same_traj = defaultdict(list)
for key, rs in seqs.items():
    rs.sort(key=lambda r: r[0]); fr = np.array([r[0] for r in rs])
    gaps += int(not (np.diff(fr) == 1).all() or fr[0] != 1)
    T = np.array([r[2] for r in rs]); K2 = np.array([r[1] for r in rs])
    d2 = np.linalg.norm(np.diff(K2, axis=0), axis=2).max(1); d3 = np.linalg.norm(np.diff(T, axis=0), axis=1)
    a3 = np.linalg.norm(np.diff(T, 2, axis=0), axis=1)
    j2 += d2.tolist(); j3 += d3.tolist(); acc += a3.tolist(); per_scene[key[0]].append(np.linalg.norm(T[-1]-T[0]))
    worst += [(d2[i], rs[i+1][3]) for i in np.argsort(d2)[-2:]]
    same_traj[(key[0], key[2])].append(T)  # same scene+background: same trajectory across drones?
pc = lambda x, ps: " ".join(f"p{p}={np.percentile(x,p):.4g}" for p in ps)
print("sequences:", len(seqs), " with frame gaps / not starting at 1:", gaps)
print("2D kpt jump px:", pc(j2, (50, 99, 99.9, 100)))
print("3D translation step (cam units):", pc(j3, (50, 99, 99.9, 100)))
print("3D 2nd difference (accel proxy):", pc(acc, (50, 99, 99.9, 100)))
print("worst 2D jumps:", sorted(worst, reverse=True)[:6])
print("net displacement first->last per scene (median):", {s: round(float(np.median(v)), 2) for s, v in sorted(per_scene.items())})
# do different drones in same scene share the trajectory (camera frame)?
diffs = []
for s in sorted({k[0] for k in seqs}):
    Ts = [np.array([r[2] for r in sorted(rs, key=lambda r: r[0])]) for k, rs in seqs.items() if k[0] == s]
    L = min(len(t) for t in Ts); Ts = np.array([t[:L] for t in Ts])
    diffs.append((s, float(np.abs(Ts - Ts[0]).max())))
print("max |traj_i - traj_0| across the 21 sequences of a scene:", diffs)
