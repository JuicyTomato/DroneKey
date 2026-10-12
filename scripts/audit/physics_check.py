"""Is attitude physically linked to acceleration (quadrotor: a_horiz ≈ g * tan(tilt))? Or keyframed animation?"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
ROOT = Path("/Volumes/MakeMePasta/6DroneSyn"); KEYS = ["key1","key2","key3","key4"]
E = lambda v: Rot.from_euler("xyz", v).as_matrix(); Rz = lambda g: Rot.from_euler("z", g, degrees=True).as_matrix()
DT, G = 1/30, 9.81
B2FLU = np.array([[0,-1,0],[1,0,0],[0,0,1]])
seqs = {}
for p in sorted(ROOT.glob("scene_*/*/*/*_anno.json")):
    if p.name.startswith("._"): continue
    a = json.load(open(p)); es = [e for k, e in a.items() if k != "camera_parameters"]
    bg = es[0]["background"]; es = sorted([e for e in es if e["background"] == bg], key=lambda e: e["filename"])
    d = es[0]["drone_type"]
    P = np.array([np.mean([e["3DKey(world)"][k] for k in KEYS], 0) for e in es]) * 0.582   # world z-up, camera units
    U = np.array([((E(e["drone"]["3Drotation(world)"]) @ (np.eye(3) if d == "02" else Rz(int(e["degree"])))) @ B2FLU.T)[:, 2] for e in es])  # body up axis in world
    seqs[(es[0]["seq_type"], d)] = (P, U)
rows = defaultdict(list); speeds = defaultdict(list)
for (sc, d), (P, U) in seqs.items():
    V = np.gradient(P, DT, axis=0); A = np.gradient(V, DT, axis=0)
    a_h = A[2:-2, :2]; pred = G * U[2:-2, :2] / U[2:-2, 2:3]        # quadrotor model, hover thrust ~ g
    tilt = np.degrees(np.arccos(np.clip(U[:, 2], -1, 1)))
    m = np.linalg.norm(a_h, axis=1) > 0.3                           # only where it actually accelerates
    cos = np.sum(a_h * pred, 1) / (np.linalg.norm(a_h, axis=1) * np.linalg.norm(pred, axis=1) + 1e-9)
    rows[sc].append((np.median(cos[m]) if m.any() else np.nan, np.median(np.linalg.norm(a_h, axis=1)), np.median(tilt), np.max(tilt), m.mean()))
    speeds[sc].append((np.median(np.linalg.norm(V, axis=1)), np.max(np.linalg.norm(V, axis=1)), np.max(np.linalg.norm(A, axis=1))))
print("scene | cos(a_oriz, direzione inclinazione) med | |a_oriz| med m/s² | tilt med / max ° | % frame con a>0.3 | velocità med/max m/s | acc max m/s²")
for sc in sorted(rows):
    r = np.array(rows[sc]); v = np.array(speeds[sc])
    print(f"  {sc}  {np.nanmedian(r[:,0]):+.2f}   {np.median(r[:,1]):6.2f}   {np.median(r[:,2]):5.1f} / {np.max(r[:,3]):6.1f}   {100*np.mean(r[:,4]):5.0f}%   {np.median(v[:,0]):.2f} / {np.max(v[:,1]):.2f}   {np.max(v[:,2]):6.1f}")
