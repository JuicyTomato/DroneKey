"""Derive per-drone 3D model in FLU frame by inverting the rigid transform: m = R^T (k_cam - t_cam).
Compares raw annotated R vs corrected R; then maps Blender body axes to FLU and reports dimensions."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
ROOT = Path("/Volumes/MakeMePasta/6DroneSyn"); KEYS = ["key1","key2","key3","key4"]
Z_ORDER = {"01","04","05","06","08"}
E = lambda v: Rot.from_euler("xyz", v).as_matrix(); Rz = lambda g: Rot.from_euler("z", g, degrees=True).as_matrix()
raw, fix = defaultdict(list), defaultdict(list)
for p in sorted(ROOT.glob("scene_*/*/*/*_anno.json")):
    if p.name.startswith("._"): continue
    a = json.load(open(p))
    for fid, e in list(a.items())[1::3]:
        if fid == "camera_parameters": continue
        d = e["drone_type"]; k = np.array([e["3DKey(camera)"][x] for x in KEYS])
        if d in Z_ORDER: k = k[[0,1,3,2]]                       # canonical circular order
        t = np.array(e["drone"]["3Dtranslation(camera)"]); R = E(e["drone"]["3Drotation(camera)"])
        raw[d].append((R.T @ (k - t).T).T)
        fix[d].append(((R if d == "02" else R @ Rz(int(e["degree"]))).T @ (k - t).T).T)
B2FLU = np.array([[0,-1,0],[1,0,0],[0,0,1]])   # Blender body (x=left? y=back, z=up) -> FLU; checked below
print("Inversione rigida m = R^T (k - t): dispersione del modello tra frame (max std, unità camera)")
models = {}
for d in sorted(fix):
    Mr, Mf = np.array(raw[d]), np.array(fix[d])
    m = (B2FLU @ Mf.mean(0).T).T; models[d] = m
    print(f"  drone {d}: R annotata {Mr.std(0).max():.3f}   R corretta {Mf.std(0).max():.1e}")
print("\nControllo assi FLU (atteso: key1,key2 x>0 davanti; key1 y<0 destra; key2 y>0 sinistra):")
print("\nModello FLU [x avanti, y sinistra, z su] origine = origine oggetto Blender; dimensioni in unità camera")
print(f"{'drone':6}{'key1 FR':>24}{'key2 FL':>24}{'key3 RL':>24}{'key4 RR':>24}")
for d, m in models.items():
    print(f"  {d}  " + "".join(f"{str(np.round(r,3).tolist()):>24}" for r in m))
print(f"\n{'drone':6}{'larg.ant':>9}{'larg.post':>10}{'lungh.':>8}{'diag 1-3':>9}{'diag 2-4':>9}{'dz ant-post':>12}{'|orig-centr|':>13}")
for d, m in models.items():
    fw = np.linalg.norm(m[0]-m[1]); rw = np.linalg.norm(m[2]-m[3]); L = (m[0]+m[1])[0]/2 - (m[2]+m[3])[0]/2
    print(f"  {d}  {fw:9.3f}{rw:10.3f}{L:8.3f}{np.linalg.norm(m[0]-m[2]):9.3f}{np.linalg.norm(m[1]-m[3]):9.3f}{(m[0,2]+m[1,2]-m[2,2]-m[3,2])/2:12.3f}{np.linalg.norm(m.mean(0)):13.3f}")
