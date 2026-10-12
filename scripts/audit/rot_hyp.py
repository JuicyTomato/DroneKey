import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
ROOT = Path("/Volumes/MakeMePasta/6DroneSyn"); KEYS = ["key1","key2","key3","key4"]
recs = []
for p in sorted(ROOT.glob("scene_*/*/*/*_anno.json")):
    if p.name.startswith("._"): continue
    a = json.load(open(p)); ce = a["camera_parameters"]["extrinsic"]["rotation_euler"]
    for fid, e in a.items():
        if fid == "camera_parameters": continue
        recs.append(dict(fid=fid, scene=e["seq_type"], drone=e["drone_type"], deg=int(e["degree"]), camyaw=round(np.degrees(ce[2])),
            kw=np.array([e["3DKey(world)"][k] for k in KEYS]), kc=np.array([e["3DKey(camera)"][k] for k in KEYS]),
            tw=np.array(e["drone"]["3Dtranslation(world)"]), rw=np.array(e["drone"]["3Drotation(world)"]),
            tc=np.array(e["drone"]["3Dtranslation(camera)"]), rc=np.array(e["drone"]["3Drotation(camera)"])))
print("degree vs camera yaw pairs:", sorted({(r["deg"], r["camyaw"]) for r in recs}))
def kabsch(A,B):
    ma,mb=A.mean(0),B.mean(0); U,S,Vt=np.linalg.svd((B-mb).T@(A-ma)); D=np.diag([1,1,np.sign(np.linalg.det(U@Vt))]); return U@D@Vt
def geod(A,B): return np.degrees(np.linalg.norm(Rot.from_matrix(A.T@B).as_rotvec()))
E = lambda v: Rot.from_euler("xyz", v).as_matrix()
by = defaultdict(list)
for r in recs: by[r["drone"]].append(r)
for d, rs in sorted(by.items()):
    # body model from a deg-000 frame with ~identity rotation, assuming annotation right there
    cands = [r for r in rs if r["deg"]==0]
    ref = min(cands, key=lambda r: np.linalg.norm(r["rw"]))
    model = (E(ref["rw"]).T @ (ref["kw"]-ref["kw"].mean(0)).T).T
    res = defaultdict(lambda: defaultdict(list))
    for r in rs[::4]:
        Rt = kabsch(model, r["kw"])  # true absolute world rotation (body->world)
        Ra = E(r["rw"]); Rz = Rot.from_euler("z", r["deg"], degrees=True).as_matrix()
        res["as-is"][r["deg"]].append(geod(Ra, Rt))
        res["Rz(deg)@Ra"][r["deg"]].append(geod(Rz@Ra, Rt))
        res["Ra@Rz(deg)"][r["deg"]].append(geod(Ra@Rz, Rt))
        # camera-frame rotation field vs keypoints in camera frame
        Rtc = kabsch(model, r["kc"])
        res["cam as-is"][r["deg"]].append(geod(E(r["rc"]), Rtc))
    print(f"drone {d} (ref {ref['fid']}):")
    for name, dd in res.items():
        print(f"   {name:12s}", {k: f"{np.median(v):.2f}/{np.max(v):.2f}" for k, v in sorted(dd.items())})
