import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
ROOT = Path("/Volumes/MakeMePasta/6DroneSyn"); KEYS=["key1","key2","key3","key4"]
E=lambda v: Rot.from_euler("xyz",v).as_matrix()
data=defaultdict(list); K=None
for p in sorted(ROOT.glob("scene_*/*/*/*_anno.json")):
    if p.name.startswith("._"): continue
    a=json.load(open(p)); K=np.array(a["camera_parameters"]["intrinsic"]["camera_matrix_K"])
    for fid,e in a.items():
        if fid!="camera_parameters": data[e["drone_type"]].append(e)
kc=lambda e: np.array([e["3DKey(camera)"][k] for k in KEYS]); tc=lambda e: np.array(e["drone"]["3Dtranslation(camera)"])
def proj(X): P=K@X.T; return (P[:2]/P[2]).T
print("Errore medio riproiezione (px) con rotazione ANNOTATA, modello 3D costruito dai frame di un solo degree (riferimento)")
for d in ["01","05","02"]:
    for ref in ["000","180"]:
        m=np.mean([E(e["drone"]["3Drotation(camera)"]).T@(kc(e)-tc(e)).T for e in data[d] if e["degree"]==ref],0).T
        row=[]
        for g in ["000","090","180","270"]:
            es=[e for e in data[d] if e["degree"]==g][::10]
            row.append(np.median([np.linalg.norm(proj((E(e["drone"]["3Drotation(camera)"])@m.T).T+tc(e))-np.array([e["2DKey"][k] for k in KEYS]),axis=1).mean() for e in es]))
        print(f"  drone {d}, riferimento degree {ref}:  " + "  ".join(f"{g}:{v:6.1f}" for g,v in zip(["000","090","180","270"],row)))
