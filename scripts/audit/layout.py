import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
ROOT = Path("/Volumes/MakeMePasta/6DroneSyn"); KEYS = ["key1","key2","key3","key4"]
E = lambda v: Rot.from_euler("xyz", v).as_matrix()
def fix(R, d, deg): return R if d == "02" else R @ Rot.from_euler("z", deg, degrees=True).as_matrix()
models = defaultdict(list); tight = []
for p in sorted(ROOT.glob("scene_*/*/*/*_anno.json")):
    if p.name.startswith("._"): continue
    a = json.load(open(p))
    for fid, e in list(a.items())[1::7]:
        if fid == "camera_parameters": continue
        d = e["drone_type"]; kc = np.array([e["3DKey(camera)"][k] for k in KEYS]); t = np.array(e["drone"]["3Dtranslation(camera)"])
        R = fix(E(e["drone"]["3Drotation(camera)"]), d, int(e["degree"]))
        models[d].append((R.T @ (kc - t).T).T)
        k2 = np.array([e["2DKey"][k] for k in KEYS]); b = e["bbox"]
        tight.append(((k2[:,0].max()-k2[:,0].min())*(k2[:,1].max()-k2[:,1].min()) / (b["width"]*b["height"]),
                      b["width"]/(k2[:,0].ptp() if hasattr(k2[:,0],'ptp') else np.ptp(k2[:,0])+1e-9)))
print("Body-frame keypoints (camera units ~ m), mean over frames; columns x y z. Sign pattern -> layout")
for d, M in sorted(models.items()):
    M = np.array(M); m = M.mean(0); sd = M.std(0).max()
    pat = ["".join("+" if v > 0 else "-" for v in row[:2]) for row in m]
    print(f" drone {d}: std={sd:.1e}  (x,y) signs per key1..4 = {pat}")
    print("   ", np.round(m, 3).tolist())
tight = np.array(tight)
print("\nkpt-hull area / bbox area: p5=%.2f p50=%.2f p95=%.2f" % tuple(np.percentile(tight[:,0],[5,50,95])))
print("bbox width / kpt x-extent: p5=%.2f p50=%.2f p95=%.2f" % tuple(np.percentile(tight[:,1],[5,50,95])))
