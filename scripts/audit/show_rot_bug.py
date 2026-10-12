"""Visualize rotation-field mismatch: project the drone's 3D model with the ANNOTATED pose vs the CORRECTED pose."""
import json, random
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
ROOT = Path("/Volumes/MakeMePasta/6DroneSyn"); KEYS = ["key1","key2","key3","key4"]
OUT = Path(__file__).parent / "vis"; OUT.mkdir(exist_ok=True)
random.seed(3)
E = lambda v: Rot.from_euler("xyz", v).as_matrix()
Rz = lambda deg: Rot.from_euler("z", deg, degrees=True).as_matrix()
data = defaultdict(list); K = None
for p in sorted(ROOT.glob("scene_*/*/*/*_anno.json")):
    if p.name.startswith("._"): continue
    a = json.load(open(p)); K = np.array(a["camera_parameters"]["intrinsic"]["camera_matrix_K"])
    for fid, e in a.items():
        if fid == "camera_parameters": continue
        e["_img"] = p.parent / e["path"].split("/")[-2] / e["filename"]; data[e["drone_type"]].append(e)
kc = lambda e: np.array([e["3DKey(camera)"][k] for k in KEYS])
tc = lambda e: np.array(e["drone"]["3Dtranslation(camera)"])
# body model per drone, built ONLY from degree=000 frames using the annotated rotation (there annotation and keypoints agree for all drones)
model = {}
for d, es in data.items():
    ms = [E(e["drone"]["3Drotation(camera)"]).T @ (kc(e) - tc(e)).T for e in es if e["degree"] == "000"]
    model[d] = np.mean(ms, 0).T
def proj(X): P = (K @ X.T); return (P[:2] / P[2]).T
def err_px(e, R): return np.linalg.norm(proj((R @ model[e["drone_type"]].T).T + tc(e)) - np.array([e["2DKey"][k] for k in KEYS]), axis=1).mean()

# numeric table: mean reprojection error (px) of model under annotated rotation, per drone x degree
print("Errore medio di riproiezione (px) usando la rotazione ANNOTATA, per drone x degree:")
print("drone   " + "".join(f"{g:>10}" for g in ["000","090","180","270"]))
for d in sorted(data):
    row = []
    for g in ["000","090","180","270"]:
        es = [e for e in data[d] if e["degree"] == g][::10]
        row.append(np.median([err_px(e, E(e["drone"]["3Drotation(camera)"])) for e in es]) if es else np.nan)
    print(f"  {d}   " + "".join(f"{v:10.1f}" for v in row))

def panel(e, R, title, ok_color):
    im = cv2.imread(str(e["_img"])); m = model[e["drone_type"]]; t = tc(e)
    gt = np.array([e["2DKey"][k] for k in KEYS]); pr = proj((R @ m.T).T + t)
    for ax, c in zip(np.eye(3) * 0.15, [(0,0,255),(0,255,0),(255,0,0)]):
        P = proj(np.stack([t, t + R @ ax])).astype(int); cv2.arrowedLine(im, tuple(P[0]), tuple(P[1]), c, 3, tipLength=0.2)
    for i in range(4):
        g = tuple(gt[i].astype(int)); q = tuple(pr[i].astype(int))
        cv2.line(im, g, q, (255,0,255), 2)
        cv2.circle(im, g, 9, (0,255,255), -1); cv2.putText(im, str(i+1), (g[0]+10, g[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
        cv2.drawMarker(im, q, (255,0,255), cv2.MARKER_TILTED_CROSS, 26, 4)
    b = e["bbox"]; cx, cy = int((b["x_min"]+b["x_max"])/2), int((b["y_min"]+b["y_max"])/2); s = int(max(b["width"], b["height"])*0.8)
    crop = cv2.resize(im[max(cy-s,0):cy+s, max(cx-s,0):cx+s], (420, 420))
    cv2.rectangle(crop, (0,0), (419,52), (0,0,0), -1)
    cv2.putText(crop, title, (6,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    err = np.linalg.norm(pr - gt, axis=1).mean()
    cv2.putText(crop, f"errore medio {err:.1f} px", (6,44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ok_color(err), 2)
    return crop
color = lambda err: (0,200,0) if err < 2 else (0,0,255)
cases = [("01","000"),("01","090"),("01","180"),("01","270"),("05","090"),("05","180"),("02","090"),("02","180"),("02","270")]
rows = []
for d, g in cases:
    e = random.choice([x for x in data[d] if x["degree"] == g])
    tag = f"drone {d} | degree {g} | {e['path'].split('/')[-2]}/{e['filename'][:4]}"
    raw = panel(e, E(e["drone"]["3Drotation(camera)"]), "ANNOTATA  " + tag, color)
    fixd = panel(e, E(e["drone"]["3Drotation(camera)"]) @ (np.eye(3) if d == "02" else Rz(int(g))), "CORRETTA  " + tag, color)
    rows.append(np.hstack([raw, np.full((420,8,3),255,np.uint8), fixd]))
for i, chunk in enumerate([rows[:4], rows[4:]]):
    cv2.imwrite(str(OUT / f"rotbug_{i+1}.jpg"), np.vstack([np.vstack([r, np.full((8, r.shape[1], 3), 255, np.uint8)]) for r in chunk]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("ok")
