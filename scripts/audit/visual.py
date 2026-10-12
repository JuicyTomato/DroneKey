"""Contact sheet: random frames per drone type, keypoints numbered, bbox, body axes from CORRECTED camera rotation."""
import json, random
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
ROOT = Path("/Volumes/MakeMePasta/6DroneSyn"); KEYS = ["key1","key2","key3","key4"]
OUT = Path(__file__).parent / "vis"; OUT.mkdir(exist_ok=True)
random.seed(7)
E = lambda v: Rot.from_euler("xyz", v).as_matrix()
def fix(R, drone, deg):  # corrected body->frame rotation
    return R if drone == "02" else R @ Rot.from_euler("z", deg, degrees=True).as_matrix()
entries = defaultdict(list); camK = None
for p in sorted(ROOT.glob("scene_*/*/*/*_anno.json")):
    if p.name.startswith("._"): continue
    a = json.load(open(p)); camK = np.array(a["camera_parameters"]["intrinsic"]["camera_matrix_K"])
    for fid, e in a.items():
        if fid == "camera_parameters": continue
        sub = p.parent; img = sub / e["path"].split("/")[-2] / e["filename"]
        entries[e["drone_type"]].append((img, e))
# check corrected camera rotation vs keypoints for all frames (centered model from first deg-0 frame)
def kabsch(A,B):
    ma,mb=A.mean(0),B.mean(0); U,S,Vt=np.linalg.svd((B-mb).T@(A-ma)); D=np.diag([1,1,np.sign(np.linalg.det(U@Vt))]); return U@D@Vt
for d, es in sorted(entries.items()):
    ref = next(e for _, e in es if e["degree"] == "000")
    kc = lambda e: np.array([e["3DKey(camera)"][k] for k in KEYS])
    Rr = fix(E(ref["drone"]["3Drotation(camera)"]), d, 0)
    model = (Rr.T @ (kc(ref) - kc(ref).mean(0)).T).T
    errs = [np.degrees(np.linalg.norm(Rot.from_matrix(fix(E(e["drone"]["3Drotation(camera)"]), d, int(e["degree"])).T @ kabsch(model, kc(e))).as_rotvec())) for _, e in es[::3]]
    print(f"drone {d}: corrected CAMERA rotation vs keypoints: median {np.median(errs):.3f} max {np.max(errs):.3f} deg")
COL = [(0,0,255),(0,200,0),(255,0,0),(0,220,255)]  # BGR: key1 red, key2 green, key3 blue, key4 yellow
tiles = []
for d in sorted(entries):
    picks = random.sample(entries[d], 4)
    row = []
    for img_p, e in picks:
        im = cv2.imread(str(img_p))
        if im is None: row.append(np.zeros((360,360,3),np.uint8)); continue
        b = e["bbox"]; x0,y0,x1,y1 = map(int,[b["x_min"],b["y_min"],b["x_max"],b["y_max"]])
        cv2.rectangle(im,(x0,y0),(x1,y1),(255,255,255),1)
        # body axes from corrected rotation
        R = fix(E(e["drone"]["3Drotation(camera)"]), d, int(e["degree"])); t = np.array(e["drone"]["3Dtranslation(camera)"])
        L = 0.15
        for ax, c in zip(np.eye(3), [(0,0,255),(0,255,0),(255,0,0)]):
            P = camK @ np.stack([t, t + L*(R@ax)]).T; P = (P[:2]/P[2]).T.astype(int)
            cv2.arrowedLine(im, tuple(P[0]), tuple(P[1]), c, 2, tipLength=0.2)
        for i, k in enumerate(KEYS):
            x, y = e["2DKey"][k]
            cv2.circle(im, (int(x),int(y)), 5, COL[i], -1); cv2.circle(im, (int(x),int(y)), 6, (0,0,0), 1)
            cv2.putText(im, str(i+1), (int(x)+7,int(y)-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL[i], 2)
        cx, cy, s = (x0+x1)//2, (y0+y1)//2, int(max(x1-x0, y1-y0)*0.75)
        crop = im[max(cy-s,0):cy+s, max(cx-s,0):cx+s]
        crop = cv2.resize(crop, (360,360))
        cv2.putText(crop, f"d{d} {e['path'].split('/')[-2]}/{e['filename'][:4]}", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        row.append(crop)
    tiles.append(np.hstack(row))
sheet = np.vstack(tiles)
cv2.imwrite(str(OUT/"sheet_all.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
# split into two halves for readability
h = sheet.shape[0]//2
cv2.imwrite(str(OUT/"sheet_a.jpg"), sheet[:360*4], [cv2.IMWRITE_JPEG_QUALITY, 85])
cv2.imwrite(str(OUT/"sheet_b.jpg"), sheet[360*4:], [cv2.IMWRITE_JPEG_QUALITY, 85])
# one full frame
img_p, e = random.choice(entries["05"]); im = cv2.imread(str(img_p))
for i,k in enumerate(KEYS):
    x,y = e["2DKey"][k]; cv2.circle(im,(int(x),int(y)),6,COL[i],-1)
cv2.imwrite(str(OUT/"full.jpg"), cv2.resize(im,(960,540)))
print("ok", sheet.shape)
