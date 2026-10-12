"""Pilot: per drone, frames with keypoints in CANONICAL circular order, to let a human mark the front (gimbal) side."""
import json, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path("/Volumes/MakeMePasta/6DroneSyn"); KEYS = ["key1","key2","key3","key4"]
OUT = Path(__file__).parent / "vis"; OUT.mkdir(exist_ok=True)
DRONES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["01","02","08"]
Z_ORDER = {"01","04","05","06","08"}             # swap key3<->key4 to get circular order
def canon(d, pts): return pts[[0,1,3,2]] if d in Z_ORDER else pts
COL = [(0,0,255),(0,200,0),(255,80,0),(0,220,255)]  # BGR: 1 red, 2 green, 3 blue, 4 yellow
SIDES = [(0,1,"A"),(1,2,"B"),(2,3,"C"),(3,0,"D")]
rows = []
for d in DRONES:
    cands = []
    for p in sorted(ROOT.glob(f"scene_*/*/*_{d}/*_anno.json")):
        if p.name.startswith("._"): continue
        a = json.load(open(p))
        for fid, e in list(a.items())[1::5]:
            if fid == "camera_parameters": continue
            k = canon(d, np.array([e["2DKey"][x] for x in KEYS]))
            area = cv2.contourArea(k.astype(np.float32)) if cv2.isContourConvex(k.astype(np.float32).reshape(-1,1,2)) else 0
            area = area / (e["bbox"]["width"] * e["bbox"]["height"]) * min(e["bbox"]["width"], 500)  # favour top/bottom views of decent size
            cands.append((area,p.parent / e["path"].split("/")[-2] / e["filename"], e, k))
    cands.sort(key=lambda c: -c[0])
    # two well-spread views from different scenes
    picks, seen = [], set()
    for c in cands:
        sc = c[2]["seq_type"]
        if sc in seen: continue
        picks.append(c); seen.add(sc)
        if len(picks) == 2: break
    tiles = []
    for area, img_p, e, k in picks:
        im = cv2.imread(str(img_p))
        for i, j, lab in SIDES:  # label each side of the propeller quadrilateral
            a_, b_ = k[i], k[j]; m = ((a_ + b_) / 2).astype(int)
            cv2.line(im, tuple(a_.astype(int)), tuple(b_.astype(int)), (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(im, lab, (m[0]-10, m[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 6)
            cv2.putText(im, lab, (m[0]-10, m[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        for i in range(4):
            x, y = k[i].astype(int)
            cv2.circle(im, (x,y), 10, COL[i], -1); cv2.circle(im, (x,y), 11, (0,0,0), 2)
            cv2.putText(im, str(i+1), (x+12, y-12), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,0), 6)
            cv2.putText(im, str(i+1), (x+12, y-12), cv2.FONT_HERSHEY_SIMPLEX, 1.4, COL[i], 3)
        b = e["bbox"]; cx, cy = int((b["x_min"]+b["x_max"])/2), int((b["y_min"]+b["y_max"])/2)
        s = int(max(np.ptp(k[:,0]), np.ptp(k[:,1])) * 0.85)
        crop = cv2.resize(im[max(cy-s,0):cy+s, max(cx-s,0):cx+s], (560,560))
        cv2.rectangle(crop, (0,0), (559,30), (0,0,0), -1)
        cv2.putText(crop, f"DRONE {d} | {e['path'].split('/')[-2]}/{e['filename'][:4]}", (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        tiles.append(crop)
    rows.append(np.hstack([tiles[0], np.full((560,8,3),255,np.uint8), tiles[1]]))
sheet = np.vstack([np.vstack([r, np.full((8, r.shape[1], 3), 255, np.uint8)]) for r in rows])
name = OUT / f"front_{'_'.join(DRONES)}.jpg"; cv2.imwrite(str(name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(name)
