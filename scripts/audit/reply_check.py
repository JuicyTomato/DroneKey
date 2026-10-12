"""Verify the authors' email reply (2026-10-09) against the data.

(A) Their rotation fix R_anno @ Rz(degree), applied to ALL drones (incl. 02), vs ours (02 without Rz).
(B) Keypoint diagonal in world vs camera units against DJI specs.
(D) Their bbox formula: keypoint extent + pad = max(0.4w, 0.4h, 20px) per side, clipped to the image.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/Volumes/MakeMePasta/6DroneSyn")
KEYS = ["key1", "key2", "key3", "key4"]
Z_ORDER = {"01", "04", "05", "06", "08"}
SPECS_MM = {"01": 247, "02": 213, "03": 362, "04": 302, "05": 354, "06": 380.1}  # DJI motor diagonals


def euler_xyz(e):  # authors' code: Blender 'XYZ' -> R = Rz @ Ry @ Rx
    x, y, z = e
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


data = defaultdict(list)
for p in sorted(ROOT.glob("scene_*/*/*/*_anno.json")):
    if p.name.startswith("._"):
        continue
    a = json.load(open(p))
    for fid, e in list(a.items())[1::4]:
        if fid != "camera_parameters":
            data[e["drone_type"]].append(e)

kw = lambda e: np.array([e["3DKey(world)"][k] for k in KEYS])
tw = lambda e: np.array(e["drone"]["3Dtranslation(world)"])

print("(A) body-model spread across frames, m = R^T (k_world - t_world), max std in mm (world units)")
for d, es in sorted(data.items()):
    authors = lambda e: euler_xyz(e["drone"]["3Drotation(world)"]) @ rot_z(np.deg2rad(float(e["degree"])))
    ours = lambda e: euler_xyz(e["drone"]["3Drotation(world)"]) @ (np.eye(3) if d == "02" else rot_z(np.deg2rad(float(e["degree"]))))
    for name, f in [("authors (always Rz)", authors), ("ours (02 without Rz)", ours)]:
        M = np.array([f(e).T @ (kw(e) - tw(e)).T for e in es])
        print(f"   drone {d}  {name:22s} {1000 * M.std(0).max():9.3f} mm")

print("\n(B) propeller diagonal (mean of the two diagonals) vs DJI spec")
for d, es in sorted(data.items()):
    k = kw(es[0])
    k = k[[0, 1, 3, 2]] if d in Z_ORDER else k
    diag = (np.linalg.norm(k[0] - k[2]) + np.linalg.norm(k[1] - k[3])) / 2 * 1000
    sp = SPECS_MM.get(d)
    tail = f"ratio world {diag / sp:.2f} | camera {diag * 0.582 / sp:.2f}" if sp else "(Tello: spec uncertain, ~100-110 mm)"
    print(f"   drone {d}: world {diag:5.0f} mm | camera {diag * 0.582:5.0f} mm | {tail}")

print("\n(D) bbox formula check")
errs, pads = [], []
for es in data.values():
    for e in es:
        k = np.array([e["2DKey"][x] for x in KEYS])
        (x0, y0), (x1, y1) = k.min(0), k.max(0)
        w, h = x1 - x0, y1 - y0
        pad = max(0.4 * w, 0.4 * h, 20)
        pred = np.array([max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, 1919), min(y1 + pad, 1079)])
        b = e["bbox"]
        gt = np.array([b["x_min"], b["y_min"], b["x_max"], b["y_max"]])
        errs.append(np.abs(pred - gt).max())
        pads.append(((x0 - b["x_min"] + b["x_max"] - x1) / 2, (y0 - b["y_min"] + b["y_max"] - y1) / 2, w, h))
errs, pads = np.array(errs), np.array(pads)
print(f"   max abs error per box (px): p50={np.median(errs):.1f} p95={np.percentile(errs, 95):.1f} max={errs.max():.1f}  "
      f"boxes within 1 px: {100 * np.mean(errs < 1):.1f}%")
print(f"   actual pad: horiz/w p50={np.median(pads[:, 0] / pads[:, 2]):.3f}  vert/h p50={np.median(pads[:, 1] / pads[:, 3]):.3f}  "
      f"horiz/max(w,h) p50={np.median(pads[:, 0] / np.maximum(pads[:, 2], pads[:, 3])):.3f}  "
      f"vert/max(w,h) p50={np.median(pads[:, 1] / np.maximum(pads[:, 2], pads[:, 3])):.3f}")
