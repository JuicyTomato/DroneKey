"""Geometric consistency audit of 6DroneSyn annotations (no images needed)."""
import json, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path("/Volumes/MakeMePasta/6DroneSyn")
KEYS = ["key1", "key2", "key3", "key4"]

recs = []
for anno_path in sorted(ROOT.glob("scene_*/*/*/*_anno.json")):
    if anno_path.name.startswith("._"):
        continue
    a = json.load(open(anno_path))
    cam = a["camera_parameters"]
    K = np.array(cam["intrinsic"]["camera_matrix_K"])
    cam_eul = np.array(cam["extrinsic"]["rotation_euler"])
    for fid, e in a.items():
        if fid == "camera_parameters":
            continue
        recs.append(dict(
            fid=fid, scene=e["seq_type"], drone=e["drone_type"], bg=e["background"], deg=e["degree"],
            frame=int(e["filename"][:4]), K=K, cam_eul=cam_eul,
            img_hw=e["image_size"], bbox=e["bbox"],
            k2=np.array([e["2DKey"][k] for k in KEYS], float),
            kw=np.array([e["3DKey(world)"][k] for k in KEYS], float),
            kc=np.array([e["3DKey(camera)"][k] for k in KEYS], float),
            tw=np.array(e["drone"]["3Dtranslation(world)"]), rw=np.array(e["drone"]["3Drotation(world)"]),
            tc=np.array(e["drone"]["3Dtranslation(camera)"]), rc=np.array(e["drone"]["3Drotation(camera)"]),
            c2=np.array(e["drone"]["2Dlocation"]),
        ))
print("records:", len(recs))

def pct(x, ps=(50, 95, 99, 100)):
    x = np.asarray(x)
    return " ".join(f"p{p}={np.percentile(x, p):.4g}" for p in ps)

# ---------- 1. basic 2D sanity ----------
H, W = 1080, 1920
out_img = [r["fid"] for r in recs if (r["k2"][:, 0] < 0).any() or (r["k2"][:, 0] >= W).any()
           or (r["k2"][:, 1] < 0).any() or (r["k2"][:, 1] >= H).any()]
nan = [r["fid"] for r in recs if not all(np.isfinite(r[k]).all() for k in ("k2", "kw", "kc", "tw", "rw", "tc", "rc"))]
def bb(r): b = r["bbox"]; return np.array([b["x_min"], b["y_min"], b["x_max"], b["y_max"]])
bad_bbox = [r["fid"] for r in recs if not (bb(r)[2] > bb(r)[0] and bb(r)[3] > bb(r)[1])]
# keypoint distance outside bbox (px), negative = inside
def outside(r):
    b = bb(r); k = r["k2"]
    dx = np.maximum(b[0] - k[:, 0], k[:, 0] - b[2]); dy = np.maximum(b[1] - k[:, 1], k[:, 1] - b[3])
    return np.maximum(dx, dy).max()
outs = np.array([outside(r) for r in recs])
print("\n[1] 2D sanity")
print(f"  kpts outside image: {len(out_img)}   NaN/inf: {len(nan)}   invalid bbox: {len(bad_bbox)}")
print(f"  max kpt distance outside bbox (px): {pct(outs)}  frames with >5px: {(outs > 5).sum()}")
bw = np.array([bb(r)[2] - bb(r)[0] for r in recs]); bh = np.array([bb(r)[3] - bb(r)[1] for r in recs])
print(f"  bbox width px: {pct(bw, (0, 5, 50, 95, 100))}")
print(f"  bbox height px: {pct(bh, (0, 5, 50, 95, 100))}")

# ---------- 2. reprojection: K * 3DKey(camera) vs 2DKey ----------
def proj(K, X):
    p = (K @ X.T).T
    return p[:, :2] / p[:, 2:3]
rep = np.array([np.linalg.norm(proj(r["K"], r["kc"]) - r["k2"], axis=1).max() for r in recs])
behind = sum((r["kc"][:, 2] <= 0).any() for r in recs)
print("\n[2] reprojection K·3DKey(camera) -> 2DKey  (max over 4 kpts, px)")
print(f"  {pct(rep)}   frames >1px: {(rep > 1).sum()}   kpts behind camera: {behind}")
repc = np.array([np.linalg.norm(proj(r["K"], r["tc"][None]) - r["c2"], axis=1)[0] for r in recs])
print(f"  drone center 3Dtranslation(camera) -> 2Dlocation: {pct(repc)}")

# ---------- 3. world <-> camera relation ----------
# fit similarity kc = s * R kw + t per camera orientation, report s and residual
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / Xc.var(0).sum()
    return s, R, my - s * R @ mx
by_cam = defaultdict(list)
for r in recs:
    by_cam[tuple(np.round(r["cam_eul"], 3))].append(r)
print("\n[3] similarity fit 3DKey(world) -> 3DKey(camera), per camera orientation")
for ce, rs in by_cam.items():
    X = np.concatenate([r["kw"] for r in rs]); Y = np.concatenate([r["kc"] for r in rs])
    s, R, t = umeyama(X, Y)
    res = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
    print(f"  cam_euler={ce}: n={len(rs)} scale={s:.5f} t={np.round(t, 5)} residual max={res.max():.2e}")
ratio_t = np.array([np.linalg.norm(r["tc"]) / np.linalg.norm(r["tw"]) for r in recs])
print(f"  |t_cam|/|t_world| ratio: {pct(ratio_t, (0, 50, 100))}")

# ---------- 4. rigidity + size per drone type ----------
print("\n[4] rigidity: 6 pairwise kpt distances per drone type (world | camera units)")
pairs = list(itertools.combinations(range(4), 2))
def pd(X): return np.array([np.linalg.norm(X[i] - X[j]) for i, j in pairs])
by_drone = defaultdict(list)
for r in recs:
    by_drone[r["drone"]].append(r)
for d, rs in sorted(by_drone.items()):
    Dw = np.array([pd(r["kw"]) for r in rs]); Dc = np.array([pd(r["kc"]) for r in rs])
    print(f"  drone {d}: n={len(rs)}")
    print(f"     world  mean={np.round(Dw.mean(0), 4)}  max std={Dw.std(0).max():.2e}")
    print(f"     camera mean={np.round(Dc.mean(0), 4)}  max std={Dc.std(0).max():.2e}")

# ---------- 5. Euler convention + body-frame model ----------
print("\n[5] Euler convention for 3Drotation(world): body model = R^T (kw - tw) should be constant")
convs = ["xyz", "XYZ", "zyx", "ZYX", "xzy", "XZY", "yxz", "YXZ", "zxy", "ZXY", "yzx", "YZX"]
for d, rs in sorted(by_drone.items()):
    sub = rs[:: max(1, len(rs) // 2000)]
    best = []
    for c in convs:
        M = np.array([Rot.from_euler(c, r["rw"]).as_matrix().T @ (r["kw"] - r["tw"]).T for r in sub])
        best.append((np.abs(M - M.mean(0)).max(), c, M.mean(0).T))
    best.sort(key=lambda x: x[0])
    err, c, model = best[0]
    print(f"  drone {d}: best conv={c} max dev={err:.2e} (2nd {best[1][1]}={best[1][0]:.2e})")
    print(f"     body model (world units):\n{np.round(model, 4)}")
    by_drone[d] = (rs, c, model)

# rotation range (how much does attitude actually vary?)
print("\n[5b] attitude range in world (deg), per scene")
by_scene = defaultdict(list)
for r in recs:
    by_scene[r["scene"]].append(np.degrees(r["rw"]))
for s, E in sorted(by_scene.items()):
    E = np.array(E)
    print(f"  scene {s}: roll[{E[:,0].min():6.1f},{E[:,0].max():6.1f}] pitch[{E[:,1].min():6.1f},{E[:,1].max():6.1f}] yaw[{E[:,2].min():7.1f},{E[:,2].max():7.1f}]")

# ---------- 6. camera-frame pose consistency ----------
print("\n[6] camera-frame pose: does (3Drotation(camera), 3Dtranslation(camera)) map body model to 3DKey(camera)?")
for d, (rs, c, model) in sorted(by_drone.items()):
    sub = rs[:: max(1, len(rs) // 1000)]
    s = np.mean([np.linalg.norm(r["kc"] - r["kc"].mean(0)) / np.linalg.norm(r["kw"] - r["kw"].mean(0)) for r in sub])
    results = []
    for cc in convs:
        err = [np.abs((s * Rot.from_euler(cc, r["rc"]).as_matrix() @ model.T).T + r["tc"] - r["kc"]).max() for r in sub]
        results.append((np.max(err), np.median(err), cc))
    results.sort()
    print(f"  drone {d}: scale world->cam={s:.4f}  best conv={results[0][2]} max err={results[0][0]:.3e} med={results[0][1]:.3e}")

# ---------- 7. PnP round trip ----------
import cv2
print("\n[7] PnP round trip: body model (camera units) + 2DKey + K -> pose, vs GT 3DKey(camera)")
for d, (rs, c, model) in sorted(by_drone.items()):
    sub = rs[:: max(1, len(rs) // 500)]
    s = 0.582 if True else 1
    obj = (model * np.mean([np.linalg.norm(r["kc"] - r["kc"].mean(0)) / np.linalg.norm(r["kw"] - r["kw"].mean(0)) for r in sub])).astype(np.float64)
    terr, rerr, amb = [], [], 0
    for r in sub:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, r["k2"].astype(np.float64), r["K"], None, flags=cv2.SOLVEPNP_IPPE)
        best = None
        for rv, tv in zip(rvecs, tvecs):
            Rm = cv2.Rodrigues(rv)[0]
            X = (Rm @ obj.T).T + tv.ravel()
            e = np.linalg.norm(X - r["kc"], axis=1).max()
            if best is None or e < best[0]:
                best = (e, np.linalg.norm(tv.ravel() - r["kc"].mean(0)))
        # IPPE solutions ordered by reproj error: was the first one the right one?
        Rm0 = cv2.Rodrigues(rvecs[0])[0]
        e0 = np.linalg.norm((Rm0 @ obj.T).T + tvecs[0].ravel() - r["kc"], axis=1).max()
        if e0 > best[0] + 1e-6:
            amb += 1
        terr.append(best[0])
    terr = np.array(terr)
    print(f"  drone {d}: n={len(sub)} best-of-2 3D kpt err (cam units) {pct(terr, (50, 95, 100))}  first-solution-wrong: {amb}/{len(sub)}")

# ---------- 8. temporal smoothness ----------
print("\n[8] temporal: per-sequence consecutive-frame jumps")
seqs = defaultdict(list)
for r in recs:
    seqs[(r["scene"], r["drone"], r["bg"])].append(r)
jumps2d, jumps3d, gaps, worst = [], [], 0, []
for key, rs in seqs.items():
    rs.sort(key=lambda r: r["frame"])
    fr = np.array([r["frame"] for r in rs])
    if not (np.diff(fr) == 1).all():
        gaps += 1
    for a, b in zip(rs[:-1], rs[1:]):
        j2 = np.linalg.norm(a["k2"] - b["k2"], axis=1).max()
        j3 = np.linalg.norm(a["tw"] - b["tw"])
        jumps2d.append(j2); jumps3d.append(j3); worst.append((j2, b["fid"]))
print(f"  sequences: {len(seqs)}   with frame-number gaps: {gaps}")
print(f"  2D kpt jump px: {pct(jumps2d, (50, 95, 99, 99.9, 100))}")
print(f"  3D translation jump (world units): {pct(jumps3d, (50, 95, 99, 99.9, 100))}")
worst.sort(reverse=True)
print("  worst 2D jumps:", [(round(j, 1), f) for j, f in worst[:8]])
seq_len = defaultdict(set)
for (s, d, b), rs in seqs.items():
    seq_len[s].add(len(rs))
print("  frames per sequence by scene:", dict(sorted((k, sorted(v)) for k, v in seq_len.items())))

# depth range
dep = np.array([r["tc"][2] for r in recs])
print(f"\n[9] depth (camera z, camera units): {pct(dep, (0, 5, 50, 95, 100))}")
