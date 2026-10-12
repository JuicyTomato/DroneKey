"""Is 3Drotation consistent with keypoint motion? Kabsch vs annotated rotation."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

ROOT = Path("/Volumes/MakeMePasta/6DroneSyn")
KEYS = ["key1", "key2", "key3", "key4"]
recs = []
for p in sorted(ROOT.glob("scene_*/*/*/*_anno.json")):
    if p.name.startswith("._"):
        continue
    a = json.load(open(p)); K = np.array(a["camera_parameters"]["intrinsic"]["camera_matrix_K"])
    for fid, e in a.items():
        if fid == "camera_parameters":
            continue
        recs.append(dict(fid=fid, scene=e["seq_type"], drone=e["drone_type"], K=K,
                         k2=np.array([e["2DKey"][k] for k in KEYS], float),
                         kw=np.array([e["3DKey(world)"][k] for k in KEYS], float),
                         kc=np.array([e["3DKey(camera)"][k] for k in KEYS], float),
                         tw=np.array(e["drone"]["3Dtranslation(world)"]), rw=np.array(e["drone"]["3Drotation(world)"]),
                         tc=np.array(e["drone"]["3Dtranslation(camera)"]), rc=np.array(e["drone"]["3Drotation(camera)"])))

def kabsch(A, B):  # R, t with B ~ R A + t
    ma, mb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, mb - R @ ma

def geod(R1, R2):
    return np.degrees(np.linalg.norm(Rot.from_matrix(R1.T @ R2).as_rotvec()))

by_drone = defaultdict(list)
for r in recs:
    by_drone[r["drone"]].append(r)

print("Per drone: R_kp = Kabsch(world kpts of frame ref -> frame i). Annotated R_ann = euler xyz(3Drotation(world)).")
print("If R_ann^T R_kp const -> right offset;  if R_kp R_ann^T const -> left offset. Spread in deg (max geodesic to mean).")
models = {}
for d, rs in sorted(by_drone.items()):
    # reference: frame with annotated rotation closest to identity
    ref = min(rs, key=lambda r: np.linalg.norm(r["rw"]))
    Rref = Rot.from_euler("xyz", ref["rw"]).as_matrix()
    # body model in world units, expressed in body frame using ref's annotated rotation
    model_w = (Rref.T @ (ref["kw"] - ref["tw"]).T).T
    right, left, ident, tres, cres = [], [], [], [], []
    for r in rs[::5]:
        Rk, tk = kabsch(ref["kw"], r["kw"])  # motion relative to ref
        Rk_abs = Rk @ Rref                    # absolute if ref's annotation were right
        Ra = Rot.from_euler("xyz", r["rw"]).as_matrix()
        right.append(Ra.T @ Rk_abs); left.append(Rk_abs @ Ra.T); ident.append(geod(Ra, Rk_abs))
        tres.append(np.linalg.norm(r["tw"] - r["kw"].mean(0)))
        cres.append(np.linalg.norm((Rk @ (ref["kw"]).T).T + tk - r["kw"], axis=1).max())
    def spread(Ms):
        mean = Rot.from_matrix(np.array(Ms)).mean().as_matrix()
        return max(geod(mean, M) for M in Ms), mean
    sr, Mr = spread(right); sl, Ml = spread(left)
    print(f"\n drone {d}: ref={ref['fid']} rw_ref(deg)={np.round(np.degrees(ref['rw']),1)}")
    print(f"   rigid Kabsch residual max={max(cres):.1e}  |tw - kpt centroid| max={max(tres):.4f}")
    print(f"   angle(R_ann, R_kp): median={np.median(ident):.2f} max={np.max(ident):.2f} deg")
    print(f"   right-offset spread={sr:.2f} deg  left-offset spread={sl:.2f} deg")
    if min(sr, sl) < 0.5:
        side = "right" if sr < sl else "left"
        M = Mr if side == "right" else Ml
        print(f"   -> CONSTANT {side} offset = {np.round(Rot.from_matrix(M).as_euler('xyz', degrees=True), 2)} deg (xyz)")
    models[d] = model_w

# rotation per scene for one bad drone vs good drone: does the error depend on scene?
print("\n angle(R_ann, R_kp) median per scene [drone01 | drone02]:")
for d in ("01", "02"):
    rs = by_drone[d]; ref = min(rs, key=lambda r: np.linalg.norm(r["rw"])); Rref = Rot.from_euler("xyz", ref["rw"]).as_matrix()
    per = defaultdict(list)
    for r in rs[::3]:
        Rk, _ = kabsch(ref["kw"], r["kw"])
        per[r["scene"]].append(geod(Rot.from_euler("xyz", r["rw"]).as_matrix(), Rk @ Rref))
    print("  ", d, {s: round(float(np.median(v)), 1) for s, v in sorted(per.items())})

# PnP round trip using rigid model from keypoints themselves (camera units)
print("\nPnP round trip (model = kpts of ref frame in camera units, centered):")
for d, rs in sorted(by_drone.items()):
    ref = rs[0]; obj = ref["kc"] - ref["kc"].mean(0)
    errs, wrong_first, nsol, fails = [], 0, [], 0
    for r in rs[::15]:
        n, rvecs, tvecs, rep = cv2.solvePnPGeneric(obj, r["k2"], r["K"], None, flags=cv2.SOLVEPNP_IPPE)
        if n == 0:
            fails += 1; continue
        nsol.append(n)
        e = [np.linalg.norm((cv2.Rodrigues(rv)[0] @ obj.T).T + tv.ravel() - r["kc"], axis=1).max() for rv, tv in zip(rvecs, tvecs)]
        errs.append(min(e)); wrong_first += int(e[0] > min(e) + 1e-4)
    errs = np.array(errs)
    planar = np.linalg.svd(obj)[1]
    print(f"  drone {d}: n={len(errs)} fails={fails} sol_count={np.bincount(nsol).tolist()} "
          f"3D err p50={np.median(errs):.2e} max={errs.max():.2e}  IPPE 1st-solution wrong={wrong_first}  "
          f"planarity (sv3/sv1)={planar[2]/planar[0]:.3f}")

dep = np.array([r["tc"][2] for r in recs])
print("\ndepth z camera units: p0=%.2f p5=%.2f p50=%.2f p95=%.2f p100=%.2f" % tuple(np.percentile(dep, [0, 5, 50, 95, 100])))
