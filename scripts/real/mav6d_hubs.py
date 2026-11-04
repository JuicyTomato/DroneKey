"""Triangulate the 4 propeller hubs of the Mavic 2 in its VICON body frame from the clicks of annotate_hubs.py, and
correct the authors' VICON -> camera transform on the same clicks.

  .venv/bin/python scripts/real/mav6d_hubs.py [--ann experiments/real/mav6d/hub_annotations.json]
      [--out-dir experiments/real/mav6d]

1. Each hub is a fixed point of the body: the rays of its clicks through the known poses meet at it (linear least
   squares, then Huber refinement in pixels, 3 px), with the authors' calibration.
2. A frame where at least two clicks fit another hub better than their own by a factor 2 has its hubs in the wrong
   order (an annotation slip, e.g. front and back swapped): it is excluded and reported.
3. Joint refinement of the 4 hubs and of a rigid correction of VICON -> camera (rotation + translation), Huber 3 px.
   The clicks show a rigid offset per frame (all 4 hubs moved together, ~11 px median) that does not follow the drone
   velocity (not a time offset): an extrinsic error; with the correction the error falls to the annotation level.
Frames marked "holdout" are used in no step: their clicks measure the error of the final keypoints.
Writes OUT/hub_model.json (hubs in the body frame, metres, key1-key4; FLU model centred on their centroid; refined
cam_from_vicon) and OUT/hub_check.json (errors per click with both calibrations, excluded frames, geometry).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
from dronepose.data.mav6d import CAM_FROM_VICON, project, read_label, undistort  # noqa: E402

HUBER_PX = 3.0


def triangulate(obs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """obs: (R, t, click px). Linear rays intersection, then Huber refinement in pixels."""
    A, b = [], []
    for R, t, px in obs:
        x, y = undistort(px[None])[0]
        for c, row in ((x, 0), (y, 1)):
            A.append(c * R[2] - R[row])
            b.append(t[row] - c * t[2])
    X0 = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    res = lambda X: np.concatenate([project(X[None], R, t)[0] - px for R, t, px in obs])   # noqa: E731
    return least_squares(res, X0, loss="huber", f_scale=HUBER_PX).x


def corrected(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    """cam_from_vicon after the rigid correction x = (rotation vector, translation m), applied on the camera side."""
    dR = Rotation.from_rotvec(x[:3]).as_matrix()
    return np.hstack([dR @ T[:, :3], (dR @ T[:, 3] + x[3:6])[:, None]])


def click_errors(ann: dict, hubs: np.ndarray, T: np.ndarray, root: Path) -> list[dict]:
    rows = []
    for f, a in ann.items():
        R, t = read_label(root / "labels" / f"{f}.txt", T)
        proj = project(hubs, R, t)
        for j, c in enumerate(a["hubs"]):
            if c is not None:
                e = np.linalg.norm(proj - np.array(c), axis=1)
                rows.append({"frame": f, "use": a["use"], "hub": j + 1, "err_px": float(e[j]),
                             "best_other": int(np.argmin(e)) + 1, "dist_m": float(np.linalg.norm(t)),
                             "swap": bool(np.argmin(e) != j and e.min() < 0.5 * e[j])})
    return rows


def summary(rows: list[dict], holdout: bool) -> dict:
    e = np.array([r["err_px"] for r in rows if (r["use"] == "holdout") == holdout])
    return {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "max": float(e.max()), "n": len(e)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/Volumes/MakeMePasta/MAV6D/mavic2")
    ap.add_argument("--ann", default="experiments/real/mav6d/hub_annotations.json")
    ap.add_argument("--out-dir", default="experiments/real/mav6d")
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out_dir)
    ann = {f: a for f, a in json.loads(Path(args.ann).read_text()).items() if not a["skipped"]}
    pose = {f: read_label(root / "labels" / f"{f}.txt") for f in ann}

    def fit_hubs(frames):
        return np.array([triangulate([(*pose[f], np.array(ann[f]["hubs"][j])) for f in frames
                                      if ann[f]["use"] != "holdout" and ann[f]["hubs"][j] is not None])
                         for j in range(4)])

    hubs0 = fit_hubs(ann)
    swapped = sorted({r["frame"] for r in click_errors(ann, hubs0, CAM_FROM_VICON, root)
                      if sum(q["swap"] for q in click_errors({r["frame"]: ann[r["frame"]]}, hubs0, CAM_FROM_VICON, root)) >= 2})
    ann = {f: a for f, a in ann.items() if f not in swapped}
    hubs0 = fit_hubs(ann)
    fit = [f for f, a in ann.items() if a["use"] != "holdout"]

    def res(x):
        T, H = corrected(CAM_FROM_VICON, x[:6]), x[6:].reshape(4, 3)
        out_ = []
        for f in fit:
            R, t = read_label(root / "labels" / f"{f}.txt", T)
            p = project(H, R, t)
            out_ += [p[j] - ann[f]["hubs"][j] for j in range(4) if ann[f]["hubs"][j] is not None]
        return np.concatenate(out_)

    x = least_squares(res, np.concatenate([np.zeros(6), hubs0.ravel()]), loss="huber", f_scale=HUBER_PX).x
    T, hubs = corrected(CAM_FROM_VICON, x[:6]), x[6:].reshape(4, 3)

    rows0, rows = click_errors(ann, hubs0, CAM_FROM_VICON, root), click_errors(ann, hubs, T, root)
    centroid = hubs.mean(0)
    d = lambda i, k: float(np.linalg.norm(hubs[i] - hubs[k]))   # noqa: E731
    geo = {"diag_13_m": d(0, 2), "diag_24_m": d(1, 3), "front_12_m": d(0, 1), "rear_34_m": d(2, 3),
           "side_14_m": d(0, 3), "side_23_m": d(1, 2)}
    corr = {"rotation_deg": float(np.degrees(np.linalg.norm(x[:3]))), "translation_mm": (x[3:6] * 1000).tolist()}
    check = {"excluded_swapped_frames": swapped, "correction": corr, "geometry": geo,
             "authors_calibration": {"fit_px": summary(rows0, False), "holdout_px": summary(rows0, True)},
             "refined": {"fit_px": summary(rows, False), "holdout_px": summary(rows, True)}, "rows": rows}
    model = {"frame": "VICON body of MAV6D Mavic 2 (FLU: x front, y left, z up), metres",
             "order": "key1 front-right, key2 front-left, key3 rear-left, key4 rear-right",
             "hubs_body": hubs.tolist(), "centroid_body": centroid.tolist(), "kpts_flu": (hubs - centroid).tolist(),
             "cam_from_vicon": T.tolist(), "n_frames_fit": len(fit), "excluded_swapped_frames": swapped}
    (out / "hub_model.json").write_text(json.dumps(model, indent=1))
    (out / "hub_check.json").write_text(json.dumps(check, indent=1))
    print(f"excluded (hubs in the wrong order): {swapped}")
    for name, c in (("authors' calibration", check["authors_calibration"]), ("refined", check["refined"])):
        print(f"{name:21s} fit median {c['fit_px']['median']:5.2f} px p90 {c['fit_px']['p90']:5.2f} | holdout median "
              f"{c['holdout_px']['median']:5.2f} p90 {c['holdout_px']['p90']:5.2f} max {c['holdout_px']['max']:5.2f}"
              f" ({c['holdout_px']['n']} clicks)")
    print(f"correction: {corr['rotation_deg']:.3f} deg, {np.round(corr['translation_mm'], 1)} mm; "
          f"geometry (m): { {k: round(v, 4) for k, v in geo.items()} }")
    print("hubs FLU about the centroid (m):\n", np.round(hubs - centroid, 4))


if __name__ == "__main__":
    main()
