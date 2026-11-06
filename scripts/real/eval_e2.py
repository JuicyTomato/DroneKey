"""E2 on MAV6D (real Mavic 2): keypoints against the hubs from VICON, and pose against VICON (protocol in
PROJECT_MEMORY §6octodecies). External test set: no choice is made on it.

  .venv/bin/python scripts/real/eval_e2.py --out experiments/real/mav6d/e2_b.json [--ref yolo_s0]
      [--group yolo=yolo_s0,yolo_s1,yolo_s2 ...] [--det-metrics C=DETS.jsonl] NAME=KPTS.jsonl ...

2D metrics on the original (distorted) image pixels, the ones the models see. Pose: keypoints and GT moved to an ideal
pinhole camera (cv2.undistortPoints with the MAV6D K), then the E1 protocol (IPPE + `global`, true class, no gravity
gate), with two 3D models of the Mavic: "real" (hubs triangulated by mav6d_hubs.py, metres) and "syn05" (the 6DroneSyn
model of drone 05 scaled to metres with METERS_PER_UNIT["05"], the one the pipeline would use). Translation errors
are in metres for both. Subsets: all, authors' test split, distance and image-speed bins. Paired bootstrap per
sequence (61 units) against --ref on rotation (per 3D model) and on the pixel error.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
from dronepose.data.canonical import read_jsonl  # noqa: E402
from dronepose.data.drone_models import DroneModel, load_drone_models  # noqa: E402
from dronepose.eval.bootstrap import by_trajectory, paired_bootstrap  # noqa: E402
from dronepose.eval.detection import detection_metrics  # noqa: E402
from dronepose.eval.e1 import e1_metrics, mean_std  # noqa: E402
from dronepose.eval.keypoint_metrics import frame_px_error  # noqa: E402
from dronepose.eval.units import METERS_PER_UNIT  # noqa: E402
from dronepose.keypoints.detections import class_causal_vote, load_detections  # noqa: E402
from dronepose.pose.replay import load_keypoints, replay, sequence_key  # noqa: E402

N_BOOT = 2000
POSE_KEYS = ("coverage", "median_rot_sym", "p90_rot_sym", "median_t_camera", "flip_rate")
KPT_KEYS = ("px_median", "px_p90", "pck_2px", "pck_5px", "pck_10px", "pck_0.02diag", "pck_0.05diag", "identity",
            "ap_paper")


def subsets(records: list[dict]) -> dict[str, set[str]]:
    def ids(cond):
        return {r["frame_id"] for r in records if cond(r)}
    subs = {"all": ids(lambda r: True), "authors_test": ids(lambda r: r["authors_split"] == "test"),
            "dist<3": ids(lambda r: r["distance_m"] < 3), "dist3-4": ids(lambda r: 3 <= r["distance_m"] < 4),
            "dist>=4": ids(lambda r: r["distance_m"] >= 4),
            "speed<150": ids(lambda r: r["px_speed"] < 150), "speed150-500": ids(lambda r: 150 <= r["px_speed"] < 500),
            "speed>=500": ids(lambda r: r["px_speed"] >= 500)}
    return {k: v for k, v in subs.items() if v}


def restrict(kp: dict, by_id: dict) -> dict:
    """Predictions of the frames under evaluation only: an export may hold more frames than --canonical."""
    return {f: v for f, v in kp.items() if f in by_id}


def undistort_kpts(kp: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(4,3) keypoints in distorted pixels -> ideal pinhole pixels (same K); NaN rows stay NaN."""
    out = np.array(kp, float, copy=True)
    ok = np.isfinite(out[:, :2]).all(1)
    if ok.any():
        out[ok, :2] = cv2.undistortPoints(out[ok, :2].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    return out


def planar(k: np.ndarray) -> np.ndarray:
    """Project the points on their least-squares plane: IPPE needs exactly coplanar points (the triangulated hubs are
    off by < 1 mm)."""
    c = k - k.mean(0)
    n = np.linalg.svd(c)[2][2]
    return k - np.outer(c @ n, n)


def models_3d(hub_model: Path, syn_models: Path) -> dict[str, dict[str, DroneModel]]:
    real = planar(np.array(json.loads(hub_model.read_text())["kpts_flu"]))
    syn = load_drone_models(syn_models)["05"].kpts_flu * METERS_PER_UNIT["05"]
    return {name: {"05": DroneModel(k, np.zeros(3), 0.0)} for name, k in (("real", real), ("syn05", syn))}


def vote_or_true(records: list[dict], dets: dict) -> dict[str, str]:
    """Causal class vote per sequence. A sequence without detections has no keypoints in setting B, hence no pose: the
    class given to it does not count, and it must not stop a run that happens once."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(sequence_key(r["frame_id"]), []).append(r)
    out: dict[str, str] = {}
    for recs in groups.values():
        try:
            out.update(class_causal_vote(recs, dets))
        except ValueError:
            out.update({r["frame_id"]: r["drone"] for r in recs})
    return out


def check_dump(subs: dict, dump_frames: str | None) -> None:
    """Per-frame rotations feed the choice of the recipe: they must never come from the authors' test split."""
    if dump_frames and "authors_test" in subs:
        raise ValueError("--dump-frames on records with authors_test: use authors_train/gt.jsonl")


def detection_line(name: str, m: dict) -> str:
    """recall@0.5 counts every frame; the IoU fields only the detected ones, and they are None without detections."""
    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"
    return (f"detection {name}: recall@0.5 {fmt(m['recall_05'])}  no-det {fmt(m['no_det_rate'])}  IoU median "
            f"{fmt(m['iou_median'])}  >=0.5 of detected {fmt(m['iou_ge_05'])}  class acc {fmt(m['class_acc'])}")


def syn_models(path: Path) -> dict[str, DroneModel]:
    """All 6DroneSyn models in metres, as the pipeline would pick them from a predicted class. Drone 08 has no scale
    (1.0 here): it changes the translation of those frames, not the rotation."""
    return {d: DroneModel(m.kpts_flu * METERS_PER_UNIT.get(d, 1.0), np.zeros(3), 0.0)
            for d, m in load_drone_models(path).items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", metavar="NAME=KPTS.jsonl")
    ap.add_argument("--canonical", default="data/processed/mav6d_gt.jsonl")
    ap.add_argument("--hub-model", default="experiments/real/mav6d/hub_model.json")
    ap.add_argument("--syn-models", default="configs/drone_models.json")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--group", action="append", default=[], metavar="NAME=N1,N2,N3")
    ap.add_argument("--det-metrics", action="append", default=[], metavar="NAME=DETS.jsonl")
    ap.add_argument("--class-from", default=None, metavar="NAME=DETS.jsonl",
                    help="extra pose row with the class voted from these detections")
    ap.add_argument("--dump-frames", default=None, help="folder: per model, rotation (real 3D model) per frame")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    records = read_jsonl(Path(args.canonical))
    if any(r["split"] != "mav6d" for r in records):
        raise ValueError("eval_e2.py reads MAV6D records only")
    by_id = {r["frame_id"]: r for r in records}
    K, dist = np.array(records[0]["K"]), np.array(records[0]["dist"])
    rec_u = [{**r, "kpts2d": undistort_kpts(np.c_[np.array(r["kpts2d"]), np.ones(4)], K, dist)[:, :2].tolist()}
             for r in records]
    subs = subsets(records)
    check_dump(subs, args.dump_frames)
    m3d = models_3d(Path(args.hub_model), Path(args.syn_models))
    specs = dict(s.split("=", 1) for s in args.specs)
    ref = args.ref or next(iter(specs))

    out = {"split": "mav6d", "n_frames": len(records), "models": {}, "bootstrap": {}, "groups": {}}
    vote, syn_all = None, None
    if args.class_from:
        vote = vote_or_true(records, load_detections(Path(args.class_from.split("=", 1)[1])))
        syn_all = syn_models(Path(args.syn_models))
        out["class_pred"] = {}
    if args.dump_frames:
        Path(args.dump_frames).mkdir(parents=True, exist_ok=True)
    rows_rot = {v: {} for v in m3d}
    rows_px = {}
    for name, path in specs.items():
        kp = restrict(load_keypoints(Path(path)), by_id)
        kp_u = {f: undistort_kpts(k, K, dist) for f, k in kp.items()}
        entry = {"kpts": path}
        for v, models in m3d.items():
            res = [r for rs in replay(rec_u, kp_u, models, "global").values() for r in rs]
            table = {s: e1_metrics([r for r in res if r["frame_id"] in ids], by_id, kp) for s, ids in subs.items()}
            entry[v] = {s: {k: t[k] for k in POSE_KEYS} for s, t in table.items()}
            if v == "real":
                entry["kpts2d"] = {s: {k: t[k] for k in KPT_KEYS + ("n_frames",)} for s, t in table.items()}
            rows_rot[v][name] = by_trajectory(res, "rot_err_sym_deg")
            if v == "real" and args.dump_frames:
                (Path(args.dump_frames) / f"{name}.json").write_text(json.dumps({r["frame_id"]: r["rot_err_sym_deg"] for r in res}))
        if vote is not None:
            res = [r for rs in replay(rec_u, kp_u, syn_all, "global", model_drone=vote).values() for r in rs]
            t = e1_metrics(res, by_id, kp)
            out["class_pred"][name] = {k: t[k] for k in POSE_KEYS}
        rows_px[name] = by_trajectory([{"frame_id": f, "px": frame_px_error(kp.get(f, np.full((4, 3), np.nan)),
                                                                            np.array(by_id[f]["kpts2d"]))}
                                       for f in by_id], "px")
        out["models"][name] = entry
        a, r_ = entry["kpts2d"]["all"], entry["real"]["all"]
        print(f"{name:10s} px {a['px_median']:.2f} (p90 {a['px_p90']:.1f})  PCK5 {a['pck_5px']:.3f}  ident "
              f"{a['identity']:.3f} | real: cov {r_['coverage']:.3f} rot {r_['median_rot_sym']:.2f} "
              f"(p90 {r_['p90_rot_sym']:.1f}) t {r_['median_t_camera']:.3f} m flip {r_['flip_rate']:.3f} | syn05 rot "
              f"{entry['syn05']['all']['median_rot_sym']:.2f} t {entry['syn05']['all']['median_t_camera']:.3f}",
              flush=True)

    out["bootstrap"] = {f"rot_{v}": paired_bootstrap(rows_rot[v], ref, N_BOOT) for v in m3d}
    out["bootstrap"]["px"] = paired_bootstrap(rows_px, ref, N_BOOT)
    print(f"\npaired bootstrap per sequence against {ref} (median, 95%)")
    for key, b in out["bootstrap"].items():
        for n, (d, lo, hi) in b.items():
            if n != ref:
                print(f"  {key:9s} {n:10s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for g in args.group:
        gname, members = g.split("=", 1)
        members = members.split(",")
        tabs = {k: [{s: out["models"][m][k][s] for s in subs} for m in members] for k in ("kpts2d", "real", "syn05")}
        out["groups"][gname] = {"members": members, **{k: mean_std(v) for k, v in tabs.items()}}
        ms = out["groups"][gname]
        print(f"group {gname}: px {ms['kpts2d']['all']['px_median']['mean']:.2f}±{ms['kpts2d']['all']['px_median']['std']:.2f}"
              f"  rot real {ms['real']['all']['median_rot_sym']['mean']:.2f}±{ms['real']['all']['median_rot_sym']['std']:.2f}"
              f"  t real {ms['real']['all']['median_t_camera']['mean']:.3f}  rot syn05 "
              f"{ms['syn05']['all']['median_rot_sym']['mean']:.2f}")
    if args.det_metrics:
        out["detection"] = {}
        for item in args.det_metrics:
            name, path = item.split("=", 1)
            m = detection_metrics(load_detections(Path(path)), records)
            out["detection"][name] = m
            print(detection_line(name, m))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
