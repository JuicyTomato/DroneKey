"""Export the poses of one combination (and, on val, the error archive) for the forecast phase.

Usage:
  python scripts/export_poses.py --combo NAME --det DETS.jsonl --split val|test [--allow-test] --seed N
      --kpts KPTS.jsonl --out-dir experiments [--archive]
NAME = solver[+lm]+strategy, e.g. ippe+global. Writes, for both drone classes (causal = predicted class with causal
vote, gt = oracle class), OUT_DIR/poses/poses_{split}_s{N}_{causal,gt}.jsonl: one row per frame, no header, `null`
pose fields for frames without a pose. R_g = G R, t_g = G t s_c (metres; dataset units and `meters: false` for
drones without s_c). `covariance` is null (hook for the uncertainty).
--archive (val only) also writes OUT_DIR/pose_errors/archive_val_s{N}.jsonl (class causal; see pose/error_archive.py).
The test split needs --allow-test; it is exported, never evaluated.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.eval.e3 import CLASSES, Combo, all_combos
from dronepose.eval.units import GRAVITY_FROM_CAMERA, METERS_PER_UNIT
from dronepose.keypoints.detections import class_causal_vote, load_detections
from dronepose.pose.error_archive import archive_rows, header, write_archive
from dronepose.pose.replay import check_no_test, load_keypoints, replay, sequence_key


def parse_combo(name: str) -> Combo:
    """`solver[+lm]+strategy` -> Combo (class left to the caller, 'gt' here); ValueError if unknown."""
    for c in all_combos():
        if c.name == name:
            return c
    raise ValueError(f"unknown combo {name!r}; expected solver[+lm]+strategy, e.g. ippe+global")


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _flat(results: dict[str, list[dict]]) -> list[dict]:
    return [r for rs in results.values() for r in rs]


def pose_rows(results: list[dict], records_by_id: dict[str, dict], model_drone: dict[str, str] | None,
              split: str, seed: int, combo: str) -> list[dict]:
    """One row per frame sorted by (sequence, frame_idx); pose fields are null when the frame has no valid pose."""
    rows = []
    for res in results:
        fid = res["frame_id"]
        rec = records_by_id[fid]
        drone = rec["drone"]
        s_c = METERS_PER_UNIT.get(drone)
        row = {"frame_id": fid, "split": split, "sequence": sequence_key(fid), "frame_idx": int(res["frame_idx"]),
               "drone": drone, "drone_model": model_drone[fid] if model_drone else drone, "valid": False,
               "reason": res.get("reject_reason"), "R_cam": None, "t_cam": None, "R_g": None, "t_g": None,
               "meters": s_c is not None, "covariance": None, "seed": seed, "combo": combo}
        R, t = res.get("R"), res.get("t")
        if res.get("rejected") or R is None or t is None:
            if row["reason"] is None:
                row["reason"] = "no_pose"
        elif not (np.all(np.isfinite(np.asarray(R, float))) and np.all(np.isfinite(np.asarray(t, float)))):
            row["reason"] = "non_finite"
        else:
            R, t = np.asarray(R, float), np.asarray(t, float)
            G = GRAVITY_FROM_CAMERA
            row.update(valid=True, reason=None, R_cam=R.tolist(), t_cam=t.tolist(), R_g=(G @ R).tolist(),
                       t_g=(G @ t * (1.0 if s_c is None else s_c)).tolist())
        rows.append(row)
    rows.sort(key=lambda r: (r["sequence"], r["frame_idx"]))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")


def export(combo: str, det: str, split: str, allow_test: bool, seed: int, kpts: str, out_dir: str,
           archive: bool = False, records: list[dict] | None = None, canonical: str | None = None,
           models_path: str = "configs/drone_models.json") -> list[Path]:
    """Replay `combo` on `split` and write the pose files (and the archive); returns the written paths."""
    if split == "test" and not allow_test:
        raise ValueError("the test split needs --allow-test (it is exported, never evaluated)")
    if archive and split != "val":
        raise ValueError("--archive is for the val split only")
    c = parse_combo(combo)
    if records is None:
        records = read_jsonl(Path(canonical or f"data/processed/canonical_{split}.jsonl"))
    check_no_test(records, allow_test)
    models = load_drone_models(Path(models_path))
    kp = load_keypoints(Path(kpts))
    dets = load_detections(Path(det))
    by_id = {r["frame_id"]: r for r in records}
    model_drone = {"gt": None, "causal": class_causal_vote(records, dets)}
    out, written = Path(out_dir), []
    for cls in CLASSES:
        res = _flat(replay(records, kp, models, c.strategy, model_drone=model_drone[cls], solver=c.solver,
                           refine=c.refine))
        path = out / "poses" / f"poses_{split}_s{seed}_{cls}.jsonl"
        write_jsonl(path, pose_rows(res, by_id, model_drone[cls], split, seed, combo))
        written.append(path)
        print(path, f"({len(res)} frames, {sum(r['R'] is not None and not r.get('rejected') for r in res)} with pose)")
        if archive and cls == "causal":
            path = out / "pose_errors" / f"archive_val_s{seed}.jsonl"
            write_archive(path, header(combo, seed, git_commit()), archive_rows(res, by_id, model_drone[cls], seed))
            written.append(path)
            print(path)
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combo", required=True, help="e.g. ippe+global")
    ap.add_argument("--det", required=True, help="detections JSONL (class vote)")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--kpts", required=True, help="keypoints JSONL of this seed")
    ap.add_argument("--out-dir", default="experiments")
    ap.add_argument("--archive", action="store_true", help="also write the error archive (val only)")
    args = ap.parse_args(argv)
    export(args.combo, args.det, args.split, args.allow_test, args.seed, args.kpts, args.out_dir, args.archive)


if __name__ == "__main__":
    main()
