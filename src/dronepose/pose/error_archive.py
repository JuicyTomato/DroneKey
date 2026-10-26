"""Pose-error archive: real pose errors (position in the gravity frame, rotation in the body frame) per frame.

Conventions (consumed by `forecast/noise.py`):
  dp_g = G @ (t_est - t_gt) * s_c     meters, gravity frame (dataset units, `meters: false`, for drones without s_c)
  dR   = R_gt.T @ R_est               body frame, applied as `obs_R @ dR`
"""
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.eval.metrics import SYMMETRIC_DRONES
from dronepose.eval.units import GRAVITY_FROM_CAMERA, METERS_PER_UNIT
from dronepose.pose.replay import sequence_key

VERSION = 1
FLIP_DEG = 90.0


@dataclass
class SequenceErrors:
    sequence: str
    drone: str
    meters: bool
    frame_idx: np.ndarray    # (N,) int, ascending
    valid: np.ndarray        # (N,) bool
    dp_g: np.ndarray         # (N, 3), NaN where not valid
    dR: np.ndarray           # (N, 3, 3), NaN where not valid
    distance_m: np.ndarray   # (N,) |t_gt| * s_c (dataset units when not meters)


def header(combo: str, seed: int, commit: str) -> dict:
    return {
        "kind": "header", "version": VERSION, "combo": combo, "seed": seed, "commit": commit,
        "conventions": {"dp_g": "G @ (t_est - t_gt) * s_c", "dR": "R_gt.T @ R_est",
                        "rotvec_deg": "rotation vector of dR in degrees", "G": GRAVITY_FROM_CAMERA.tolist()},
        "units": {"dp_g": "meters (gravity frame); dataset units when meters is false",
                  "distance_m": "meters; dataset units when meters is false",
                  "los_g": "unit vector from camera to drone (G @ t_gt / |t_gt|), gravity frame",
                  "rotation": "degrees"},
    }


def _finite(x) -> bool:
    return x is not None and bool(np.all(np.isfinite(np.asarray(x, float))))


def archive_rows(results: list[dict], records_by_id: dict[str, dict],
                 model_drone: dict[str, str] | None, seed: int) -> list[dict]:
    """One row per frame of `results` (flat list, all sequences), sorted by (sequence, frame_idx)."""
    rows = []
    for res in results:
        fid = res["frame_id"]
        rec = records_by_id[fid]
        drone = rec["drone"]
        s_c = METERS_PER_UNIT.get(drone)
        meters = s_c is not None
        scale = s_c if meters else 1.0
        t_gt = np.asarray(rec["t_gt"], float)
        distance = float(np.linalg.norm(t_gt) * scale)
        los_g = GRAVITY_FROM_CAMERA @ t_gt / np.linalg.norm(t_gt)      # unit line of sight, gravity frame
        row = {"frame_id": fid, "sequence": sequence_key(fid), "frame_idx": int(res["frame_idx"]),
               "scene": rec["scene"], "drone": drone,
               "drone_model": model_drone[fid] if model_drone else drone, "seed": seed,
               "valid": False, "reason": res.get("reject_reason"), "dp_g": None, "meters": meters,
               "dR": None, "rotvec_deg": None, "rot_err_deg": None, "flip": False,
               "symmetric": drone in SYMMETRIC_DRONES, "distance_m": distance,
               "los_g": [float(v) for v in los_g]}
        if res.get("rejected") or res.get("R") is None or res.get("t") is None:
            if row["reason"] is None:
                row["reason"] = "no_pose"
        elif not (_finite(res["R"]) and _finite(res["t"])):
            row["reason"] = "non_finite"
        else:
            R_est, t_est = np.asarray(res["R"], float), np.asarray(res["t"], float)
            dp = GRAVITY_FROM_CAMERA @ (t_est - t_gt) * scale
            dR = np.asarray(rec["R_gt"], float).T @ R_est
            rot_err = res.get("rot_err_deg")
            row.update(valid=True, reason=None, dp_g=[float(v) for v in dp],
                       dR=[[float(v) for v in line] for line in dR],
                       rotvec_deg=[float(v) for v in Rotation.from_matrix(dR).as_rotvec(degrees=True)],
                       rot_err_deg=None if rot_err is None else float(rot_err),
                       flip=rot_err is not None and rot_err > FLIP_DEG)
        rows.append(row)
    rows.sort(key=lambda r: (r["sequence"], r["frame_idx"]))
    return rows


def write_archive(path: Path, head: dict, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(head, allow_nan=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")


def load_archive(path: Path) -> tuple[dict, dict[str, SequenceErrors]]:
    head, groups = None, defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "header":
                head = row
            else:
                groups[row["sequence"]].append(row)
    if head is None:
        raise ValueError(f"{path}: no header line")
    out = {}
    for seq, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["frame_idx"])
        n = len(rows)
        valid = np.array([bool(r["valid"]) for r in rows])
        dp = np.full((n, 3), np.nan)
        dR = np.full((n, 3, 3), np.nan)
        for i, r in enumerate(rows):
            if r["valid"]:
                dp[i] = r["dp_g"]
                dR[i] = r["dR"]
        out[seq] = SequenceErrors(
            sequence=seq, drone=rows[0]["drone"], meters=bool(rows[0]["meters"]),
            frame_idx=np.array([r["frame_idx"] for r in rows], dtype=int), valid=valid, dp_g=dp, dR=dR,
            distance_m=np.array([math.nan if r["distance_m"] is None else r["distance_m"] for r in rows], float))
    return head, out


def windows(archive: dict[str, SequenceErrors], length: int = 15, step: int = 1) -> list[tuple[str, int]]:
    """(sequence, start index) of every run of `length` consecutive frames (no gap in `frame_idx`).
    Invalid frames stay inside the windows."""
    out = []
    for seq, se in archive.items():
        idx = se.frame_idx
        for i in range(0, len(idx) - length + 1, step):
            if idx[i + length - 1] - idx[i] == length - 1:
                out.append((seq, i))
    return out
