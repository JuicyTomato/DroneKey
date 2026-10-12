"""Canonical per-frame record: circular keypoint order, FLU pose GT (Kabsch), centroid translation."""
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from dronepose.common.geometry import kabsch, project
from dronepose.data.drone_models import DroneModel
from dronepose.data.sixdronesyn import SPLIT_BY_SCENE, RawFrame, to_circular


def to_canonical(frame: RawFrame, models: dict[str, DroneModel]) -> dict:
    k2 = to_circular(frame.drone, frame.kpts2d)
    k3 = to_circular(frame.drone, frame.kpts3d_cam)
    R, t = kabsch(models[frame.drone].kpts_flu, k3)   # model is centred, so t = centroid of k3
    return {
        "frame_id": frame.frame_id, "image_path": str(frame.image_path),
        "scene": frame.scene, "drone": frame.drone, "background": frame.background,
        "degree": frame.degree, "frame_idx": frame.frame_idx, "split": SPLIT_BY_SCENE[frame.scene],
        "units": "camera", "K": frame.K.tolist(), "bbox_xyxy": frame.bbox_xyxy.tolist(),
        "kpts2d": k2.tolist(), "kpts3d_cam": k3.tolist(),
        "R_gt": R.tolist(), "t_gt": t.tolist(), "t_blender": frame.t_cam_blender.tolist(),
    }


def model_reprojection_px(record: dict, models: dict[str, DroneModel]) -> float:
    R, t = np.array(record["R_gt"]), np.array(record["t_gt"])
    pts = (R @ models[record["drone"]].kpts_flu.T).T + t
    return float(np.linalg.norm(project(np.array(record["K"]), pts) - np.array(record["kpts2d"]), axis=1).mean())


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
