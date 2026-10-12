"""Per-drone 3D propeller models in FLU, recovered from the annotations by inverting the rigid transform."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from dronepose.common.geometry import BLENDER_BODY_TO_FLU
from dronepose.data.sixdronesyn import RawFrame, corrected_rotation, to_circular


@dataclass
class DroneModel:
    kpts_flu: np.ndarray           # (4,3) circular order, centred on the propeller centroid
    origin_offset_flu: np.ndarray  # (3,) Blender object origin relative to the centroid, FLU
    spread: float                  # max std across frames (dataset units): rigidity check


def body_points_flu(frame: RawFrame) -> np.ndarray:
    k3 = to_circular(frame.drone, frame.kpts3d_cam)
    body = (corrected_rotation(frame).T @ (k3 - frame.t_cam_blender).T).T
    return (BLENDER_BODY_TO_FLU @ body.T).T


def build_drone_models(frames: Iterable[RawFrame], max_spread: float = 1e-4) -> dict[str, DroneModel]:
    per_drone: dict[str, list[np.ndarray]] = {}
    for f in frames:
        per_drone.setdefault(f.drone, []).append(body_points_flu(f))
    models = {}
    for drone, pts in sorted(per_drone.items()):
        stack = np.stack(pts)
        spread = float(stack.std(0).max())
        if spread > max_spread:
            raise ValueError(f"drone {drone}: spread {spread:.3g} > {max_spread} (rotation fix or keypoint order wrong?)")
        mean = stack.mean(0)
        centroid = mean.mean(0)
        models[drone] = DroneModel(kpts_flu=mean - centroid, origin_offset_flu=-centroid, spread=spread)
    return models


def save_drone_models(models: dict[str, DroneModel], path: Path) -> None:
    out = {d: {"kpts_flu": m.kpts_flu.tolist(), "origin_offset_flu": m.origin_offset_flu.tolist(),
               "spread": m.spread} for d, m in models.items()}
    meta = {"frame": "FLU (x front = propellers 1-2, y left, z up)", "units": "6DroneSyn camera-frame units",
            "order": "key1 front-right, key2 front-left, key3 rear-left, key4 rear-right"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"meta": meta, "models": out}, indent=2))


def load_drone_models(path: Path) -> dict[str, DroneModel]:
    data = json.loads(Path(path).read_text())["models"]
    return {d: DroneModel(np.array(m["kpts_flu"]), np.array(m["origin_offset_flu"]), m["spread"])
            for d, m in data.items()}
