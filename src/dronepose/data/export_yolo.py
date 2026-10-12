"""Ultralytics YOLO-Pose labels: class cx cy w h then (x y v) x 4, normalised to the crop."""
from pathlib import Path

import numpy as np
import yaml

from dronepose.data.crops import CROP_SIZE
from dronepose.data.sixdronesyn import DRONE_IDS

CLASS_IDS = {d: i for i, d in enumerate(DRONE_IDS)}


def yolo_label_line(cls_id: int, kpts_crop: np.ndarray, size: int = CROP_SIZE, pad: float = 0.2) -> str:
    lo, hi = kpts_crop.min(0), kpts_crop.max(0)
    p = pad * float(max(hi - lo))
    lo, hi = np.clip(lo - p, 0, size), np.clip(hi + p, 0, size)
    cx, cy = (lo + hi) / 2 / size
    w, h = (hi - lo) / size
    kp = " ".join(f"{x / size:.6f} {y / size:.6f} 2" for x, y in kpts_crop)
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kp}"


def write_data_yaml(path: Path, dataset_root: str) -> None:
    cfg = {"path": dataset_root, "train": "images/train", "val": "images/val", "test": "images/test",
           "kpt_shape": [4, 3], "flip_idx": [1, 0, 3, 2], "names": {i: d for d, i in CLASS_IDS.items()}}
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
