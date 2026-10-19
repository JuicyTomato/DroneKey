"""Ultralytics YOLO-Pose labels: class cx cy w h then (x y v) x 4, normalised to the image (crop or full frame)."""
from pathlib import Path

import numpy as np
import yaml

from dronepose.data.crops import BOX_PAD, CROP_SIZE
from dronepose.data.sixdronesyn import DRONE_IDS

CLASS_IDS = {d: i for i, d in enumerate(DRONE_IDS)}


def yolo_label_line(cls_id: int, kpts: np.ndarray, size: int | tuple[float, float] = CROP_SIZE,
                    pad: float = BOX_PAD) -> str:
    """`size` = side of a square crop, or (width, height) of a full frame; `kpts` in that image's pixels."""
    wh = np.array((size, size) if np.isscalar(size) else size, float)
    lo, hi = kpts.min(0), kpts.max(0)
    p = pad * float(max(hi - lo))
    lo, hi = np.clip(lo - p, 0, wh), np.clip(hi + p, 0, wh)
    cx, cy = (lo + hi) / 2 / wh
    w, h = (hi - lo) / wh
    kp = " ".join(f"{x / wh[0]:.6f} {y / wh[1]:.6f} 2" for x, y in kpts)
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kp}"


def write_data_yaml(path: Path, dataset_root: str) -> None:
    cfg = {"path": dataset_root, "train": "images/train", "val": "images/val", "test": "images/test",
           "kpt_shape": [4, 3], "flip_idx": [1, 0, 3, 2], "names": {i: d for d, i in CLASS_IDS.items()}}
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
