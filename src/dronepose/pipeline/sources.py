"""Sources of the integrated pipeline: frames in order plus what is known about the scene (camera, gravity, class).

Only 6DroneSyn for now; MAV6D gets its own source with the same interface (spec 2026-11-14 §5.1).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import DroneModel, load_drone_models
from dronepose.eval.units import GRAVITY_FROM_CAMERA
from dronepose.pose.replay import _by_sequence, check_no_test


@dataclass
class Frame:
    frame_id: str
    frame_idx: int
    image: np.ndarray            # BGR, original resolution
    K: np.ndarray                # (3, 3)
    record: dict | None = None   # canonical record (ground truth, metrics), None on unlabelled sources


@dataclass
class SceneInfo:
    G: np.ndarray                     # camera -> gravity-aligned frame
    fixed_class: str | None           # known drone class (stage skipped) or None
    models: dict[str, DroneModel]     # candidate 3D models


class SixDroneSynSource:
    def __init__(self, canonical: Path = Path("data/processed/canonical_val.jsonl"),
                 models_path: Path = Path("configs/drone_models.json")):
        records = read_jsonl(Path(canonical))
        check_no_test(records, allow_test=False)
        self._seqs = _by_sequence(records)
        self._models = load_drone_models(Path(models_path))

    def sequences(self) -> list[str]:
        return list(self._seqs)

    def records(self, seq: str) -> list[dict]:
        return self._seqs[seq]

    def info(self, seq: str) -> SceneInfo:
        if seq not in self._seqs:
            raise KeyError(seq)
        return SceneInfo(G=GRAVITY_FROM_CAMERA, fixed_class=None, models=self._models)

    def frames(self, seq: str) -> Iterator[Frame]:
        for r in self._seqs[seq]:
            img = cv2.imread(str(r["image_path"]), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(r["image_path"])
            yield Frame(r["frame_id"], int(r["frame_idx"]), img, np.asarray(r["K"], float), r)
