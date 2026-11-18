"""Sources of the integrated pipeline: frames in order plus what is known about the scene (camera, gravity, class).

6DroneSyn and MAV6D (authors' train only), same interface (spec 2026-11-14 §5.1, 2026-11-18).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import DroneModel, load_drone_models
from dronepose.data.mav6d import planar
from dronepose.eval.units import GRAVITY_FROM_CAMERA, METERS_PER_UNIT
from dronepose.pose.replay import _by_sequence, check_no_test


@dataclass
class Frame:
    frame_id: str
    frame_idx: int
    image: np.ndarray            # BGR, original resolution
    K: np.ndarray                # (3, 3)
    record: dict | None = None   # canonical record (ground truth, metrics), None on unlabelled sources
    dist: np.ndarray | None = None   # OpenCV distortion (k1 k2 p1 p2 k3); None = ideal pinhole


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


class MAV6DSource:
    """MAV6D (real Mavic 2), authors' train split only, everything in the units of the 6DroneSyn model 05."""

    def __init__(self, gt: Path = Path("data/processed/mav6d_gt.jsonl"),
                 hub_model: Path = Path("experiments/real/mav6d/hub_model.json")):
        records = read_jsonl(Path(gt))
        # test sequences are only remembered by name, to refuse them: their records are dropped here
        self._test = {r["sequence"] for r in records if r["authors_split"] == "test"}
        self._seqs = _by_sequence([r for r in records if r["authors_split"] != "test"])
        hub = json.loads(Path(hub_model).read_text())
        m = METERS_PER_UNIT["05"]
        self._model = DroneModel(planar(np.array(hub["kpts_flu"], float)) / m, np.zeros(3), 0.0)
        self._G = np.asarray(hub["cam_from_vicon"], float)[:, :3].T    # camera -> VICON (z up)

    def _check(self, seq: str) -> None:
        if seq in self._test:
            raise ValueError(f"sequence {seq} belongs to the authors' test split: not allowed")
        if seq not in self._seqs:
            raise KeyError(seq)

    def sequences(self) -> list[str]:
        return list(self._seqs)

    def records(self, seq: str) -> list[dict]:
        self._check(seq)
        m = METERS_PER_UNIT["05"]
        return [{**r, "t_gt": (np.asarray(r["t_gt"], float) / m).tolist()} for r in self._seqs[seq]]

    def info(self, seq: str) -> SceneInfo:
        self._check(seq)
        return SceneInfo(G=self._G, fixed_class="05", models={"05": self._model})

    def fps(self, seq: str) -> float:
        self._check(seq)
        ts = [int(r["frame_id"].rsplit("_", 1)[1]) for r in self._seqs[seq]]
        return 1e9 / float(np.median(np.diff(ts)))

    def frames(self, seq: str) -> Iterator[Frame]:
        for r in self.records(seq):
            img = cv2.imread(str(r["image_path"]), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(r["image_path"])
            yield Frame(r["frame_id"], int(r["frame_idx"]), img, np.asarray(r["K"], float), r,
                        np.asarray(r["dist"], float))
