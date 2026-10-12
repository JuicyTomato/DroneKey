"""A tiny fake 6DroneSyn: same folder layout, same JSON fields, same rotation bug, same mixed keypoint order."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.common.geometry import BLENDER_BODY_TO_FLU, project, rot_z

FAKE_K = np.array([[2666.6667, 0.0, 960.0], [0.0, 2666.6667, 540.0], [0.0, 0.0, 1.0]])
# Real FLU models (PROJECT_MEMORY §3.5), circular order, origin = Blender object origin, camera units.
FLU_MODELS = {
    "01": np.array([[0.081, -0.154, -0.015], [0.081, 0.151, -0.015], [-0.119, 0.110, 0.020], [-0.119, -0.107, 0.020]]),
    "02": np.array([[0.059, -0.109, 0.016], [0.059, 0.108, 0.016], [-0.096, 0.105, -0.011], [-0.096, -0.104, -0.011]]),
}
_Z_ORDER = {"01"}


@dataclass
class FakeSequence:
    scene: str
    drone: str
    background: str
    degree: int
    n_frames: int
    R_true: np.ndarray                     # Blender body -> camera
    t0: np.ndarray                         # Blender origin at frame 1, camera frame
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))   # units per frame
    inner_folder: str | None = None        # e.g. "scene_12" inside scene_13, like the real dataset
    missing_frames: tuple = ()             # frame numbers whose PNG is not written


def raw_body_points(drone: str) -> np.ndarray:
    body = (BLENDER_BODY_TO_FLU.T @ FLU_MODELS[drone].T).T
    return body[[0, 1, 3, 2]] if drone in _Z_ORDER else body


def _entry(seq: FakeSequence, idx: int):
    t = seq.t0 + seq.velocity * (idx - 1)
    k3 = (seq.R_true @ raw_body_points(seq.drone).T).T + t
    k2 = project(FAKE_K, k3)
    r_anno = seq.R_true if seq.drone == "02" else seq.R_true @ rot_z(-seq.degree)   # the dataset bug
    folder = f"{seq.scene}_{seq.drone}_{seq.background}_{seq.degree:03d}"
    w, h = np.ptp(k2[:, 0]), np.ptp(k2[:, 1])
    pad = max(0.4 * w, 0.4 * h, 20)
    (x0, y0), (x1, y1) = k2.min(0) - pad, k2.max(0) + pad
    keys = ["key1", "key2", "key3", "key4"]
    entry = {
        "path": f"datasets/scene_{seq.scene}/{seq.scene}_{seq.drone}/{folder}/{idx:04d}.png",
        "seq_type": seq.scene, "drone_type": seq.drone, "background": seq.background,
        "degree": f"{seq.degree:03d}", "filename": f"{idx:04d}.png", "image_size": [1080, 1920],
        "bbox": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1, "width": x1 - x0, "height": y1 - y0},
        "2DKey": {k: v.tolist() for k, v in zip(keys, k2)},
        "3DKey(world)": {k: (v / 0.582).tolist() for k, v in zip(keys, k3)},
        "3DKey(camera)": {k: v.tolist() for k, v in zip(keys, k3)},
        "drone": {
            "2Dlocation": project(FAKE_K, t[None])[0].tolist(),
            "3Dtranslation(world)": (t / 0.582).tolist(),
            "3Drotation(world)": [0.0, 0.0, 0.0],
            "3Dtranslation(camera)": t.tolist(),
            "3Drotation(camera)": Rotation.from_matrix(r_anno).as_euler("xyz").tolist(),
        },
    }
    return f"{folder}_{idx:04d}", entry, folder, k2


def write_fake_dataset(root: Path, sequences: list[FakeSequence]) -> list[FakeSequence]:
    by_pair: dict[tuple, list[FakeSequence]] = {}
    for s in sequences:
        by_pair.setdefault((s.scene, s.drone), []).append(s)
    for (scene, drone), seqs in by_pair.items():
        inner = seqs[0].inner_folder or f"scene_{scene}"
        sub = Path(root) / f"scene_{scene}" / inner / f"{scene}_{drone}"
        sub.mkdir(parents=True, exist_ok=True)
        anno = {"camera_parameters": {"intrinsic": {"camera_matrix_K": FAKE_K.tolist()}}}
        for s in seqs:
            for idx in range(1, s.n_frames + 1):
                key, entry, folder, k2 = _entry(s, idx)
                anno[key] = entry
                img_dir = sub / folder
                img_dir.mkdir(exist_ok=True)
                if idx in s.missing_frames:
                    continue
                img = np.zeros((1080, 1920, 3), np.uint8)
                for x, y in k2:
                    cv2.circle(img, (int(round(x)), int(round(y))), 6, (255, 255, 255), -1)
                cv2.imwrite(str(img_dir / f"{idx:04d}.png"), img)
                (img_dir / f"._{idx:04d}.png").write_bytes(b"junk")          # macOS AppleDouble
        (sub / f"{scene}_{drone}_anno.json").write_text(json.dumps(anno))
        (sub / f"._{scene}_{drone}_anno.json").write_bytes(b"junk")
    return sequences


def default_sequences() -> list[FakeSequence]:
    r1 = Rotation.from_euler("xyz", [0.2, -0.1, 0.7]).as_matrix()
    r2 = Rotation.from_euler("xyz", [-0.3, 0.25, -1.2]).as_matrix()
    return [
        FakeSequence("03", "01", "05", 90, 3, r1, np.array([0.10, -0.05, 3.0])),                       # val, Z-order, bug
        FakeSequence("02", "02", "06", 180, 2, r2, np.array([-0.20, 0.05, 2.5])),                     # train, drone 02
        FakeSequence("13", "01", "01", 270, 2, r2, np.array([0.0, 0.1, 4.0]), inner_folder="scene_12"),  # misnamed folder
        FakeSequence("06", "02", "14", 90, 3, r1, np.array([0.3, 0.0, 3.5]), missing_frames=(2,)),     # test, missing PNG
    ]
