"""Reader for the raw 6DroneSyn annotations (DroneKey++). Fixes are applied downstream, never on disk."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from dronepose.common.geometry import euler_xyz, rot_z

DRONE_IDS = ("01", "02", "03", "04", "05", "06", "08")
Z_ORDER_DRONES = frozenset({"01", "04", "05", "06", "08"})
_TO_CIRCULAR = np.array([0, 1, 3, 2])
_KEYS = ("key1", "key2", "key3", "key4")
FPS = 30
# Folder scene numbers (11-14 = paper 10-13), PROJECT_MEMORY §2.
SPLIT_BY_SCENE = {
    **{s: "train" for s in ("01", "02", "04", "05", "08", "11", "12", "13")},
    **{s: "val" for s in ("03", "09", "14")},
    **{s: "test" for s in ("06", "07")},
}


@dataclass(frozen=True)
class RawFrame:
    frame_id: str
    image_path: Path
    scene: str
    drone: str
    background: str
    degree: int
    frame_idx: int
    K: np.ndarray
    bbox_xyxy: np.ndarray
    kpts2d: np.ndarray        # raw annotation order
    kpts3d_cam: np.ndarray    # raw annotation order, camera frame, dataset units
    t_cam_blender: np.ndarray # Blender object origin in camera frame
    rot_cam_euler: np.ndarray # raw '3Drotation(camera)' (radians, Blender XYZ), missing the 'degree' yaw


def iter_annotation_files(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).glob("scene_*/*/*/*_anno.json") if not p.name.startswith("._"))


def parse_annotation_file(path: Path) -> list[RawFrame]:
    data = json.loads(Path(path).read_text())
    K = np.array(data["camera_parameters"]["intrinsic"]["camera_matrix_K"], dtype=float)
    frames = []
    for fid, e in data.items():
        if fid == "camera_parameters":
            continue
        b = e["bbox"]
        frames.append(RawFrame(
            frame_id=fid,
            image_path=Path(path).parent / e["path"].split("/")[-2] / e["filename"],
            scene=e["seq_type"], drone=e["drone_type"], background=e["background"],
            degree=int(e["degree"]), frame_idx=int(e["filename"][:4]), K=K,
            bbox_xyxy=np.array([b["x_min"], b["y_min"], b["x_max"], b["y_max"]], dtype=float),
            kpts2d=np.array([e["2DKey"][k] for k in _KEYS], dtype=float),
            kpts3d_cam=np.array([e["3DKey(camera)"][k] for k in _KEYS], dtype=float),
            t_cam_blender=np.array(e["drone"]["3Dtranslation(camera)"], dtype=float),
            rot_cam_euler=np.array(e["drone"]["3Drotation(camera)"], dtype=float),
        ))
    return frames


def iter_frames(root: Path) -> Iterator[RawFrame]:
    for path in iter_annotation_files(root):
        yield from parse_annotation_file(path)


def to_circular(drone: str, arr: np.ndarray) -> np.ndarray:
    return arr[_TO_CIRCULAR].copy() if drone in Z_ORDER_DRONES else arr.copy()


def corrected_rotation(frame: RawFrame) -> np.ndarray:
    """Blender body -> camera rotation. The field misses the 'degree' yaw for every drone except 02 (§3.10)."""
    r = euler_xyz(frame.rot_cam_euler)
    return r if frame.drone == "02" else r @ rot_z(frame.degree)
