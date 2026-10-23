"""2DroneKey (1920x1080 PNG + COCO-like JSON, 4 propeller points) -> 256 px crops + YOLO-Pose labels, same layout as yolo_crops."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from dronepose.data.crops import crop_box_from_kpts, crop_image, frame_to_crop
from dronepose.data.export_yolo import yolo_label_line

SPLITS = ("train", "valid", "test")
OUT_SPLIT = {"train": "train", "valid": "val", "test": "test"}  # source folder -> output split
JSON_NAME = {"train": "train", "valid": "val", "test": "test"}  # source folder -> instances_<x>2017.json
KAGGLE_ROOT = "/kaggle/input/dronekey-2dk"


def load_2dk_annotations(json_path: Path) -> dict[str, np.ndarray]:
    """file name without extension -> (4, 2) pixel keypoints ordered by category_id 1-4 (propeller index)."""
    data = json.loads(Path(json_path).read_text())
    names = {im["id"]: Path(im["file_name"]).stem for im in data["images"]}
    per_image: dict[str, dict[int, list]] = defaultdict(dict)
    for a in data["annotations"]:
        name = names[a["image_id"]]
        if a["category_id"] in per_image[name]:
            raise ValueError(f"{json_path}: image {name} has two annotations for category {a['category_id']}")
        per_image[name][a["category_id"]] = a["keypoints"][:2]
    out = {}
    for name in names.values():
        cats = per_image.get(name, {})
        if sorted(cats) != [1, 2, 3, 4]:
            raise ValueError(f"{json_path}: image {name} has categories {sorted(cats)}, expected exactly 1-4")
        out[name] = np.array([cats[c] for c in (1, 2, 3, 4)], float)
    return out


def build(src: Path, out: Path, limit: int | None = None, splits: tuple[str, ...] = SPLITS,
          kaggle_root: str = KAGGLE_ROOT) -> None:
    src, out = Path(src), Path(out)
    for folder in splits:
        split = OUT_SPLIT[folder]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        ann = load_2dk_annotations(src / f"instances_{JSON_NAME[folder]}2017.json")
        meta, gt = [], []
        t0 = time.time()
        for i, (fid, k2) in enumerate(sorted(ann.items())):
            if limit is not None and i >= limit:
                break
            img = cv2.imread(str(src / folder / f"{fid}.png"))
            if img is None:
                raise FileNotFoundError(src / folder / f"{fid}.png")
            box = crop_box_from_kpts(k2)
            cv2.imwrite(str(out / "images" / split / f"{fid}.jpg"), crop_image(img, box), [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / split / f"{fid}.txt").write_text(yolo_label_line(0, frame_to_crop(k2, box)) + "\n")
            meta.append({"frame_id": fid, "box": [box.x0, box.y0, box.side]})
            gt.append({"frame_id": fid, "kpts2d": k2.tolist()})
            if (i + 1) % 1000 == 0:
                print(f"{split} {i + 1}/{len(ann)} {time.time() - t0:.0f}s", flush=True)
        (out / f"crops_{split}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
        (out / f"gt_{split}.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gt))
        print(split, len(meta), f"{time.time() - t0:.0f}s", flush=True)
    cfg = {"path": kaggle_root, "train": "images/train", "val": "images/val", "test": "images/test",
           "kpt_shape": [4, 3], "flip_idx": [1, 0, 3, 2], "names": {0: "drone"}}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Volumes/MakeMePasta/2DroneKey/images")
    ap.add_argument("--out", default="data/processed/dronekey_2dk")
    ap.add_argument("--limit", type=int, default=None, help="first N images per split (smoke test)")
    ap.add_argument("--kaggle-root", default=KAGGLE_ROOT)
    a = ap.parse_args()
    build(Path(a.src), Path(a.out), a.limit, kaggle_root=a.kaggle_root)
