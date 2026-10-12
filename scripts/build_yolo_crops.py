"""Canonical records -> YOLO-Pose crops (all splits) + full frames of one sequence for the skeleton video."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.data.crops import crop_box_from_kpts, crop_image, frame_to_crop
from dronepose.data.export_yolo import CLASS_IDS, write_data_yaml, yolo_label_line

ap = argparse.ArgumentParser()
ap.add_argument("--processed", default="data/processed")
ap.add_argument("--kaggle-root", default="/kaggle/input/dronepose-cache/yolo_crops")
ap.add_argument("--sequence", default="14_03_", help="frame_id prefix of the val sequence used by the skeleton")
args = ap.parse_args()
proc = Path(args.processed)
out = proc / "yolo_crops"

for split in ("train", "val", "test"):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    meta = []
    for r in read_jsonl(proc / f"canonical_{split}.jsonl"):
        img = cv2.imread(r["image_path"])
        k2 = np.array(r["kpts2d"])
        box = crop_box_from_kpts(k2)
        cv2.imwrite(str(out / "images" / split / f"{r['frame_id']}.jpg"), crop_image(img, box),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / split / f"{r['frame_id']}.txt").write_text(
            yolo_label_line(CLASS_IDS[r["drone"]], frame_to_crop(k2, box)) + "\n")
        meta.append({"frame_id": r["frame_id"], "box": [box.x0, box.y0, box.side]})
    (out / f"crops_{split}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    print(split, len(meta))
write_data_yaml(out / "data.yaml", args.kaggle_root)

# full frames of the first background of the chosen val sequence
val = [r for r in read_jsonl(proc / "canonical_val.jsonl") if r["frame_id"].startswith(args.sequence)]
first_bg = min(r["background"] for r in val)
seq = [r for r in val if r["background"] == first_bg]
seq_name = f"{args.sequence}{first_bg}"
fdir = proc / "frames" / seq_name
fdir.mkdir(parents=True, exist_ok=True)
for r in seq:
    cv2.imwrite(str(fdir / f"{r['frame_id']}.jpg"), cv2.imread(r["image_path"]), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("sequenza", seq_name, len(seq), "frame")
