"""Canonical records -> YOLO-Pose full frames (setting C): 1920x1080 PNG resized to 640x360 JPEG, all splits.

Labels are normalised to the frame, so they do not depend on the resolution; the box follows the crop labels
(keypoint extent + BOX_PAD), so setting B can rebuild the setting-A crop from a detected box (crop_box_from_det_box).
Reads the original dataset only (never writes there). Resumable: frames already written are skipped.
Usage: python scripts/build_yolo_frames.py [--width 640] [--workers 6]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.data.export_yolo import CLASS_IDS, write_data_yaml, yolo_label_line

ap = argparse.ArgumentParser()
ap.add_argument("--processed", default="data/processed")
ap.add_argument("--kaggle-root", default="/kaggle/input/dronepose-frames/yolo_frames")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()
proc = Path(args.processed)
out = proc / "yolo_frames"


def build(job: tuple[str, dict]) -> str | None:
    split, r = job
    img_path = out / "images" / split / f"{r['frame_id']}.jpg"
    lab_path = out / "labels" / split / f"{r['frame_id']}.txt"
    if img_path.exists() and lab_path.exists():
        return r["frame_id"]
    img = cv2.imread(r["image_path"])
    if img is None:
        return None
    h, w = img.shape[:2]
    small = cv2.resize(img, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(img_path), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lab_path.write_text(yolo_label_line(CLASS_IDS[r["drone"]], np.array(r["kpts2d"]), size=(w, h)) + "\n")
    return r["frame_id"]


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        recs = read_jsonl(proc / f"canonical_{split}.jsonl")
        done, missing = [], []
        with Pool(args.workers) as pool:
            for i, (r, fid) in enumerate(zip(recs, pool.imap(build, [(split, r) for r in recs], chunksize=16))):
                (done if fid else missing).append(r["frame_id"])
                if (i + 1) % 2000 == 0:
                    print(f"{split}: {i + 1}/{len(recs)}", flush=True)
        (out / f"frames_{split}.jsonl").write_text("".join(json.dumps({"frame_id": f}) + "\n" for f in done))
        print(f"{split}: {len(done)} frame, {len(missing)} immagini mancanti {missing[:5]}", flush=True)
    write_data_yaml(out / "data.yaml", args.kaggle_root)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"yolo_frames: {size / 1024 ** 3:.2f} GB", flush=True)
