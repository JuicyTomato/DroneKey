"""Small Kaggle dataset `dronepose-bench` for scripts/kaggle/bench_fps.py: the 550 evenly spaced val frames that the
bench picks (same choice as bench_fps.py on the full frames_val.jsonl) plus the C and crop weights.

usage: .venv/bin/python scripts/package_bench.py --c-weights W --crop-weights W [--frames 550]
Writes data/dronepose-bench.zip: yolo_frames/frames_val.jsonl (only those frames), yolo_frames/images/val/*.jpg,
c.pt, crop.pt. On Kaggle: bench_fps.py --frames-dir $B/yolo_frames --c-weights $B/c.pt --crop-weights $B/crop.pt.
Latency does not depend on the weight values, only on the architecture.
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--c-weights", required=True)
ap.add_argument("--crop-weights", required=True)
ap.add_argument("--frames", type=int, default=550)
ap.add_argument("--frames-dir", default="data/processed/yolo_frames")
ap.add_argument("--out", default="data/dronepose-bench.zip")
args = ap.parse_args()
src = Path(args.frames_dir)
rows = [json.loads(l) for l in open(src / "frames_val.jsonl") if l.strip()]
pick = np.linspace(0, len(rows) - 1, min(args.frames, len(rows))).astype(int)   # as bench_fps.py
with zipfile.ZipFile(args.out, "w") as z:
    z.writestr("yolo_frames/frames_val.jsonl", "".join(json.dumps(rows[i]) + "\n" for i in pick))
    for i in pick:
        fid = rows[i]["frame_id"]
        z.write(src / "images" / "val" / f"{fid}.jpg", f"yolo_frames/images/val/{fid}.jpg", compress_type=zipfile.ZIP_STORED)
    z.write(args.c_weights, "c.pt")
    z.write(args.crop_weights, "crop.pt")
print(f"{args.out}: {len(pick)} frames, {Path(args.out).stat().st_size / 1024 ** 2:.1f} MB")
