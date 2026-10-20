"""Run YOLO-Pose on the cached crops of val/test and write keypoints in ORIGINAL frame coordinates (setting A).

Kaggle (after pip install ultralytics, inputs: dronepose-cache, dronepose-code, weights):
    python -u $CODE/scripts/kaggle/export_keypoints.py --weights <best.pt> --splits val test
Output: /kaggle/working/keypoints/keypoints_{split}.jsonl
Every step prints elapsed time (flushed), so a hang shows up as the last line printed.
"""
import time

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)   # flush: stdout is a pipe under %%bash


log("start, importing torch")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

log(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}"
    + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " -> running on CPU"))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # kaggle_paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # dronepose without pip install
from kaggle_paths import find_dataset  # noqa: E402

log("importing ultralytics")
from ultralytics import YOLO  # noqa: E402

from dronepose.data.crops import CROP_SIZE, CropBox  # noqa: E402
from dronepose.keypoints.export import crop_result_to_frame  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--splits", nargs="+", default=["val", "test"])
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--out", default="/kaggle/working/keypoints")
ap.add_argument("--crops", default=None, help="crops directory (default: dronepose-cache/yolo_crops)")
ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N crops per split")
args = ap.parse_args()
log(f"args: {vars(args)}")

crops = Path(args.crops) if args.crops else find_dataset("dronepose-cache") / "yolo_crops"
log(f"crops: {crops}")
model = YOLO(args.weights)
log(f"model loaded: {args.weights}")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for split in args.splits:
    boxes = {}
    with open(crops / f"crops_{split}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            boxes[row["frame_id"]] = CropBox(*row["box"])
    log(f"{split}: {len(boxes)} boxes read, listing images")
    images = sorted((crops / "images" / split).glob("*.jpg"))[:args.limit]
    log(f"{split}: {len(images)} images")
    n = 0
    with open(out / f"keypoints_{split}.jsonl", "w") as f:
        # explicit chunks: some ultralytics versions ignore batch= for a list source and run it all as one batch (OOM)
        for i in range(0, len(images), args.batch):
            first = i == 0
            if first:
                log(f"{split}: first predict (includes GPU warmup)")
            tc = time.time()
            chunk = [str(p) for p in images[i:i + args.batch]]
            for res in model.predict(chunk, imgsz=CROP_SIZE, verbose=False):
                fid = Path(res.path).stem
                kp = res.keypoints
                xy = None if kp is None else kp.xy.cpu().numpy()
                kconf = None if kp is None or kp.conf is None else kp.conf.cpu().numpy()
                bconf = None if res.boxes is None else res.boxes.conf.cpu().numpy()
                kpts = crop_result_to_frame(xy, kconf, bconf, boxes[fid], CROP_SIZE)
                f.write(json.dumps({"frame_id": fid, "kpts_pred": kpts.tolist()}, allow_nan=True) + "\n")
                n += 1
            if first or i // args.batch % 20 == 0:
                log(f"{split}: {n}/{len(images)} (last chunk {time.time() - tc:.2f}s)")
    log(f"{split}: {n} frame -> {out / f'keypoints_{split}.jsonl'}")
log("done")
