"""Run YOLO-Pose on the cached crops of val/test and write keypoints in ORIGINAL frame coordinates (setting A).

Kaggle (after pip install ultralytics, inputs: dronepose-cache, dronepose-code, weights):
    python $CODE/scripts/kaggle/export_keypoints.py --weights <best.pt> --splits val test
Output: /kaggle/working/keypoints/keypoints_{split}.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # kaggle_paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # dronepose without pip install
from kaggle_paths import find_dataset  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from dronepose.data.crops import CROP_SIZE, CropBox  # noqa: E402
from dronepose.keypoints.export import crop_result_to_frame  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--splits", nargs="+", default=["val", "test"])
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--out", default="/kaggle/working/keypoints")
args = ap.parse_args()

crops = find_dataset("dronepose-cache") / "yolo_crops"
model = YOLO(args.weights)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for split in args.splits:
    boxes = {}
    with open(crops / f"crops_{split}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            boxes[row["frame_id"]] = CropBox(*row["box"])
    images = sorted((crops / "images" / split).glob("*.jpg"))
    n = 0
    with open(out / f"keypoints_{split}.jsonl", "w") as f:
        for res in model.predict([str(p) for p in images], imgsz=CROP_SIZE, batch=args.batch, stream=True,
                                 verbose=False):
            fid = Path(res.path).stem
            kp = res.keypoints
            xy = None if kp is None else kp.xy.cpu().numpy()
            kconf = None if kp is None or kp.conf is None else kp.conf.cpu().numpy()
            bconf = None if res.boxes is None else res.boxes.conf.cpu().numpy()
            kpts = crop_result_to_frame(xy, kconf, bconf, boxes[fid], CROP_SIZE)
            f.write(json.dumps({"frame_id": fid, "kpts_pred": kpts.tolist()}, allow_nan=True) + "\n")
            n += 1
    print(f"{split}: {n} frame -> {out / f'keypoints_{split}.jsonl'}")
