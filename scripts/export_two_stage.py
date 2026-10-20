"""Setting B: boxes of setting C -> crop from the original 1920x1080 PNG (rule and JPEG q95 of setting A)
-> crop pose model -> keypoints in frame pixels. Frames where C found nothing stay as no detection.

usage (ultralytics venv, from the repo root):
  python scripts/export_two_stage.py --det DETS.jsonl --weights W --split val|test --out OUT.jsonl
      [--device mps] [--gt-box] [--limit N] [--threads 8]
--gt-box: take the label box from the GT keypoints instead of C (must reproduce the setting-A crops).
PNGs are read from `image_path` of the canonical records (read-only).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # dronepose without pip install
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from dronepose.data.crops import CROP_SIZE, label_box_xyxy  # noqa: E402
from dronepose.keypoints.export import crop_result_to_frame  # noqa: E402
from dronepose.keypoints.predictors import no_detection  # noqa: E402
from dronepose.keypoints.two_stage import crop_from_detection  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", required=True, help="JSONL of setting C (box per frame)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--canonical", default=None, help="default data/processed/canonical_{split}.jsonl")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--gt-box", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    with open(args.canonical or f"data/processed/canonical_{args.split}.jsonl") as f:
        recs = {r["frame_id"]: r for r in map(json.loads, (l for l in f if l.strip()))}
    with open(args.det) as f:
        rows = [json.loads(l) for l in f if l.strip()][:args.limit]

    def load(row):
        r = recs[row["frame_id"]]
        xyxy = label_box_xyxy(np.array(r["kpts2d"], float)) if args.gt_box else row["box"]
        if xyxy is None:
            return row["frame_id"], None, None
        img = cv2.imread(r["image_path"])          # read-only original
        if img is None:
            raise FileNotFoundError(r["image_path"])
        crop, box = crop_from_detection(img, xyxy)
        return row["frame_id"], box, crop

    model = YOLO(args.weights)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n, B = time.time(), 0, 64
    with open(out, "w") as f, ThreadPoolExecutor(args.threads) as ex:
        for i in range(0, len(rows), B):
            items = list(ex.map(load, rows[i:i + B]))
            todo = [it for it in items if it[1] is not None]
            res = model.predict([it[2] for it in todo], imgsz=CROP_SIZE, device=args.device,
                                verbose=False) if todo else []
            pred = {}
            for (fid, box, _), r in zip(todo, res):
                kp = r.keypoints
                xy = None if kp is None else kp.xy.cpu().numpy()
                kc = None if kp is None or kp.conf is None else kp.conf.cpu().numpy()
                bc = None if r.boxes is None else r.boxes.conf.cpu().numpy()
                pred[fid] = crop_result_to_frame(xy, kc, bc, box, CROP_SIZE)
            for fid, _, _ in items:
                f.write(json.dumps({"frame_id": fid, "kpts_pred": pred.get(fid, no_detection()).tolist()},
                                   allow_nan=True) + "\n")
                n += 1
            if i // B % 40 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"done {n} frames in {dt:.0f}s ({1000 * dt / max(n, 1):.1f} ms/frame incl. PNG read) -> {out}", flush=True)


if __name__ == "__main__":
    main()
