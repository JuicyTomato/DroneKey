"""Setting C: full 640x360 frames -> keypoints + box + class in 1920x1080 frame pixels (top box by confidence).

usage (ultralytics venv, from the repo root):
  python scripts/export_frames_c.py --weights W --split val|test|mav6d --out OUT.jsonl
      [--frames-dir data/processed/yolo_frames]
      [--device mps] [--batch 32] [--limit N] [--all-boxes [--conf-min 0.01]]
Row: {"frame_id", "kpts_pred", "box", "box_conf", "cls"}; frames without detection stay NaN / box null.
--all-boxes: predict at conf --conf-min and add "boxes" ([x1, y1, x2, y2, conf, cls] of every detection, frame
pixels); the top-box fields still use only boxes with conf >= 0.25 (the Ultralytics default), so they match the
export without the option. Use --batch 8 with --all-boxes: at conf 0.01 the Ultralytics NMS can exceed its time limit
per batch (2 s + 0.05 s per image) and silently return no boxes for the last images of a batch (seen at batch 32:
39 of 12600 val frames empty).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # dronepose without pip install
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from dronepose.keypoints.two_stage import all_boxes, top_detection_row  # noqa: E402

DEFAULT_CONF = 0.25   # Ultralytics predict default: threshold of the top-box fields


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test", "mav6d"])
    ap.add_argument("--frames-dir", default="data/processed/yolo_frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all-boxes", action="store_true")
    ap.add_argument("--conf-min", type=float, default=0.01, help="predict threshold with --all-boxes")
    args = ap.parse_args(argv)
    extra = {"conf": args.conf_min} if args.all_boxes else {}

    frames = Path(args.frames_dir)
    with open(frames / f"frames_{args.split}.jsonl") as f:
        ids = [json.loads(l)["frame_id"] for l in f if l.strip()][:args.limit]
    images = [frames / "images" / args.split / f"{i}.jpg" for i in ids]
    model = YOLO(args.weights)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n, nodet = time.time(), 0, 0
    with open(out, "w") as f:
        for i in range(0, len(images), args.batch):
            chunk = [str(p) for p in images[i:i + args.batch]]
            for res in model.predict(chunk, imgsz=640, device=args.device, verbose=False, **extra):
                has = res.boxes is not None and len(res.boxes) > 0
                if has:
                    xyxy, conf, cls = (v.cpu().numpy() for v in (res.boxes.xyxy, res.boxes.conf, res.boxes.cls))
                    kxy = res.keypoints.xy.cpu().numpy()
                    kconf = None if res.keypoints.conf is None else res.keypoints.conf.cpu().numpy()
                else:
                    xyxy, conf, cls, kxy, kconf = np.zeros((0, 4)), np.zeros(0), np.zeros(0), None, None
                    nodet += 1
                scale = 1920 / res.orig_shape[1]
                boxes = all_boxes(xyxy, conf, cls, scale) if args.all_boxes else None
                if args.all_boxes and has:
                    keep = conf >= DEFAULT_CONF
                    xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
                    kxy, kconf = kxy[keep], None if kconf is None else kconf[keep]
                    if len(conf) == 0:
                        nodet += 1
                row = top_detection_row(Path(res.path).stem, xyxy, conf, cls, kxy, kconf, scale)
                if boxes is not None:
                    row["boxes"] = boxes
                f.write(json.dumps(row, allow_nan=True) + "\n")
                n += 1
            if i // args.batch % 80 == 0:
                print(f"{args.weights}: {n}/{len(images)} {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"done {n} frames, {nodet} no detection, {dt:.0f}s ({1000 * dt / max(n, 1):.1f} ms/frame) -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
