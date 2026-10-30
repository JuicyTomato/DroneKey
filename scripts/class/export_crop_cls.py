"""Crop classifier on val: box of setting C -> crop from the original PNG (as setting B) -> 7 class probabilities.

usage (ultralytics venv, from the repo root):
  python scripts/class/export_crop_cls.py --det DETS.jsonl --weights W --out OUT.jsonl [--device mps] [--threads 8]
Row: {"frame_id", "probs": {drone: p} | null}; frames where C found nothing get null. The test split needs --allow-test
(with --canonical data/processed/canonical_test.jsonl). PNGs are read from `image_path` of the canonical records
(read-only). Probabilities are keyed by the model's class names.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from dronepose.data.crops import CROP_SIZE  # noqa: E402
from dronepose.keypoints.class_scores import probs_row  # noqa: E402
from dronepose.keypoints.two_stage import crop_from_detection  # noqa: E402
from dronepose.pose.replay import check_no_test  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", required=True, help="JSONL of setting C (box per frame)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)

    with open(args.canonical) as f:
        records = [json.loads(l) for l in f if l.strip()]
    check_no_test(records, allow_test=args.allow_test)
    recs = {r["frame_id"]: r for r in records}
    with open(args.det) as f:
        rows = [json.loads(l) for l in f if l.strip()][:args.limit]

    def load(row):
        if row["box"] is None:
            return row["frame_id"], None
        r = recs[row["frame_id"]]
        img = cv2.imread(r["image_path"])          # read-only original
        if img is None:
            raise FileNotFoundError(r["image_path"])
        return row["frame_id"], crop_from_detection(img, row["box"])[0]

    model = YOLO(args.weights)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n, B = time.time(), 0, 64
    with open(out, "w") as f, ThreadPoolExecutor(args.threads) as ex:
        for i in range(0, len(rows), B):
            items = list(ex.map(load, rows[i:i + B]))
            todo = [it for it in items if it[1] is not None]
            res = model.predict([it[1] for it in todo], imgsz=CROP_SIZE, device=args.device,
                                verbose=False) if todo else []
            pred = {fid: r.probs.data.cpu().numpy() for (fid, _), r in zip(todo, res)}
            for fid, _ in items:
                f.write(json.dumps(probs_row(fid, pred.get(fid), model.names)) + "\n")
                n += 1
            if i // B % 40 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"done {n} frames in {dt:.0f}s ({1000 * dt / max(n, 1):.1f} ms/frame incl. PNG read) -> {out}", flush=True)


if __name__ == "__main__":
    main()
