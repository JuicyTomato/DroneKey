"""MAV6D records -> 640x360 JPEG frames for detector C, as build_yolo_frames.py does for 6DroneSyn (INTER_AREA, q92).

  .venv/bin/python scripts/real/mav6d_frames.py [--canonical data/processed/mav6d_gt.jsonl]
      [--out-dir /Volumes/MakeMePasta/MAV6D/frames640]
Writes OUT/images/mav6d/<frame_id>.jpg and OUT/frames_mav6d.jsonl (the layout export_frames_c.py reads).
Originals are only read; existing frames are skipped (resumable).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical", default="data/processed/mav6d_gt.jsonl")
    ap.add_argument("--out-dir", default="/Volumes/MakeMePasta/MAV6D/frames640")
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args(argv)
    recs = [json.loads(line) for line in open(args.canonical)]
    out = Path(args.out_dir)
    (out / "images" / "mav6d").mkdir(parents=True, exist_ok=True)

    def one(r):
        dst = out / "images" / "mav6d" / f"{r['frame_id']}.jpg"
        if dst.exists():
            return
        img = cv2.imread(r["image_path"])
        if img is None:
            raise FileNotFoundError(r["image_path"])
        h, w = img.shape[:2]
        small = cv2.resize(img, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), small, [cv2.IMWRITE_JPEG_QUALITY, 92])

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, recs))
    with open(out / "frames_mav6d.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({"frame_id": r["frame_id"]}) + "\n")
    print(f"{len(recs)} frames -> {out}")


if __name__ == "__main__":
    main()
