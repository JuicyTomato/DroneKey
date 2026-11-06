"""One crop per frame geometry of the 6DroneSyn train split, at native resolution, for SAM on a GPU machine.

The renders of a frame on different backgrounds share keypoints and box, so one mask serves all of them; --render picks
which one goes to SAM (a second pass with --render 1 can retry the frames that failed the quality gate).
Usage: .venv/bin/python scripts/aug/build_sam_inputs.py --out $AUG/sam_inputs [--render 0] [--limit N] [--only FILE]
Reads the original PNGs only (never writes there). Resumable.
"""
import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
from dronepose.data.canonical import read_jsonl  # noqa: E402
from dronepose.data.crops import crop_box_from_kpts, crop_image, frame_to_crop  # noqa: E402
from dronepose.data.masks import geom_key  # noqa: E402

MAX_SIDE = 1024  # SAM works at 1024 anyway


def select_renders(records: list[dict], render: int) -> list[dict]:
    """One record per geometry: the `render`-th by frame_id, the last one when the geometry has fewer renders."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[geom_key(r["frame_id"])].append(r)
    return [sorted(v, key=lambda r: r["frame_id"])[min(render, len(v) - 1)] for _, v in sorted(groups.items())]


def build(job: tuple[dict, Path]) -> dict | None:
    r, out = job
    kpts = np.array(r["kpts2d"], float)
    box = crop_box_from_kpts(kpts)
    size = min(round(box.side), MAX_SIDE)
    geom = geom_key(r["frame_id"])
    path = out / "images" / f"{geom}.jpg"
    if not path.exists():
        img = cv2.imread(r["image_path"])
        if img is None:
            return None
        cv2.imwrite(str(path), crop_image(img, box, size), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"geom": geom, "frame_id": r["frame_id"], "drone": r["drone"], "box": [box.x0, box.y0, box.side],
            "size": size, "kpts": frame_to_crop(kpts, box, size).tolist()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", type=int, default=0)
    ap.add_argument("--only", default=None, help="text file with the geometries to process, one per line")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    picks = select_renders(read_jsonl(Path(args.processed) / "canonical_train.jsonl"), args.render)
    if args.only:
        keep = set(Path(args.only).read_text().split())
        picks = [r for r in picks if geom_key(r["frame_id"]) in keep]
    picks = picks[:args.limit]
    with Pool(args.workers) as pool:
        rows = [row for row in pool.imap(build, [(r, out) for r in picks], chunksize=8) if row is not None]
    (out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)}/{len(picks)} crops -> {out}")


if __name__ == "__main__":
    main()
