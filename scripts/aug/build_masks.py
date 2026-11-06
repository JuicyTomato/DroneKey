"""SAM output -> accepted masks for the 256 crops and the 640x360 frames, with the quality gate of spec §5.

A mask is accepted when all 4 hubs are inside and its area stays below median of the drone x 2.5 (the hubs alone would
accept a mask that covers the whole crop). The difference between renders is a check, used where its own area is
plausible. Only accepted masks are written: a frame without a mask is never pasted (dronepose.keypoints.composite).
Usage: .venv/bin/python scripts/aug/build_masks.py --inputs $AUG/sam_inputs --sam $AUG/sam_out --out $AUG/masks
Output: crop/, frame/, quality.jsonl, qc/{random,worst}.jpg, rejected.txt.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
from dronepose.data.canonical import read_jsonl  # noqa: E402
from dronepose.data.crops import CropBox, crop_image  # noqa: E402
from dronepose.data.masks import (area_limits, diff_mask, geom_key, keypoints_in_mask, mask_iou, native_to_crop,  # noqa: E402
                                  native_to_frame, soften)

TILE = 160
GATE = 0.99


def measure(job: tuple[dict, Path]) -> dict | None:
    r, sam = job
    m = cv2.imread(str(sam / f"{r['geom']}.png"), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return {**r, "kpts_ok": False, "area": None}
    return {**r, "kpts_ok": keypoints_in_mask(m, np.array(r["kpts"])), "area": float((m > 0).mean())}


def check(job: tuple[dict, list[str], Path, float | None]) -> dict:
    """Agreement with the difference between renders; None when the difference mask is itself implausible."""
    r, paths, sam, limit = job
    box = CropBox(*r["box"])
    crops = [crop_image(cv2.imread(p), box, r["size"]) for p in paths]
    m = cv2.imread(str(sam / f"{r['geom']}.png"), cv2.IMREAD_GRAYSCALE)
    iou = None
    if m is not None and len(crops) >= 2:
        d = diff_mask(crops, np.array(r["kpts"]))
        if limit is None or d.mean() <= limit:
            iou = mask_iou(d, m)
    return {"geom": r["geom"], "diff_iou": iou}


def tile(r: dict, inputs: Path, sam: Path) -> np.ndarray:
    img = cv2.imread(str(inputs / "images" / f"{r['geom']}.jpg"))
    m = cv2.imread(str(sam / f"{r['geom']}.png"), cv2.IMREAD_GRAYSCALE)
    a = (m if m is not None else np.zeros(img.shape[:2], np.uint8))[..., None] / 255.0
    vis = (img * a + np.array([0, 255, 0]) * (1 - a)).astype(np.uint8)
    return np.hstack([cv2.resize(img, (TILE, TILE)), cv2.resize(vis, (TILE, TILE))])


def montage(rows: list[dict], inputs: Path, sam: Path, path: Path, per_row: int = 5) -> None:
    tiles = [tile(r, inputs, sam) for r in rows]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % per_row)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--sam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    inputs, sam, out = Path(args.inputs), Path(args.sam), Path(args.out)
    for sub in ("crop", "frame", "qc"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (inputs / "index.jsonl").read_text().splitlines() if l.strip()][:args.limit]
    renders: dict[str, list[str]] = defaultdict(list)
    for rec in read_jsonl(Path(args.processed) / "canonical_train.jsonl"):
        renders[geom_key(rec["frame_id"])].append(rec["image_path"])

    with Pool(args.workers) as pool:
        rows = pool.map(measure, [(r, sam) for r in rows], chunksize=32)
        areas: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r["area"] is not None:
                areas[r["drone"]].append(r["area"])
        limits = area_limits(areas)
        checks = pool.map(check, [(r, renders[r["geom"]], sam, limits.get(r["drone"])) for r in rows], chunksize=8)
    iou = {c["geom"]: c["diff_iou"] for c in checks}

    accepted = 0
    with open(out / "quality.jsonl", "w") as q, open(out / "rejected.txt", "w") as rej:
        for r in rows:
            r["diff_iou"] = iou[r["geom"]]
            r["accepted"] = bool(r["kpts_ok"] and r["area"] is not None and r["area"] <= limits[r["drone"]])
            q.write(json.dumps({k: r[k] for k in ("geom", "drone", "kpts_ok", "area", "diff_iou", "accepted")}) + "\n")
            if not r["accepted"]:
                rej.write(r["geom"] + "\n")
                continue
            accepted += 1
            m = soften(cv2.imread(str(sam / f"{r['geom']}.png"), cv2.IMREAD_GRAYSCALE))
            cv2.imwrite(str(out / "crop" / f"{r['geom']}.png"), native_to_crop(m))
            cv2.imwrite(str(out / "frame" / f"{r['geom']}.png"), native_to_frame(m, CropBox(*r["box"])))

    print("drone  frames  hubs ok  area median  limit  accepted  diff IoU median")
    for d in sorted(areas):
        x = [r for r in rows if r["drone"] == d]
        ious = [r["diff_iou"] for r in x if r["diff_iou"] is not None]
        print(f"{d:5s}  {len(x):6d}  {np.mean([r['kpts_ok'] for r in x]):7.3f}  {np.median(areas[d]):11.3f}  "
              f"{limits[d]:5.3f}  {np.mean([r['accepted'] for r in x]):8.3f}  {np.median(ious) if ious else float('nan'):.3f}")
    ok = float(np.mean([r["kpts_ok"] for r in rows]))
    print(f"gate: hubs inside {ok:.4f} (need {GATE}) -> {'PASS' if ok >= GATE else 'FAIL'}; accepted {accepted}/{len(rows)}")

    random.seed(0)
    montage(random.sample(rows, min(50, len(rows))), inputs, sam, out / "qc" / "random.jpg")
    worst = sorted((r for r in rows if r["diff_iou"] is not None), key=lambda r: r["diff_iou"])[:50]
    if worst:
        montage(worst, inputs, sam, out / "qc" / "worst.jpg")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
