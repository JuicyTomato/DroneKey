"""Background images for copy-paste: a fixed random subset of COCO train2017, long side 640, JPEG q90.

Never frames of MAV6D: the real test must stay unseen, also as a background.
Usage: .venv/bin/python scripts/aug/build_backgrounds.py --coco-dir $AUG/coco/train2017 --out $AUG/backgrounds
"""
import argparse
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2

LONG_SIDE = 640


def check_source(path: Path) -> None:
    if "mav6d" in str(path).lower():
        raise ValueError(f"{path}: frames of MAV6D cannot be backgrounds")


def pick_images(paths: list[Path], n: int, seed: int) -> list[Path]:
    """`n` paths chosen by hash of (seed, name): independent of the listing order and of the machine."""
    return sorted(paths, key=lambda p: hashlib.sha1(f"{seed}/{p.name}".encode()).hexdigest())[:n]


def build(job: tuple[Path, Path]) -> bool:
    src, dst = job
    if dst.exists():
        return True
    img = cv2.imread(str(src))
    if img is None:
        return False
    h, w = img.shape[:2]
    s = LONG_SIDE / max(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    src, out = Path(args.coco_dir), Path(args.out)
    check_source(src)
    out.mkdir(parents=True, exist_ok=True)
    paths = [p for p in src.glob("*.jpg") if not p.name.startswith(".")]
    picks = pick_images(paths, args.n, args.seed)
    with Pool(args.workers) as pool:
        ok = sum(pool.imap_unordered(build, [(p, out / p.name) for p in picks], chunksize=32))
    print(f"{ok}/{len(picks)} backgrounds -> {out}")


if __name__ == "__main__":
    main()
