"""Zip data/processed/yolo_frames into data/dronepose-frames.zip for the private Kaggle dataset `dronepose-frames`.

Flat layout (yolo_frames/ at the zip root, like yolo_crops/ in dronepose-cache). JPEGs are stored, not deflated.
"""
import zipfile
from pathlib import Path

src = Path("data/processed/yolo_frames")
dst = Path("data/dronepose-frames.zip")
files = sorted(p for p in src.rglob("*") if p.is_file() and not p.name.startswith("._") and p.name != ".DS_Store")
with zipfile.ZipFile(dst, "w") as z:
    for p in files:
        kind = zipfile.ZIP_STORED if p.suffix == ".jpg" else zipfile.ZIP_DEFLATED
        z.write(p, p.relative_to(src.parent), compress_type=kind)
print(f"{dst}: {len(files)} file, {dst.stat().st_size / 1024 ** 3:.2f} GB")
