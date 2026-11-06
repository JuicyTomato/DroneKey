"""Zips of the two private Kaggle datasets of copy-paste: dronepose-masks (crop/, frame/) and dronepose-bgs.

Usage: .venv/bin/python scripts/aug/package_aug.py --aug $AUG
"""
import argparse
import zipfile
from pathlib import Path


def zip_folder(src: Path, out: Path, subs: list[str] | None = None) -> Path:
    """Stored zip (JPEG and PNG do not compress), paths relative to `src`, no AppleDouble `._*` files."""
    roots = [src / s for s in subs] if subs else [src]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for root in roots:
            for p in sorted(root.rglob("*")):
                if p.is_file() and not p.name.startswith("."):
                    z.write(p, p.relative_to(src).as_posix())
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aug", required=True)
    args = ap.parse_args(argv)
    aug = Path(args.aug)
    for out in (zip_folder(aug / "masks", aug / "dronepose-masks.zip", ["crop", "frame"]),
                zip_folder(aug / "backgrounds", aug / "dronepose-bgs.zip")):
        print(f"{out} {out.stat().st_size / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    main()
