"""Build (or resume) the SHA-256 manifest of a dataset folder, one file per top-level subfolder.

Usage: python scripts/make_manifest.py /Volumes/MakeMePasta/6DroneSyn data/manifest --exclude .zip
Re-running resumes: files already listed are not re-hashed; an interrupted run loses at most one subfolder.
"""
import argparse
from pathlib import Path

from dronepose.backup.manifest import build_manifest, read_manifest, write_manifest

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("out_dir", type=Path)
ap.add_argument("--exclude", nargs="*", default=[], help="file suffixes not to hash, e.g. .zip")
args = ap.parse_args()
total = 0
for sub in sorted(p for p in args.root.iterdir() if p.is_dir() and not p.name.startswith(".")):
    out = args.out_dir / f"{sub.name}.sha256"
    m = build_manifest(sub, read_manifest(out) if out.exists() else None, tuple(args.exclude))
    write_manifest(m, out)
    total += len(m)
    print(sub.name, len(m))
print(f"totale: {total} file in {args.out_dir}")
