"""Verify a dataset folder against the per-subfolder manifests. Exit code 1 if anything differs.

Usage: python scripts/verify_manifest.py /Volumes/MakeMePasta/6DroneSyn data/manifest --exclude .zip [--only scene_06]
"""
import argparse
import sys
from pathlib import Path

from dronepose.backup.manifest import read_manifest, verify_manifest

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("manifest_dir", type=Path)
ap.add_argument("--exclude", nargs="*", default=[])
ap.add_argument("--only", default=None, help="verify a single subfolder (faster spot check)")
args = ap.parse_args()
bad = False
for mf in sorted(args.manifest_dir.glob("*.sha256")):
    if args.only and mf.stem != args.only:
        continue
    report = verify_manifest(args.root / mf.stem, read_manifest(mf), tuple(args.exclude))
    print(mf.stem, {k: len(v) for k, v in report.items()})
    for kind, files in report.items():
        for f in files[:10]:
            print(f"   {kind}: {f}")
    bad |= any(report.values())
sys.exit(1 if bad else 0)
