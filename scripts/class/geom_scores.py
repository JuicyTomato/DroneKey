"""Geometric class evidence on val: reprojection error of IPPE + LM with each of the 7 drone models, per frame.

Usage (project venv, from the repo root):
  python scripts/class/geom_scores.py --kpts KPTS.jsonl --out experiments/class/geom/geom_val_sN.jsonl
Row: {"frame_id", "reproj_px": {drone: px} | null}; null when the keypoints are missing / not finite or a PnP failed.
Prints the mean time per frame of the 7 PnP (cost of the geometric source). The test split needs --allow-test
(with --canonical data/processed/canonical_test.jsonl).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
from dronepose.data.canonical import read_jsonl  # noqa: E402
from dronepose.data.drone_models import load_drone_models  # noqa: E402
from dronepose.keypoints.class_scores import reproj_per_class  # noqa: E402
from dronepose.pose.replay import check_no_test, load_keypoints  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kpts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
    ap.add_argument("--models", default="configs/drone_models.json")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    records = read_jsonl(Path(args.canonical))
    check_no_test(records, allow_test=args.allow_test)
    models = load_drone_models(Path(args.models))
    kpts = load_keypoints(Path(args.kpts))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n_pnp = time.perf_counter(), 0
    with open(out, "w") as f:
        for r in records:
            kp = kpts.get(r["frame_id"])
            e = None if kp is None else reproj_per_class(r, kp, models)
            n_pnp += kp is not None
            f.write(json.dumps({"frame_id": r["frame_id"], "reproj_px": e}) + "\n")
    dt = time.perf_counter() - t0
    print(f"{len(records)} frames, {n_pnp} with keypoints, {1000 * dt / max(n_pnp, 1):.3f} ms/frame (7 PnP) -> {out}")


if __name__ == "__main__":
    main()
