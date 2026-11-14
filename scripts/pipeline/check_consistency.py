"""Compare a live perception cache with the exported files of the same weights (spec 2026-11-14 §6).

Usage: python scripts/pipeline/check_consistency.py --live experiments/pipeline/perception_val_sN.jsonl
           [--config configs/pipeline_sN.json] [--out experiments/pipeline/consistency_val_sN.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dronepose.keypoints.class_scores import load_probs  # noqa: E402
from dronepose.keypoints.detections import load_detections  # noqa: E402
from dronepose.pipeline.consistency import THRESHOLDS, compare_boxes, compare_kpts, compare_probs, verdicts  # noqa: E402
from dronepose.pipeline.perception import Perception  # noqa: E402
from dronepose.pose.replay import load_keypoints  # noqa: E402

REF = {"boxes": "experiments/results/keypoints/keypoints_val_c_yolo26s_best.jsonl",
       "kpts": "experiments/dronekey/keypoints/keypoints_val_b_dk_paper_s0_last.jsonl",
       "probs": "experiments/class/k/probs_val_s0.jsonl"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None, help="takes ref_keypoints / ref_probs of this seed (default: seed 0 files)")
    args = ap.parse_args(argv)
    ref = dict(REF)
    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        ref.update(kpts=cfg["ref_keypoints"], probs=cfg["ref_probs"])
    with open(args.live) as f:
        live = {p.frame_id: p for p in (Perception.from_row(json.loads(l)) for l in f if l.strip())}
    res = {"n_live": len(live), "thresholds": THRESHOLDS,
           "boxes": compare_boxes(live, load_detections(Path(ref["boxes"]))),
           "keypoints": compare_kpts(live, load_keypoints(Path(ref["kpts"]))),
           "probs": compare_probs(live, load_probs(Path(ref["probs"])))}
    res["verdicts"] = verdicts(res["boxes"], res["keypoints"], res["probs"])
    for k in ("boxes", "keypoints", "probs"):
        print(f"{k}: {'PASS' if res['verdicts'][k] else 'FAIL'} {res[k]}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
