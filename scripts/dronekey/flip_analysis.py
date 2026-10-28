"""Where do the `global` flips come from? Per-sequence flip rate (drones != 08) and, on the flipped sequences, the
reprojection error of the correct and of the wrong IPPE branch, with and without the gravity gate.

  python scripts/dronekey/flip_analysis.py --out experiments/dronekey/flip_analysis_val_b.json \\
      yolo_s0=experiments/results/keypoints/keypoints_val_b_cbest_yolo26s.jsonl \\
      dk_s2=experiments/dronekey/keypoints/keypoints_val_b_dk_paper_s2_last.jsonl

A sequence is reported when more than --min-flip of its posed frames are flipped (rotation error > 90 deg) in any
model; "oracle" = flip rate of the better of the two candidates (0 means the keypoints allow the right pose).
Validation only: the test split is refused.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from dronepose.common.geometry import geodesic_deg
from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.pipeline.skeleton import ippe_candidates
from dronepose.pose.replay import check_no_test, load_keypoints, replay

_spec = importlib.util.spec_from_file_location("eval_e1", Path(__file__).resolve().parents[1] / "eval_e1.py")
_eval_e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_e1)
GRAVITY_UP = _eval_e1.GRAVITY_UP


def flip_stats(results: list[dict]) -> dict:
    posed = [r for r in results if r["rot_err_deg"] is not None]
    oracle = [r["oracle_rot_err_deg"] for r in results if r["oracle_rot_err_deg"] is not None]
    return {"n": len(results), "coverage": len(posed) / len(results) if results else 0.0,
            "flip": float(np.mean([r["rot_err_deg"] > 90 for r in posed])) if posed else 0.0,
            "oracle_flip": float(np.mean([e > 90 for e in oracle])) if oracle else 0.0}


def branch_margin(recs: list[dict], kpts: dict, models) -> dict:
    """Mean reprojection error (px) of the IPPE branch closest to the GT rotation and of the other one."""
    good, bad = [], []
    for r in recs:
        k = kpts.get(r["frame_id"])
        if k is None or not np.isfinite(k).all():
            continue
        sols = ippe_candidates(r, k, models)
        if len(sols) < 2:
            continue
        i = int(np.argmin([geodesic_deg(s.R, np.array(r["R_gt"])) for s in sols]))
        good.append(sols[i].reproj_px)
        bad.append(sols[1 - i].reproj_px)
    good, bad = np.array(good), np.array(bad)
    return {"frames": len(good), "reproj_correct": float(good.mean()) if len(good) else None,
            "reproj_wrong": float(bad.mean()) if len(bad) else None,
            "wrong_fits_better": float(np.mean(bad < good)) if len(good) else None}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", metavar="NAME=KPTS.jsonl")
    ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
    ap.add_argument("--models", default="configs/drone_models.json")
    ap.add_argument("--min-flip", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    specs = dict(s.split("=", 1) for s in args.specs)
    recs = [r for r in read_jsonl(Path(args.canonical)) if r["split"] == "val"]
    check_no_test(recs, False)
    models = load_drone_models(Path(args.models))
    kpts = {n: load_keypoints(Path(p)) for n, p in specs.items()}

    per_seq = {n: {} for n in specs}
    for name in specs:
        for up_name, up in (("off", None), ("gravity", GRAVITY_UP)):
            for seq, rs in replay(recs, kpts[name], models, "global", up=up).items():
                if seq.split("_")[1] != "08":
                    per_seq[name].setdefault(seq, {})[up_name] = flip_stats(rs)
    flipped = sorted({s for n in specs for s, v in per_seq[n].items() if v["off"]["flip"] > args.min_flip})

    out = {"models": specs, "flipped_sequences": {}}
    for seq in flipped:
        seq_recs = sorted([r for r in recs if r["frame_id"].rsplit("_", 1)[0] == seq], key=lambda r: r["frame_idx"])
        out["flipped_sequences"][seq] = {n: {**per_seq[n][seq], **branch_margin(seq_recs, kpts[n], models)}
                                         for n in specs}
        print(f"\n{seq}")
        for n, v in out["flipped_sequences"][seq].items():
            print(f"  {n:12s} flip {v['off']['flip']:4.0%} (oracle {v['off']['oracle_flip']:4.0%}, "
                  f"with gravity {v['gravity']['flip']:4.0%}) | reproj correct {v['reproj_correct']:5.2f} "
                  f"wrong {v['reproj_wrong']:5.2f} px, wrong fits better in {v['wrong_fits_better']:4.0%}")
    out["per_sequence"] = per_seq
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
