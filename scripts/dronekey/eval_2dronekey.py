"""Evaluate DroneKey on 2DroneKey: AP (paper definition), SR90, SR95 under two OKS readings, plus pixel error.

Reading (i):  OKS = mean_k exp(-d_k^2 / (0.2 * area)), d in frame pixels, area of `label_box_xyxy(gt)` (= `oks`).
Reading (ii): OKS = mean_k exp(-(d_k / side)^2 / (0.2 * 0.2^2)), side of the crop in crops_{split}.jsonl.
AP = mean over OKS thresholds 0.50:0.05:0.95 of the fraction of frames with OKS >= threshold (no scores).
Every GT frame is evaluated: a missing or non-finite prediction is a miss (OKS 0, no pixel error).
The split is inferred from the --gt name (gt_test.jsonl = test) and the test split needs --allow-test.

  python scripts/dronekey/eval_2dronekey.py --pred PRED.jsonl --gt data/processed/dronekey_2dk/gt_val.jsonl \\
      --crops data/processed/dronekey_2dk/crops_val.jsonl --out OUT.json [--allow-test]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.data.crops import label_box_xyxy
from dronepose.eval.keypoint_metrics import _dists, ap_paper, frame_px_error, oks, success_rate
from dronepose.pose.replay import load_keypoints

PAPER = {"ap_paper": 99.68, "sr90": 99.8, "sr95": 98.5}     # DroneKey (IROS 2025), percent


def split_of(gt_path: Path) -> str:
    m = re.fullmatch(r"gt_(\w+)\.jsonl", Path(gt_path).name)
    if not m:
        raise ValueError(f"cannot infer the split from {Path(gt_path).name!r}: expected gt_<split>.jsonl")
    return m.group(1)


def _summary(oks_v: list[float]) -> dict:
    return {"ap_paper": ap_paper(oks_v), "sr90": success_rate(oks_v, 0.90), "sr95": success_rate(oks_v, 0.95),
            "mean_oks": float(np.mean(oks_v)) if oks_v else 0.0}


def run(pred: Path, gt: Path, crops: Path, allow_test: bool = False) -> dict:
    """Metrics of `pred` against `gt` (2DroneKey frame pixels) with the crop sides of `crops`."""
    split = split_of(gt)
    if split == "test" and not allow_test:
        raise ValueError("gt is the test split: it is used once, at the end (pass allow_test to override)")
    gts = {r["frame_id"]: np.array(r["kpts2d"], float) for r in read_jsonl(Path(gt))}
    sides = {r["frame_id"]: float(r["box"][2]) for r in read_jsonl(Path(crops))}
    absent = [f for f in gts if f not in sides]
    if absent:
        raise ValueError(f"{len(absent)} GT frames have no crop in {crops} (first: {absent[0]})")
    preds = load_keypoints(Path(pred))
    oks_i, oks_ii, px, n_missing = [], [], [], 0
    for fid, g in gts.items():
        d = _dists(preds[fid], g) if fid in preds else None
        if d is None:
            n_missing += 1
            oks_i.append(0.0)
            oks_ii.append(0.0)
            continue
        x0, y0, x1, y1 = label_box_xyxy(g)
        oks_i.append(oks(preds[fid], g, (x1 - x0) * (y1 - y0)))
        oks_ii.append(float(np.mean(np.exp(-((d / sides[fid]) ** 2) / (0.2 * 0.2 ** 2)))))
        px.append(frame_px_error(preds[fid], g))
    return {"split": split, "n_frames": len(gts), "n_missing": n_missing,
            "extra_predictions": len(set(preds) - set(gts)),
            "reading_i": _summary(oks_i), "reading_ii": _summary(oks_ii),
            "px_median": float(np.median(px)) if px else None,
            "px_p90": float(np.percentile(px, 90)) if px else None,
            "paper": PAPER}


def print_table(out: dict) -> None:
    print(f"2DroneKey {out['split']}: {out['n_frames']} frames, {out['n_missing']} without a usable prediction")
    print(f"{'':28s}{'AP':>8s}{'SR90':>8s}{'SR95':>8s}")
    for name, key in (("reading (i)  frame px, box area", "reading_i"), ("reading (ii) crop-normalised", "reading_ii")):
        r = out[key]
        print(f"{name:28s}{100 * r['ap_paper']:8.2f}{100 * r['sr90']:8.2f}{100 * r['sr95']:8.2f}")
    p = out["paper"]
    print(f"{'paper (IROS 2025)':28s}{p['ap_paper']:8.2f}{p['sr90']:8.2f}{p['sr95']:8.2f}")
    fmt = lambda v: "-" if v is None else f"{v:.2f}"      # noqa: E731
    print(f"px error (median of 4 propellers per frame): median {fmt(out['px_median'])}  p90 {fmt(out['px_p90'])}")
    print("note: our input (crop from the GT box) is not necessarily the authors' one")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True, help="gt_{split}.jsonl of data/processed/dronekey_2dk")
    ap.add_argument("--crops", required=True, help="crops_{split}.jsonl of the same folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    out = run(Path(args.pred), Path(args.gt), Path(args.crops), args.allow_test)
    print_table(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
