"""Export DroneKey keypoints in the sub-project A contract: JSONL {"frame_id", "kpts_pred": [[x, y, conf] x 4]}.

Setting A: cached crops (box from crops_{split}.jsonl); works for yolo_crops (6DroneSyn) and dronekey_2dk alike.
Setting B: original PNGs of the canonical records, cropped around the boxes of C (--det, same file as YOLO26s' B);
frames without a box get NaN keypoints and confidence 0. The model config comes from the checkpoint's run_cfg;
--variant is only cross-checked (wiring/head). The test split needs --allow-test.

  python scripts/dronekey/export_dronekey.py --weights best.pt --variant paper --crops-root data/processed/yolo_crops \\
      --split val --setting A --out experiments/results/keypoints/keypoints_val_a_dronekey.jsonl
  python scripts/dronekey/export_dronekey.py --weights best.pt --variant paper --split val --setting B \\
      --det experiments/results/keypoints/keypoints_val_c_yolo26s_best.jsonl --out ...keypoints_val_b_dronekey.jsonl
"""
import argparse
import json
from pathlib import Path

from dronepose.data.canonical import read_jsonl
from dronepose.keypoints.detections import load_detections
from dronepose.keypoints.dronekey.predict import load_predictor, rows_setting_a, rows_setting_b
from dronepose.keypoints.dronekey.train import pick_device
from dronepose.pose.replay import check_no_test


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="best.pt / last.pt of train.py")
    ap.add_argument("--variant", default=None, help="name in configs/dronekey.json, checked against the checkpoint")
    ap.add_argument("--crops-root", default="data/processed/yolo_crops", help="setting A")
    ap.add_argument("--split", default="val")
    ap.add_argument("--setting", required=True, choices=["A", "B"])
    ap.add_argument("--det", default=None, help="setting B: detection JSONL of C")
    ap.add_argument("--canonical", default=None, help="setting B: default data/processed/canonical_<split>.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tiny", action="store_true", help="TinyBackbone checkpoint (tests only)")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.split == "test" and not args.allow_test:
        raise ValueError("test split is locked: pass --allow-test for the final evaluation only")
    pred = load_predictor(Path(args.weights), None, pick_device(args.device), variant=args.variant, tiny=args.tiny)
    if args.setting == "A":
        rows = rows_setting_a(pred, Path(args.crops_root), args.split, None, args.batch, args.allow_test)
    else:
        if args.det is None:
            raise ValueError("setting B needs --det")
        recs = read_jsonl(Path(args.canonical or f"data/processed/canonical_{args.split}.jsonl"))
        check_no_test(recs, args.allow_test)
        recs = [r for r in recs if r["split"] == args.split]
        rows = rows_setting_b(pred, recs, load_detections(Path(args.det)), batch=args.batch, allow_test=args.allow_test)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
