"""Run the skeleton pipeline on one sequence and write per-frame results, a summary and an overlay video."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.keypoints.predictors import GTPredictor, YoloPosePredictor
from dronepose.pose.disambiguation import TemporalParams
from dronepose.pose.global_disambiguation import GlobalParams
from dronepose.pipeline.skeleton import run_sequence, summarize
from dronepose.viz.overlay import draw_overlay

ap = argparse.ArgumentParser()
ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
ap.add_argument("--models", default="configs/drone_models.json")
ap.add_argument("--sequence", required=True, help="frame_id prefix, e.g. 14_03_05_")
ap.add_argument("--frames-dir", default=None, help="folder with <frame_id>.jpg; default: record image_path")
ap.add_argument("--predictor", default="gt", help="gt | gt+noise:<sigma_px> | yolo:<weights.pt>")
ap.add_argument("--obs", type=int, default=15)
ap.add_argument("--horizon", type=int, default=30)
ap.add_argument("--strategy", default="reproj", choices=["reproj", "temporal", "temporal_reject", "global"])
ap.add_argument("--reanchor-ratio", type=float, default=3.0)
ap.add_argument("--reanchor-frames", type=int, default=3)
ap.add_argument("--max-gap", type=int, default=5)
ap.add_argument("--max-jump-deg", type=float, default=20.0)
ap.add_argument("--max-reproj-px", type=float, default=20.0)
ap.add_argument("--reject-cost", type=float, default=None, help="global strategy: cost of a frame without pose")
ap.add_argument("--z-cap", type=float, default=None, help="global strategy: normalised jump that restarts the chain")
ap.add_argument("--up", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                help="global strategy: camera-frame vertical (gravity gate); default: gravity off")
ap.add_argument("--out", default="experiments/skeleton")
args = ap.parse_args()

records = [r for r in read_jsonl(Path(args.canonical)) if r["frame_id"].startswith(args.sequence)]
assert records, f"nessun frame con prefisso {args.sequence}"
models = load_drone_models(Path(args.models))

if args.predictor == "gt":
    predictor = GTPredictor()
elif args.predictor.startswith("gt+noise:"):
    predictor = GTPredictor(sigma_px=float(args.predictor.split(":", 1)[1]))
elif args.predictor.startswith("yolo:"):
    predictor = YoloPosePredictor(args.predictor.split(":", 1)[1])
else:
    raise SystemExit(f"predictor sconosciuto: {args.predictor}")


def load_image(r):
    path = Path(args.frames_dir) / f"{r['frame_id']}.jpg" if args.frames_dir else Path(r["image_path"])
    img = cv2.imread(str(path))
    assert img is not None, f"immagine non trovata: {path}"
    return img


if args.strategy == "global":
    base = GlobalParams()
    params = GlobalParams(max_reproj_px=args.max_reproj_px,
                          reject_cost=base.reject_cost if args.reject_cost is None else args.reject_cost,
                          z_cap=base.z_cap if args.z_cap is None else args.z_cap, max_span=base.max_span)
else:
    params = TemporalParams(args.reanchor_ratio, args.reanchor_frames, args.max_gap, args.max_jump_deg,
                            args.max_reproj_px)
results = run_sequence(records, load_image, predictor, models, obs=args.obs, horizon=args.horizon,
                       strategy=args.strategy, params=params, up=args.up)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
summary = summarize(results)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

by_id = {r["frame_id"]: r for r in records}
writer = None
for res in results:
    rec = by_id[res["frame_id"]]
    frame = draw_overlay(load_image(rec), np.array(rec["K"]), np.array(rec["kpts2d"]), np.array(res["kpts_pred"]),
                         res["R"], res["t"], res["future_pred"])
    frame = cv2.resize(frame, (960, 540))
    if writer is None:
        writer = cv2.VideoWriter(str(out / "skeleton.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 540))
    writer.write(frame)
writer.release()
print("video:", out / "skeleton.mp4")
