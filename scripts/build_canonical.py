"""Raw 6DroneSyn -> data/processed/canonical_{train,val,test}.jsonl. Frames without an image are skipped and counted."""
import argparse
import os
from pathlib import Path

from dronepose.data.canonical import model_reprojection_px, to_canonical, write_jsonl
from dronepose.data.drone_models import build_drone_models, load_drone_models, save_drone_models
from dronepose.data.sixdronesyn import iter_frames

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("SIXDRONESYN_ROOT"))
ap.add_argument("--models", default="configs/drone_models.json")
ap.add_argument("--out", default="data/processed")
args = ap.parse_args()
root, models_path, out = Path(args.root), Path(args.models), Path(args.out)

if not models_path.exists():
    save_drone_models(build_drone_models(iter_frames(root)), models_path)
models = load_drone_models(models_path)

by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
missing, worst = 0, 0.0
for frame in iter_frames(root):
    if not frame.image_path.exists():
        missing += 1
        continue
    rec = to_canonical(frame, models)
    worst = max(worst, model_reprojection_px(rec, models))
    by_split[rec["split"]].append(rec)
for split, recs in by_split.items():
    recs.sort(key=lambda r: (r["scene"], r["drone"], r["background"], r["frame_idx"]))
    write_jsonl(recs, out / f"canonical_{split}.jsonl")
    print(f"{split}: {len(recs)}")
print(f"immagini mancanti: {missing}")
print(f"riproiezione del modello, peggior frame: {worst:.4f} px")
