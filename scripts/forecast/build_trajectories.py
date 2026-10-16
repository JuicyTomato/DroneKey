"""Build the Phase 3 trajectory files from the canonical JSONL (no disk needed).

Usage: python scripts/forecast/build_trajectories.py [--canonical-dir data/processed] [--out data/processed/forecast]
"""
import argparse
from collections import Counter
from pathlib import Path

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.forecast.config import load_config, metric_scales
from dronepose.forecast.dataset import build_trajectories, save_trajectories

ap = argparse.ArgumentParser()
ap.add_argument("--canonical-dir", default="data/processed")
ap.add_argument("--out", default="data/processed/forecast")
ap.add_argument("--config", default="configs/forecast.json")
ap.add_argument("--models", default="configs/drone_models.json")
args = ap.parse_args()

cfg = load_config(Path(args.config))
scales = metric_scales(load_drone_models(Path(args.models)), cfg.dji_diagonal_mm)
print("scale metriche:", {d: (round(s, 4) if s else None) for d, s in scales.items()})
for split in ("train", "val", "test"):
    trajs = build_trajectories(read_jsonl(Path(args.canonical_dir) / f"canonical_{split}.jsonl"), cfg, scales)
    save_trajectories(trajs, Path(args.out), split)
    print(f"{split} {len(trajs)} traiettorie | lunghezze {dict(Counter(len(t.times) for t in trajs))} | "
          f"copie {dict(Counter(len(t.copies) for t in trajs))} | non metriche {sum(not t.metric for t in trajs)}")
