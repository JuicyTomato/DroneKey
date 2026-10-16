"""Video of the Phase 3 baseline forecasts on one sequence: camera view + top-down panel.

Each frame from the 15th on shows the last 0.5 s observed (white), the true next second (green) and the 1 s
forecasts of the chosen baselines, with their FDE at 1 s. Needs the 6DroneSyn disk (image paths of the records).

Usage: python scripts/forecast/make_forecast_video.py --trajectory 14_02 [--split val] [--background 07]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.tuning import check_split, predict_all
from dronepose.common.geometry import project
from dronepose.forecast.video import TopDownView, compose_frame, draw_camera_overlay, to_camera, zoom_inset
from dronepose.forecast.windows import make_windows

COLORS = {"osservato": (255, 255, 255), "vero": (0, 200, 0), "cv": (255, 170, 0), "ca": (0, 165, 255),
          "kf_ca": (255, 0, 255), "physics": (0, 0, 255)}
LABELS = {"vero": "futuro vero", "cv": "CV-{cv_k}", "ca": "CA-{ca_k}", "kf_ca": "Kalman CA", "physics": "fisica"}

ap = argparse.ArgumentParser()
ap.add_argument("--trajectory", required=True, help="scene_drone, e.g. 14_02")
ap.add_argument("--split", default="val")
ap.add_argument("--background", default=None, help="background copy for the images (default: first copy)")
ap.add_argument("--methods", nargs="+", default=["cv", "ca", "kf_ca", "physics"])
ap.add_argument("--trajectories", default="data/processed/forecast")
ap.add_argument("--canonical-dir", default="data/processed")
ap.add_argument("--params", default="experiments/forecast/tuning/tuning.json")
ap.add_argument("--config", default="configs/forecast.json")
ap.add_argument("--out", default="experiments/forecast/videos")
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--allow-test", action="store_true")
args = ap.parse_args()

check_split(args.split, args.allow_test)
cfg = load_config(Path(args.config))
params = json.loads(Path(args.params).read_text())
tr = next((t for t in load_trajectories(Path(args.trajectories), args.split) if t.key == args.trajectory), None)
if tr is None:
    raise SystemExit(f"trajectory {args.trajectory} not in split {args.split}")
bg, degree = next(c for c in tr.copies if args.background in (None, c[0]))
scene, drone = tr.scene, tr.drone
records = {r["frame_idx"]: r for r in read_jsonl(Path(args.canonical_dir) / f"canonical_{args.split}.jsonl")
           if (r["scene"], r["drone"], r["background"]) == (scene, drone, bg)}
K = np.array(next(iter(records.values()))["K"])

N, H = cfg.obs_len, cfg.pred_len
w = make_windows([tr], N, H, cfg.fps)
preds = predict_all(w, params, cfg)
units = "m" if tr.metric else "unità dataset"
labels = {k: v.format(**params) for k, v in LABELS.items()}

view = TopDownView(tr.positions, size=args.height)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
name = f"{tr.key}_{bg}_{degree:03d}"
writer, still_at = None, len(w) // 2
fde_sum = {k: 0.0 for k in args.methods}
for i in range(len(w)):
    frame_idx = int(w.start[i]) + N                           # 1-based index of the last observed frame
    img = cv2.imread(str(records[frame_idx]["image_path"]))
    if img is None:
        raise SystemExit(f"cannot read {records[frame_idx]['image_path']} (disk connected?)")
    rel = {"osservato": w.obs_p[i], "vero": np.vstack([[0, 0, 0], w.fut_p[i]])}
    rel.update({k: np.vstack([[0, 0, 0], preds[k][i]]) for k in args.methods})
    fde = {k: float(np.linalg.norm(preds[k][i, -1] - w.fut_p[i, -1])) for k in args.methods}
    for k in args.methods:
        fde_sum[k] += fde[k]

    # true future first and thicker: where a forecast agrees it keeps a green halo, where it diverges it shows alone
    cam_paths = {k: to_camera(p, w.last_p[i], tr.s_c) for k, p in rel.items()}
    cam = draw_camera_overlay(img, K, {"vero": cam_paths["vero"]}, COLORS, thickness=9)
    cam = draw_camera_overlay(cam, K, {k: v for k, v in cam_paths.items() if k != "vero"}, COLORS, thickness=3)
    centre = project(K, cam_paths["osservato"][-1:])[0]
    extent = np.ptp(project(K, np.vstack([cam_paths["osservato"], cam_paths["vero"]])), axis=0).max()
    crop = int(np.clip(2.5 * extent, 160, 600))
    inset_size = 540
    cam[-inset_size - 20:-20, 20:20 + inset_size] = zoom_inset(cam, centre, crop, inset_size)
    cv2.rectangle(cam, (20, cam.shape[0] - inset_size - 20), (20 + inset_size, cam.shape[0] - 20), (255, 255, 255), 2)
    cv2.putText(cam, f"zoom x{inset_size / crop:.1f}", (30, cam.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    panel = view.blank("m" if tr.metric else "u")
    view.draw(panel, tr.positions, (80, 80, 80), 1)
    view.draw(panel, rel["vero"] + w.last_p[i], COLORS["vero"], 6)
    for k, p in rel.items():
        if k != "vero":
            view.draw(panel, p + w.last_p[i], COLORS[k], 2)
    cv2.circle(panel, tuple(np.round(view.to_px(w.last_p[i][None])[0]).astype(int)), 5, (255, 255, 255), -1)
    frame = compose_frame(cam, panel, args.height)

    lines = [(f"{tr.key} sfondo {bg} ({args.split})  frame {frame_idx}/{len(tr.times)}  orizzonte 1 s", (255, 255, 255)),
             ("osservato (0.5 s)", COLORS["osservato"]), (labels["vero"], COLORS["vero"])]
    lines += [(f"{labels[k]}: FDE 1 s {fde[k]:.3f} {units}", COLORS[k]) for k in args.methods]
    box = frame[0:20 + 28 * len(lines), 0:640]
    box[:] = (0.35 * box).astype(np.uint8)                      # darkened legend background
    for j, (text, col) in enumerate(lines):
        cv2.putText(frame, text, (12, 30 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    if writer is None:
        writer = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps,
                                 (frame.shape[1], frame.shape[0]))
    writer.write(frame)
    if i == still_at:
        cv2.imwrite(str(out / f"{name}_frame{frame_idx:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
writer.release()
print(f"{out / name}.mp4: {len(w)} fotogrammi; FDE media a 1 s ({units}):",
      {labels[k]: round(v / len(w), 3) for k, v in fde_sum.items()})
