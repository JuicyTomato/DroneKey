"""Video of E4b on one VAL sequence: forecasts from the poses estimated by the full pipeline (camera view + top-down).

Each frame shows the true path observed (grey), the estimated poses observed (white dots; lost frames have none),
the true next second (green) and the 1 s forecasts with their FDE. Every method is the one of the trajectory's fold
(parameters tuned and nets trained with archive tracts of the OTHER two val scenes), as in e4_val.py. Reads the
6DroneSyn disk (images) only. Never touches the test.

Usage: python scripts/forecast/e4/make_e4_video.py --trajectory 14_02 [--background 07] [--pose-seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from dronepose.common.geometry import project
from dronepose.data.canonical import read_jsonl
from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.e4 import load_pose_windows
from dronepose.forecast.gaps import cv_g, last_valid_index
from dronepose.forecast.nets.device import pick_device
from dronepose.forecast.nets.evaluate import predict_run
from dronepose.forecast.nets.features import build_batch
from dronepose.forecast.tuning import predict_all
from dronepose.forecast.video import TopDownView, compose_frame, draw_camera_overlay, to_camera, zoom_inset

sys.path.insert(0, str(Path(__file__).parent))
from e4_train import e4_run_name  # noqa: E402

COLORS = {"gt_obs": (140, 140, 140), "vero": (0, 200, 0), "CV-2": (255, 170, 0), "kf_ca": (255, 0, 255),
          "kf_ca_los": (0, 165, 255), "R-CV-6D": (80, 80, 255)}
LABELS = {"vero": "futuro vero", "CV-2": "CV-2", "kf_ca": "Kalman CA", "kf_ca_los": "Kalman CA-LOS",
          "R-CV-6D": "R-CV-6D (scelto)"}

ap = argparse.ArgumentParser()
ap.add_argument("--trajectory", required=True, help="scene_drone of the val, e.g. 14_02")
ap.add_argument("--background", default=None, help="background copy (default: the first one)")
ap.add_argument("--pose-seed", type=int, default=0)
ap.add_argument("--cls", default="causal", choices=("causal", "gt"))
ap.add_argument("--data", default="data/processed/forecast")
ap.add_argument("--canonical-dir", default="data/processed")
ap.add_argument("--params", default="data/processed/forecast_archive/e4")
ap.add_argument("--poses", default="experiments/poses")
ap.add_argument("--runs", default="experiments/forecast/e4/runs")
ap.add_argument("--config", default="configs/forecast.json")
ap.add_argument("--out", default="experiments/forecast/videos/e4")
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--device", default="auto")
args = ap.parse_args()

cfg = load_config(Path(args.config))
tr = next((t for t in load_trajectories(Path(args.data), "val") if t.key == args.trajectory), None)
if tr is None or not tr.metric:
    raise SystemExit(f"{args.trajectory}: not a metric trajectory of the val")
bg, degree = next(c for c in tr.copies if args.background in (None, c[0]))
w_all, info = load_pose_windows(Path(args.poses) / f"poses_val_s{args.pose_seed}_{args.cls}.jsonl", [tr], cfg)
w = w_all.subset(w_all.meta["copy"] == f"{bg}_{int(degree):03d}")
params = json.loads((Path(args.params) / f"params_fold{tr.scene}.json").read_text())

preds = {"CV-2": cv_g(w, 2)}
allp = predict_all(w, params, cfg, los=True)
preds.update(kf_ca=allp["kf_ca"], kf_ca_los=allp["kf_ca_los"])
batch = build_batch(w, {**params, "cv_k": 2}, cfg, mask_channel=True)
lv = last_valid_index(w.obs_valid)
anchor = w.obs_p[np.arange(len(w)), lv][:, None]                 # the nets predict relative to the last valid pose
device = pick_device(args.device)
runs = [Path(args.runs) / e4_run_name("R-CV-6D", f"fold{tr.scene}", s) for s in (0, 1, 2)]
preds["R-CV-6D"] = np.mean([predict_run(r, batch, device) for r in runs], 0) + anchor
methods = list(preds)

records = {r["frame_idx"]: r for r in read_jsonl(Path(args.canonical_dir) / "canonical_val.jsonl")
           if (r["scene"], r["drone"], r["background"]) == (tr.scene, tr.drone, bg)}
K = np.array(next(iter(records.values()))["K"])
N = cfg.obs_len
view = TopDownView(tr.positions, size=args.height)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
name = f"{tr.key}_{bg}_{int(degree):03d}_e4b_s{args.pose_seed}_{args.cls}"
gt_obs = tr.positions[w.start[:, None] + np.arange(N)] - w.last_p[:, None]
writer, still_at = None, len(w) // 2
fde_sum = {k: 0.0 for k in methods}
for i in range(len(w)):
    frame_idx = int(w.start[i]) + N
    img = cv2.imread(str(records[frame_idx]["image_path"]))
    if img is None:
        raise SystemExit(f"cannot read {records[frame_idx]['image_path']} (disk connected?)")
    fde = {k: float(np.linalg.norm(preds[k][i, -1] - w.fut_p[i, -1])) for k in methods}
    for k in methods:
        fde_sum[k] += fde[k]
    est = w.obs_p[i][w.obs_valid[i]]
    start = w.obs_p[i, lv[i]]
    paths = {"gt_obs": gt_obs[i], "vero": np.vstack([[0, 0, 0], w.fut_p[i]])}
    paths.update({k: np.vstack([start, preds[k][i]]) for k in methods})
    cam_paths = {k: to_camera(p, w.last_p[i], tr.s_c) for k, p in paths.items()}
    cam = draw_camera_overlay(img, K, {"gt_obs": cam_paths["gt_obs"]}, COLORS, thickness=3)
    cam = draw_camera_overlay(cam, K, {"vero": cam_paths["vero"]}, COLORS, thickness=9)
    cam = draw_camera_overlay(cam, K, {k: cam_paths[k] for k in methods}, COLORS, thickness=3)
    for px in project(K, to_camera(est, w.last_p[i], tr.s_c)):
        cv2.circle(cam, tuple(np.round(px).astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
    centre = project(K, cam_paths["gt_obs"][-1:])[0]
    extent = np.ptp(project(K, np.vstack([cam_paths["gt_obs"], cam_paths["vero"]])), axis=0).max()
    crop = int(np.clip(2.5 * extent, 160, 600))
    inset = 540
    cam[-inset - 20:-20, 20:20 + inset] = zoom_inset(cam, centre, crop, inset)
    cv2.rectangle(cam, (20, cam.shape[0] - inset - 20), (20 + inset, cam.shape[0] - 20), (255, 255, 255), 2)
    cv2.putText(cam, f"zoom x{inset / crop:.1f}", (30, cam.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    panel = view.blank("m")
    view.draw(panel, tr.positions, (80, 80, 80), 1)
    view.draw(panel, paths["vero"] + w.last_p[i], COLORS["vero"], 6)
    for k in ["gt_obs"] + methods:
        view.draw(panel, paths[k] + w.last_p[i], COLORS[k], 2)
    for px in view.to_px(est + w.last_p[i]):
        cv2.circle(panel, tuple(np.round(px).astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    frame = compose_frame(cam, panel, args.height)
    cls = "classe predetta" if args.cls == "causal" else "classe vera"
    lines = [(f"{tr.key} sfondo {bg} (val, fold {tr.scene})  frame {frame_idx}/{len(tr.times)}  orizzonte 1 s",
              (255, 255, 255)),
             (f"pose stimate ({cls}, seme {args.pose_seed}): punti bianchi; percorso vero: grigio", (255, 255, 255)),
             (LABELS["vero"], COLORS["vero"])]
    lines += [(f"{LABELS[k]}: FDE 1 s {fde[k]:.3f} m", COLORS[k]) for k in methods]
    box = frame[0:20 + 28 * len(lines), 0:820]
    box[:] = (0.35 * box).astype(np.uint8)
    for j, (text, col) in enumerate(lines):
        cv2.putText(frame, text, (12, 30 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    if writer is None:
        writer = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps,
                                 (frame.shape[1], frame.shape[0]))
    writer.write(frame)
    if i == still_at:
        cv2.imwrite(str(out / f"{name}_frame{frame_idx:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
writer.release()
print(f"{out / name}.mp4: {len(w)} fotogrammi; FDE media a 1 s (m):",
      {LABELS[k]: round(v / len(w), 3) for k, v in fde_sum.items()},
      f"frame persi {100 * (1 - w.obs_valid.mean()):.1f}%")
