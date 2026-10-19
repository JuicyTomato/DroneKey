"""Videos of generated samples with the baseline forecasts (plan step 3.1b): top-down and side (x-z) views.

For each chosen sample (by default one per command mode and one with a flip): the path so far (white), after the
freeze the true future (green) and the 1 s forecasts (cv, ctra, physics_acro_gen), drone axes every 5 frames
(red x nose, green y left, blue z thrust). No camera view: generated samples have no images.

Usage: python scripts/forecast/make_generated_video.py [--split val] [--index I ...] [--n 4]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.metrics import fde_at, position_errors
from dronepose.forecast.tuning import check_split, predict_all
from dronepose.forecast.video import TopDownView
from dronepose.forecast.windows import freeze_windows

COLORS = {"osservato": (255, 255, 255), "vero": (0, 200, 0), "cv": (255, 170, 0), "ctra": (0, 165, 255),
          "physics_acro_gen": (255, 0, 255)}
AXES = ((0, 0, 255), (0, 200, 0), (255, 120, 0))

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--index", type=int, nargs="*", default=None, help="sample indices (default: one per mode + a flip)")
ap.add_argument("--data", default="data/processed/forecast_gen")
ap.add_argument("--params", default="experiments/forecast/generator/tuning.json")
ap.add_argument("--config", default="configs/forecast.json")
ap.add_argument("--methods", nargs="+", default=["cv", "ctra", "physics_acro_gen"])
ap.add_argument("--out", default="experiments/forecast/generator/videos")
ap.add_argument("--size", type=int, default=600)
ap.add_argument("--allow-test", action="store_true")
args = ap.parse_args()

check_split(args.split, args.allow_test)
cfg = load_config(Path(args.config))
params = json.loads(Path(args.params).read_text())
trajs = load_trajectories(Path(args.data), args.split)
if args.index:
    chosen = list(args.index)
else:
    chosen = [next(i for i, t in enumerate(trajs) if t.meta["mode"] == m) for m in ("acro", "angle", "pmode")]
    chosen += [i for i, t in enumerate(trajs) if t.meta["flip"]][:1]
sel = [trajs[i] for i in chosen]
w = freeze_windows(sel, cfg.obs_len, cfg.pred_len, cfg.fps)
preds = predict_all(w, params, cfg)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
F = sel[0].freeze_idx
for j, (i, tr) in enumerate(zip(chosen, sel)):
    P = tr.positions.astype(float)
    R = tr.R_G.astype(float)
    last = P[F]
    fut = {k: preds[k][j] + last for k in args.methods}
    fde = {k: float(fde_at(position_errors(preds[k][j:j + 1], w.fut_p[j:j + 1]), cfg.pred_len)[0]) for k in args.methods}
    allpts = np.concatenate([P] + list(fut.values()))
    top = TopDownView(allpts, size=args.size)
    side = TopDownView(allpts[:, [0, 2]], size=args.size)
    arm = 0.04 * (args.size - 2 * top.margin) / top.scale
    m = tr.meta
    name = f"{args.split}_{i:05d}_{m['mode']}{'_flip' if m['flip'] else ''}"
    path = out / f"{name}.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, (2 * args.size, args.size))
    for t in range(len(P)):
        panels = []
        for view, idx, title in ((top, [0, 1], "vista dall'alto: x destra, y avanti"), (side, [0, 2], "vista laterale: x destra, z su")):
            pane = view.blank("m", title)
            proj = lambda X: X[:, idx]
            view.draw(pane, proj(P[:t + 1]), COLORS["osservato"], 2)
            if t >= F:
                view.draw(pane, proj(P[F:]), COLORS["vero"], 2)
                for k in args.methods:
                    view.draw(pane, proj(np.vstack([last, fut[k]])), COLORS[k], 2)
            for f in range(0, t + 1, 5):
                c = view.to_px(proj(P[f:f + 1]))[0]
                for a, col in enumerate(AXES):
                    tip = view.to_px(proj((P[f] + arm * R[f][:, a])[None]))[0]
                    cv2.line(pane, tuple(np.round(c).astype(int)), tuple(np.round(tip).astype(int)), col, 1, cv2.LINE_AA)
            y = 44
            info = [f"{m['mode']} s={m['s']:.2f} {m['style']} classe {m['drone_class']} manovra {m['maneuver']}",
                    f"frame {t} / congelamento {F}" + ("  (levette ferme)" if t > F else "")]
            if t >= F:
                info += [f"{k}: FDE 1 s {fde[k]:.2f} m" for k in args.methods]
            for line in info:
                cv2.putText(pane, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
                y += 18
            panels.append(pane)
        frame = np.hstack(panels)
        vw.write(frame)
        if t == len(P) - 1:
            cv2.imwrite(str(out / f"{name}_frame{t:02d}.jpg"), frame)
    vw.release()
    print(f"{path}: FDE 1 s " + ", ".join(f"{k} {v:.2f} m" for k, v in fde.items()))
