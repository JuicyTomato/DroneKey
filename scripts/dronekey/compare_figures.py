"""Figures for E1 on val: YOLO26s against DroneKey on single instances (setting B, GT class, `global`).

  python scripts/dronekey/compare_figures.py --yolo Y.jsonl --dk D.jsonl --dk-flip F.jsonl --out-dir DIR

Writes:
  flip_14_05_16_000.png  rotation error along the flipped sequence (YOLO, DroneKey, DroneKey with gravity) and three
                         crops of one frame with the chosen pose: the upside-down IPPE branch reprojects almost on the
                         same keypoints, but its "up" axis points down;
  gallery.png            8 crops: typical frames, frames where DroneKey wins, where YOLO wins, drone 08.
Drawn on each crop: GT keypoints (white rings), predicted keypoints (dots), rigid 3D model reprojected with the chosen
pose (thin outline) and its "up" axis (arrow). Original PNGs are read only. Validation only.
"""
import argparse
import importlib.util
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dronepose.common.geometry import project
from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.pose.replay import check_no_test, load_keypoints, replay, sequence_key

_spec = importlib.util.spec_from_file_location("eval_e1", Path(__file__).resolve().parents[1] / "eval_e1.py")
_eval_e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_e1)

SURFACE, BLUE, ORANGE, GREEN, YELLOW = "#fcfcfb", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FLIP_SEQ = "14_05_16_000"
CROP = 360


def bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def by_frame(recs, kpts, models, up=None) -> dict:
    return {r["frame_id"]: r for rs in replay(recs, kpts, models, "global", up=up).values() for r in rs}


def crop_box(rec) -> tuple:
    x0, y0, x1, y1 = rec["bbox_xyxy"]
    c, half = np.array([(x0 + x1) / 2, (y0 + y1) / 2]), 0.65 * max(x1 - x0, y1 - y0)
    return c[0] - half, c[1] - half, 2 * half


def draw_crop(rec, models, preds: list[tuple]) -> np.ndarray:
    """preds: (kpts (4,3), result dict or None, hex colour). Returns an RGB crop of CROP x CROP px."""
    img = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(rec["image_path"])
    x0, y0, size = crop_box(rec)
    s = CROP / size
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    out = cv2.warpAffine(img, M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=(252, 252, 251))
    to_crop = lambda p: tuple(int(round(v)) for v in ((np.asarray(p, float) - [x0, y0]) * s))
    K, model = np.array(rec["K"]), models[rec["drone"]]
    span = float(np.linalg.norm(model.kpts_flu, axis=1).max())
    for kp, res, col in preds:
        c = bgr(col)
        if res is not None and res.get("R") is not None:
            R, t = np.array(res["R"]), np.array(res["t"])
            pts = project(K, (R @ model.kpts_flu.T).T + t)
            cv2.polylines(out, [np.array([to_crop(p) for p in pts], np.int32).reshape(-1, 1, 2)], True, c, 1,
                          cv2.LINE_AA)
            a = project(K, np.vstack([t, t + 0.8 * span * R[:, 2]]))
            cv2.arrowedLine(out, to_crop(a[0]), to_crop(a[1]), c, 2, cv2.LINE_AA, tipLength=0.25)
        for p in np.asarray(kp, float)[:, :2]:
            if np.isfinite(p).all():
                cv2.circle(out, to_crop(p), 4, c, -1, cv2.LINE_AA)
    for p in np.asarray(rec["kpts2d"], float):
        cv2.circle(out, to_crop(p), 8, (255, 255, 255), 2, cv2.LINE_AA)
    return out[:, :, ::-1]


def err(res) -> float:
    return np.nan if res is None or res.get("rot_err_sym_deg") is None else float(res["rot_err_sym_deg"])


def pick_gallery(recs, ry, rd, rng) -> list[tuple[str, str]]:
    """(frame_id, label): 2 typical, 2 DroneKey better, 2 YOLO better, 2 drone 08; one frame per drone/scene."""
    rows = []
    for r in recs:
        ey, ed = err(ry.get(r["frame_id"])), err(rd.get(r["frame_id"]))
        if np.isfinite(ey) and np.isfinite(ed) and ey < 90 and ed < 90 and sequence_key(r["frame_id"]) != FLIP_SEQ:
            rows.append((r, ey, ed))
    no8 = [x for x in rows if x[0]["drone"] != "08"]
    med_y, med_d = np.median([x[1] for x in no8]), np.median([x[2] for x in no8])
    used, out = set(), []

    def take(cands, n, label):
        for r, ey, ed in cands:
            key = (r["drone"], r["scene"])
            if key not in used:
                used.add(key)
                out.append((r["frame_id"], label))
                n -= 1
                if n == 0:
                    return

    typical = sorted(no8, key=lambda x: abs(x[1] - med_y) + abs(x[2] - med_d))[:400]
    take([typical[i] for i in rng.permutation(len(typical))], 2, "tipico")
    take(sorted(no8, key=lambda x: x[2] - x[1])[:600][::10], 2, "meglio DroneKey")
    take(sorted(no8, key=lambda x: x[1] - x[2])[:600][::10], 2, "meglio YOLO")
    d8 = sorted([x for x in rows if x[0]["drone"] == "08"], key=lambda x: x[1] - x[2])
    take(d8[: max(1, len(d8) // 4)][::7], 2, "drone 08")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo", required=True, help="YOLO26s keypoints (setting B)")
    ap.add_argument("--dk", required=True, help="DroneKey keypoints for the gallery (setting B)")
    ap.add_argument("--dk-flip", required=True, help="DroneKey keypoints of a checkpoint that flips 14_05_16_000")
    ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
    ap.add_argument("--models", default="configs/drone_models.json")
    ap.add_argument("--frame", type=int, default=180, help="frame_idx of the flipped sequence to show")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#9a9a96", "axes.labelcolor": "#33332f",
                         "xtick.color": "#5b5b57", "ytick.color": "#5b5b57"})

    recs = [r for r in read_jsonl(Path(args.canonical)) if r["split"] == "val"]
    check_no_test(recs, False)
    models = load_drone_models(Path(args.models))
    ky, kd, kf = (load_keypoints(Path(p)) for p in (args.yolo, args.dk, args.dk_flip))

    # --- flipped sequence -------------------------------------------------------------------------------------------
    seq = sorted([r for r in recs if sequence_key(r["frame_id"]) == FLIP_SEQ], key=lambda r: r["frame_idx"])
    fy, ff = by_frame(seq, ky, models), by_frame(seq, kf, models)
    fg = by_frame(seq, kf, models, up=_eval_e1.GRAVITY_UP)
    idx = [r["frame_idx"] for r in seq]
    fig = plt.figure(figsize=(11, 6.6), facecolor=SURFACE)
    ax = fig.add_axes([0.07, 0.6, 0.9, 0.3], facecolor=SURFACE)
    for res, col, lab in ((fy, BLUE, "YOLO26s"), (ff, ORANGE, "DroneKey"), (fg, GREEN, "DroneKey + gravità")):
        ax.plot(idx, [err(res[r["frame_id"]]) for r in seq], color=col, lw=1.4, label=lab)
    ax.set_yscale("log")
    ax.set_ylabel("errore di rotazione (°)")
    ax.set_xlabel("frame")
    ax.axvline(args.frame, color="#9a9a96", lw=0.8, ls="--")
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    rec = next(r for r in seq if r["frame_idx"] == args.frame)
    fid = rec["frame_id"]
    panels = ((ky, fy, BLUE, "YOLO26s"), (kf, ff, ORANGE, "DroneKey"), (kf, fg, GREEN, "DroneKey + gravità"))
    for i, (k, res, col, lab) in enumerate(panels):
        a = fig.add_axes([0.07 + i * 0.31, 0.03, 0.28, 0.47])
        a.imshow(draw_crop(rec, models, [(k[fid], res[fid], col)]))
        a.set_title(f"{lab}: {err(res[fid]):.1f}°", color=col, fontsize=10)
        a.axis("off")
    fig.suptitle(f"Sequenza {FLIP_SEQ} (scena 14, drone 05): la soluzione IPPE capovolta", x=0.07, y=0.985, ha="left",
                 fontsize=11, color="#33332f")
    fig.savefig(out_dir / f"flip_{FLIP_SEQ}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # --- gallery ----------------------------------------------------------------------------------------------------
    ry, rd = by_frame(recs, ky, models), by_frame(recs, kd, models)
    chosen = pick_gallery(recs, ry, rd, np.random.default_rng(args.seed))
    by_id = {r["frame_id"]: r for r in recs}
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.4), facecolor=SURFACE)
    for a, (fid, label) in zip(axes.ravel(), chosen):
        r = by_id[fid]
        a.imshow(draw_crop(r, models, [(ky[fid], ry[fid], BLUE), (kd[fid], rd[fid], ORANGE)]))
        a.set_title(f"{label} · drone {r['drone']}, scena {r['scene']}\n"
                    f"YOLO {err(ry[fid]):.1f}°   DroneKey {err(rd[fid]):.1f}°", fontsize=9, color="#33332f")
        a.axis("off")
    for a in axes.ravel()[len(chosen):]:
        a.axis("off")
    fig.suptitle("Setting B, val: cerchi bianchi = keypoint veri; blu = YOLO26s, arancio = DroneKey "
                 "(punti = keypoint, contorno = modello 3D riproiettato, freccia = asse \"su\" della posa)",
                 fontsize=9.5, color="#33332f")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.04, hspace=0.25)
    fig.savefig(out_dir / "gallery.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    for fid, label in chosen:
        print(f"{label:16s} {fid}  YOLO {err(ry[fid]):6.2f}  DK {err(rd[fid]):6.2f}")


if __name__ == "__main__":
    main()
