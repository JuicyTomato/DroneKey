"""Side-by-side video of the three IPPE strategies on one sequence, using saved YOLO keypoints (no GPU).

Each panel is a crop around the drone: green rings = ground-truth propellers, red dots = predicted (numbered 1-4),
arrows = chosen pose (x red = front, y green = left, z blue = up). Badges: FLIP (rotation error > 90 deg),
REJECTED (frame discarded by strategy c). Bottom strip: rotation error per frame for the three strategies.

Usage: python scripts/compare_strategies.py --sequence 09_03_01_000 [--keypoints ...] [--out experiments/compare]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from dronepose.common.geometry import project
from dronepose.data.canonical import read_jsonl
from dronepose.data.crops import CropBox, crop_image, frame_to_crop
from dronepose.data.drone_models import load_drone_models
from dronepose.pose.disambiguation import STRATEGIES
from dronepose.pose.replay import check_no_test, load_keypoints, replay

NAMES = {"reproj": "(a) riproiezione", "temporal": "(b) continuita", "temporal_reject": "(c) continuita + rifiuto"}
BGR = {"reproj": (214, 120, 42), "temporal": (52, 104, 235), "temporal_reject": (122, 175, 27)}   # plan palette
PANEL, HEAD, STRIP = 480, 58, 150

ap = argparse.ArgumentParser()
ap.add_argument("--sequence", required=True, help="sequence key, e.g. 09_03_01_000")
ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
ap.add_argument("--keypoints", default="experiments/keypoints/keypoints_val.jsonl")
ap.add_argument("--models", default="configs/drone_models.json")
ap.add_argument("--out", default="experiments/compare")
ap.add_argument("--allow-test", action="store_true")
args = ap.parse_args()

records = [r for r in read_jsonl(Path(args.canonical)) if r["frame_id"].startswith(args.sequence + "_")]
assert records, f"nessun frame per la sequenza {args.sequence}"
check_no_test(records, args.allow_test)
models = load_drone_models(Path(args.models))
kpts = load_keypoints(Path(args.keypoints))
records.sort(key=lambda r: r["frame_idx"])
results = {s: replay(records, kpts, models, s)[args.sequence] for s in STRATEGIES}
n = len(records)

# ---- timeline strip (static background, cursor drawn per frame) ----
W = PANEL * 3
strip_bg = np.full((STRIP, W, 3), 252, np.uint8)
ymax = 180.0
y_of = lambda e: int(STRIP - 14 - min(e, ymax) / ymax * (STRIP - 28))
x_of = lambda i: int(8 + i / max(n - 1, 1) * (W - 16))
for deg in (0, 90, 180):
    cv2.line(strip_bg, (8, y_of(deg)), (W - 8, y_of(deg)), (225, 225, 225), 1)
    cv2.putText(strip_bg, f"{deg}", (W - 34, y_of(deg) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
cv2.putText(strip_bg, "errore di rotazione (gradi)", (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
for k, s in enumerate(STRATEGIES):
    pts = [(x_of(i), y_of(r["rot_err_deg"])) for i, r in enumerate(results[s]) if r["rot_err_deg"] is not None]
    off = (k - 1) * 2                                    # tiny vertical offset so overlapping lines stay visible
    cv2.polylines(strip_bg, [np.array([(x, y + off) for x, y in pts], np.int32).reshape(-1, 1, 2)], False, BGR[s], 2)
    for i, r in enumerate(results[s]):
        if r["rot_err_deg"] is None:
            cv2.line(strip_bg, (x_of(i), STRIP - 8), (x_of(i), STRIP - 2), BGR[s], 1)   # no pose / rejected tick


def text(img, s, org, scale=0.55, color=(30, 30, 30), th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def panel(frame, rec, res, strategy, box):
    img = crop_image(frame, box, PANEL)
    for (x, y) in frame_to_crop(np.array(rec["kpts2d"]), box, PANEL):
        cv2.circle(img, (int(x), int(y)), 9, (0, 220, 0), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(frame_to_crop(np.array(res["kpts_pred"])[:, :2], box, PANEL)):
        if np.isfinite(x):
            cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)
            text(img, str(i + 1), (int(x) + 7, int(y) - 7), 0.6, (0, 0, 255), 2)
    if res["R"] is not None:
        R, t = np.array(res["R"]), np.array(res["t"])
        ends = frame_to_crop(project(np.array(rec["K"]), np.vstack([t, t + 0.12 * R.T])), box, PANEL)
        o = tuple(int(v) for v in ends[0])
        for e, col in zip(ends[1:], [(0, 0, 255), (0, 200, 0), (255, 60, 0)]):
            cv2.arrowedLine(img, o, tuple(int(v) for v in e), col, 3, cv2.LINE_AA, tipLength=0.2)
    bar = np.full((HEAD, PANEL, 3), 255, np.uint8)
    cv2.rectangle(bar, (0, 0), (PANEL, 6), BGR[strategy], -1)
    text(bar, NAMES[strategy], (8, 26), 0.6, (20, 20, 20), 2)
    err = res["rot_err_deg"]
    if res["rejected"]:
        label, col = f"SCARTATO ({res['reject_reason']})", (110, 110, 110)
    elif err is None:
        label, col = "nessuna posa", (110, 110, 110)
    elif err > 90:
        label, col = f"FLIP  errore {err:.0f} gradi", (0, 0, 220)
    else:
        label, col = f"ok  errore {err:.1f} gradi", (40, 140, 40)
    text(bar, label, (8, 50), 0.55, col, 2)
    if res["solution_index"] is not None:
        text(bar, f"soluzione #{res['solution_index']}", (PANEL - 125, 50), 0.45, (90, 90, 90), 1)
    return np.vstack([bar, img])


def compose(i):
    rec = records[i]
    frame = cv2.imread(str(rec["image_path"]))
    k = np.array(rec["kpts2d"])
    side = max(float(np.ptp(k[:, 0])), float(np.ptp(k[:, 1])), 40.0) * 3.0
    cx, cy = (k.min(0) + k.max(0)) / 2
    box = CropBox(cx - side / 2, cy - side / 2, side)
    row = np.hstack([panel(frame, rec, results[s][i], s, box) for s in STRATEGIES])
    strip = strip_bg.copy()
    cv2.line(strip, (x_of(i), 0), (x_of(i), STRIP), (30, 30, 30), 1)
    text(strip, f"{args.sequence}   frame {rec['frame_idx']}/{records[-1]['frame_idx']}", (10, STRIP - 22), 0.5)
    return np.vstack([row, strip])


out = Path(args.out) / args.sequence
out.mkdir(parents=True, exist_ok=True)
first = compose(0)
writer = cv2.VideoWriter(str(out / "compare.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (first.shape[1], first.shape[0]))
disagree = []
for i in range(n):
    writer.write(compose(i))
    flips = [(results[s][i]["rot_err_deg"] or 0) > 90 for s in STRATEGIES]
    if len(set(flips)) > 1:
        disagree.append(i)
writer.release()

# stills: first frame where the strategies disagree, and the middle of the longest disagreement run
picks = []
if disagree:
    picks.append(disagree[0])
    runs, cur = [], [disagree[0]]
    for d in disagree[1:]:
        if d == cur[-1] + 1:
            cur.append(d)
        else:
            runs.append(cur)
            cur = [d]
    runs.append(cur)
    longest = max(runs, key=len)
    picks.append(longest[len(longest) // 2])
for j, i in enumerate(dict.fromkeys(picks)):
    cv2.imwrite(str(out / f"still_{j}_frame{records[i]['frame_idx']}.png"), compose(i))
print(f"{args.sequence}: {n} frame, {len(disagree)} con strategie in disaccordo -> {out}")
for s in STRATEGIES:
    errs = [r["rot_err_deg"] for r in results[s] if r["rot_err_deg"] is not None]
    print(f"  {NAMES[s]:28s} flip {100 * np.mean([e > 90 for e in errs]):5.1f}%  pose {len(errs)}/{n}")
