"""Click the 4 propeller hubs of the Mavic 2 on the MAV6D frames chosen by mav6d_select.py (OpenCV window).

  .venv/bin/python scripts/real/annotate_hubs.py [--list experiments/real/mav6d/annotate_list.json]
      [--out experiments/real/mav6d/hub_annotations.json] [--target 40]

Order (the drone's own left/right, looking along the red FRONT arrow drawn from the VICON pose):
  1 front RIGHT, 2 front LEFT, 3 rear LEFT, 4 rear RIGHT  (same as key1-key4 of 6DroneSyn)
Click the centre of the motor (the rotation axis, on top of the motor, under the propeller disc).
Keys: click = next hub | x = hub not visible (estimate impossible) | u = undo | Enter/Space = frame done |
      s = skip frame (blurred or unclear) | b = back one frame | q = save and quit.
Saved after every frame (resumable): {frame: {"hubs": [[x, y] | null] x 4, "use": ..., "skipped": bool}} in pixels
of the original 1920x1080 image. Images are only read.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
from dronepose.data.mav6d import project, read_label  # noqa: E402

NAMES = ["1 anteriore DESTRA", "2 anteriore SINISTRA", "3 posteriore SINISTRA", "4 posteriore DESTRA"]
COLORS = [(0, 0, 255), (0, 200, 255), (255, 160, 0), (255, 0, 200)]
WIN = "MAV6D hub"
VIEW = 1000   # window side, px


def view_of(root: Path, item: dict):
    im = cv2.imread(str(root / "JPEGImages" / f"{item['frame']}.jpg"))
    x0, y0, x1, y1 = item["box"]
    half = int(0.8 * max(x1 - x0, y1 - y0))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ox, oy = cx - half, cy - half
    s = VIEW / (2 * half)
    M = np.float32([[s, 0, -ox * s], [0, s, -oy * s]])
    crop = cv2.warpAffine(im, M, (VIEW, VIEW), flags=cv2.INTER_CUBIC, borderValue=(60, 60, 60))
    R, t = read_label(root / "labels" / f"{item['frame']}.txt")
    span = 0.2
    ax = project(np.array([[0, 0, 0], [span, 0, 0], [0, span, 0], [0, 0, span]]), R, t)
    to_v = lambda p: (int(round((p[0] - ox) * s)), int(round((p[1] - oy) * s)))   # noqa: E731
    return crop, to_v, lambda u, v: (u / s + ox, v / s + oy), [to_v(p) for p in ax]


def draw(base, axes, pts_v, frame, n_done, target, msg):
    img = base.copy()
    o, fx, fy, fz = axes
    cv2.arrowedLine(img, o, fx, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.12)
    cv2.putText(img, "AVANTI", fx, 0, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(img, o, fy, (0, 180, 0), 2, cv2.LINE_AA, tipLength=0.12)
    cv2.putText(img, "SINISTRA", fy, 0, 0.6, (0, 180, 0), 2, cv2.LINE_AA)
    cv2.arrowedLine(img, o, fz, (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.12)
    for i, p in enumerate(pts_v):
        if p is not None:
            cv2.circle(img, p, 6, COLORS[i], 2, cv2.LINE_AA)
            cv2.drawMarker(img, p, COLORS[i], cv2.MARKER_CROSS, 10, 1)
            cv2.putText(img, str(i + 1), (p[0] + 8, p[1] - 8), 0, 0.8, COLORS[i], 2, cv2.LINE_AA)
    nxt = NAMES[len(pts_v)] if len(pts_v) < 4 else "Invio = conferma, u = annulla"
    cv2.rectangle(img, (0, 0), (VIEW, 62), (30, 30, 30), -1)
    cv2.putText(img, f"[{n_done}/{target}] {frame}   clicca: {nxt}", (10, 25), 0, 0.65, (255, 255, 255), 2)
    cv2.putText(img, "x non visibile  u annulla  s salta frame  b indietro  q esci" + (f"   {msg}" if msg else ""),
                (10, 52), 0, 0.55, (200, 200, 200), 1)
    cv2.imshow(WIN, img)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/Volumes/MakeMePasta/MAV6D/mavic2")
    ap.add_argument("--list", default="experiments/real/mav6d/annotate_list.json")
    ap.add_argument("--out", default="experiments/real/mav6d/hub_annotations.json")
    ap.add_argument("--target", type=int, default=40)
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    items = json.loads(Path(args.list).read_text())
    ann = json.loads(out.read_text()) if out.exists() else {}
    save = lambda: out.write_text(json.dumps(ann, indent=1))   # noqa: E731

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    clicks: list = []
    cv2.setMouseCallback(WIN, lambda ev, u, v, *_: clicks.append((u, v)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    i, history = 0, []
    while i < len(items):
        n_done = sum(1 for a in ann.values() if not a["skipped"])
        if n_done >= args.target:
            break
        item = items[i]
        if item["frame"] in ann:
            i += 1
            continue
        base, to_v, to_img, axes = view_of(root, item)
        pts_v, pts, msg = [], [], ""
        while True:
            while clicks and len(pts) < 4:
                u, v = clicks.pop(0)
                pts_v.append((u, v))
                pts.append([round(x, 2) for x in to_img(u, v)])
            clicks.clear()
            draw(base, axes, pts_v, item["frame"], n_done, args.target, msg)
            k = cv2.waitKey(30) & 0xFF
            if k == 255:
                continue
            msg = ""
            if k == ord("x") and len(pts) < 4:
                pts_v.append(None)
                pts.append(None)
            elif k == ord("u") and pts:
                pts_v.pop()
                pts.pop()
            elif k in (13, 10, 32):
                if len(pts) < 4:
                    msg = "servono 4 punti (x = non visibile)"
                    continue
                ann[item["frame"]] = {"hubs": pts, "use": item["use"], "skipped": False}
                break
            elif k == ord("s"):
                ann[item["frame"]] = {"hubs": None, "use": item["use"], "skipped": True}
                break
            elif k == ord("b") and history:
                prev = history.pop()
                ann.pop(prev, None)
                i = next(j for j, it in enumerate(items) if it["frame"] == prev)
                save()
                break
            elif k == ord("q"):
                save()
                cv2.destroyAllWindows()
                print(f"saved {len(ann)} frames -> {out}")
                return
        if item["frame"] in ann:
            history.append(item["frame"])
            save()
            i += 1
    save()
    cv2.destroyAllWindows()
    done = sum(1 for a in ann.values() if not a["skipped"])
    print(f"{done} frames annotated, {sum(a['skipped'] for a in ann.values())} skipped -> {out}")


if __name__ == "__main__":
    main()
