"""MAV6D Mavic 2: per-frame table and the frames to annotate by hand (propeller hubs, to triangulate the 3D model).

  python scripts/real/mav6d_select.py --root /Volumes/MakeMePasta/MAV6D/mavic2 --out-dir experiments/real/mav6d [--n 40]

Writes OUT/frames.jsonl (frame, split, distance, mask box, sharpness, image speed, viewing angles) and
OUT/annotate_list.json: N frames close to the camera (< --max-dist m), with the drone slow in the image (speed of the
projected VICON origin between consecutive frames, the cause of the motion blur; sharpness of the mask breaks ties),
whole drone inside the image, spread over the viewing azimuth (8 bins) and the sequences, at most --per-seq per
sequence; then --reserve more frames, to use when one is skipped. Every fourth frame is "holdout" (not used to
triangulate: it measures the annotation error).
Images are only read.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
from dronepose.data.mav6d import IMAGE_SIZE, project, read_label  # noqa: E402


def frame_row(root: Path, rel: str, split: str, step_ok: bool) -> dict | None:
    R, t = read_label(root / "labels" / f"{rel}.txt")
    row = {"frame": rel, "split": split, "dist": float(np.linalg.norm(t)),
           "azimuth_deg": float(np.degrees(np.arctan2(R[0, 0], R[2, 0]))),   # body x in the camera x-z plane
           "x_towards_camera": bool(R[2, 0] < 0), "uv": project(np.zeros((1, 3)), R, t)[0].tolist()}
    if not step_ok:
        return row
    m = cv2.imread(str(root / "mask" / f"{rel}.jpg"), cv2.IMREAD_GRAYSCALE)
    im = cv2.imread(str(root / "JPEGImages" / f"{rel}.jpg"), cv2.IMREAD_GRAYSCALE)
    if m is None or im is None:
        return row
    on = m > 127
    ys, xs = np.nonzero(on)
    if len(xs) < 200:
        return row
    lap = cv2.Laplacian(im, cv2.CV_64F)
    row.update(box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], mask_px=int(on.sum()),
               sharpness=float(lap[cv2.erode(on.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0].var()))
    return row


def add_speed(rows: list[dict]) -> None:
    """px_speed: pixels per second of the projected VICON origin, from the neighbouring frames of the sequence."""
    seqs: dict[str, list[dict]] = {}
    for r in rows:
        seqs.setdefault(r["frame"].rsplit("/", 1)[0], []).append(r)
    for rs in seqs.values():
        rs.sort(key=lambda r: int(r["frame"].rsplit("/", 1)[1]))
        for i, r in enumerate(rs):
            nb = [rs[j] for j in (i - 1, i + 1) if 0 <= j < len(rs)]
            v = [np.linalg.norm(np.subtract(r["uv"], o["uv"])) /
                 max(abs(int(r["frame"][-19:]) - int(o["frame"][-19:])) * 1e-9, 1e-3) for o in nb]
            r["px_speed"] = float(max(v)) if v else float("inf")


def pick(rows: list[dict], n: int, max_dist: float, per_seq: int, margin: int = 20) -> list[dict]:
    W, H = IMAGE_SIZE
    ok = [r for r in rows if "box" in r and r["dist"] < max_dist and r["box"][0] > margin and r["box"][1] > margin
          and r["box"][2] < W - margin and r["box"][3] < H - margin]
    bins: dict[int, list[dict]] = {}
    for r in sorted(ok, key=lambda r: (round(r["px_speed"] / 25), -r["sharpness"])):
        bins.setdefault(int((r["azimuth_deg"] + 180) // 45) % 8, []).append(r)
    chosen, per = [], {}
    while len(chosen) < n and any(bins.values()):
        for b in sorted(bins):
            while bins[b]:
                r = bins[b].pop(0)
                seq = r["frame"].rsplit("/", 1)[0]
                if per.get(seq, 0) < per_seq and all(abs(int(r["frame"][-19:]) - int(c["frame"][-19:])) > 5e8
                                                    for c in chosen if c["frame"].startswith(seq)):
                    per[seq] = per.get(seq, 0) + 1
                    chosen.append(r)
                    break
            if len(chosen) == n:
                break
    return chosen


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/Volumes/MakeMePasta/MAV6D/mavic2")
    ap.add_argument("--out-dir", default="experiments/real/mav6d")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--reserve", type=int, default=20)
    ap.add_argument("--max-dist", type=float, default=2.6)
    ap.add_argument("--per-seq", type=int, default=2)
    ap.add_argument("--image-step", type=int, default=5, help="read images (mask, sharpness) every N-th frame")
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out_dir)
    split = {}
    for s in ("train", "test"):
        for line in (root / f"{s}.txt").read_text().split():
            split["/".join(Path(line).with_suffix("").parts[-3:])] = s
    rels = sorted(str(p.relative_to(root / "labels").with_suffix("")) for p in (root / "labels").rglob("*.txt")
                  if not p.name.startswith("._"))
    rows = [r for i, rel in enumerate(rels)
            if (r := frame_row(root, rel, split.get(rel, "?"), i % args.image_step == 0)) is not None]
    add_speed(rows)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "frames.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    chosen = pick(rows, args.n + args.reserve, args.max_dist, args.per_seq)
    for i, r in enumerate(chosen):
        r["use"] = "reserve" if i >= args.n else ("holdout" if i % 4 == 3 else "triangulate")
    (out / "annotate_list.json").write_text(json.dumps(chosen, indent=1))
    d = np.array([r["dist"] for r in rows])
    print(f"{len(rows)} frames ({sum(r['split'] == 'test' for r in rows)} test), distance p5/50/95 "
          f"{np.percentile(d, [5, 50, 95]).round(2)} m; {len(chosen)} to annotate (median image speed "
          f"{np.median([c['px_speed'] for c in chosen]):.0f} px/s) from "
          f"{len({c['frame'].rsplit('/', 1)[0] for c in chosen})} sequences, azimuth bins "
          f"{sorted({int((c['azimuth_deg'] + 180) // 45) % 8 for c in chosen})} -> {out}")


if __name__ == "__main__":
    main()
