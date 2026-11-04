"""MAV6D Mavic 2 -> canonical-like records with keypoints and pose from VICON (real test data for E2).

  .venv/bin/python scripts/real/mav6d_build.py [--model experiments/real/mav6d/hub_model.json]
      [--frames experiments/real/mav6d/frames.jsonl] [--out data/processed/mav6d_gt.jsonl]
      [--gt-box-out data/processed/mav6d_gtbox.jsonl]

The hubs triangulated by mav6d_hubs.py are projected with the refined VICON -> camera transform on every frame.
Fields as in canonical_*.jsonl where they exist: frame_id ("<group>_<seq>_<timestamp>"), image_path, kpts2d (key1-key4,
distorted pixels of the original image, the ones a detector sees), kpts3d_cam, R_gt / t_gt (body FLU about the hub
centroid, metres), K, bbox_xyxy (rule of label_box_xyxy), drone "05" (Mavic 2 Pro in 6DroneSyn), scene (group),
sequence, frame_idx (order in the sequence), split "mav6d" (an external test set, never a 6DroneSyn split). Extra: dist
(lens distortion, OpenCV order), authors_split (their train/test), distance_m, px_speed (image speed of the drone,
motion blur proxy), in_image (all 4 keypoints inside the frame), annotated ("fit" / "holdout" / null: frames clicked
by hand). Also a detection file with the GT label box of every frame (setting A through the setting-B exporters).
Images are not opened here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
from dronepose.data.crops import label_box_xyxy  # noqa: E402
from dronepose.data.mav6d import DIST, IMAGE_SIZE, K, project, read_label  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/Volumes/MakeMePasta/MAV6D/mavic2")
    ap.add_argument("--model", default="experiments/real/mav6d/hub_model.json")
    ap.add_argument("--frames", default="experiments/real/mav6d/frames.jsonl")
    ap.add_argument("--ann", default="experiments/real/mav6d/hub_annotations.json")
    ap.add_argument("--out", default="data/processed/mav6d_gt.jsonl")
    ap.add_argument("--gt-box-out", default="data/processed/mav6d_gtbox.jsonl")
    args = ap.parse_args(argv)
    root = Path(args.root)
    model = json.loads(Path(args.model).read_text())
    hubs, centroid, T = (np.array(model[k]) for k in ("hubs_body", "centroid_body", "cam_from_vicon"))
    ann = json.loads(Path(args.ann).read_text()) if Path(args.ann).exists() else {}
    W, H = IMAGE_SIZE
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = [json.loads(line) for line in open(args.frames)]
    order: dict[str, list[int]] = {}
    for fr in frames:
        seq, ts = fr["frame"].rsplit("/", 1)
        order.setdefault(seq, []).append(int(ts))
    order = {s: {ts: i for i, ts in enumerate(sorted(v))} for s, v in order.items()}
    n, n_in = 0, 0
    with open(out, "w") as f, open(args.gt_box_out, "w") as fb:
        for fr in frames:
            seq, ts = fr["frame"].rsplit("/", 1)
            R, t = read_label(root / "labels" / f"{fr['frame']}.txt", T)
            k2 = project(hubs, R, t)
            inside = bool(((k2 >= 0) & (k2 < [W, H])).all())
            a = ann.get(fr["frame"])
            rec = {"frame_id": fr["frame"].replace("/", "_"), "image_path": str(root / "JPEGImages" / f"{fr['frame']}.jpg"),
                   "dataset": "MAV6D", "drone": "05", "scene": seq.split("/")[0], "sequence": seq.replace("/", "_"),
                   "frame_idx": order[seq][int(ts)], "split": "mav6d", "authors_split": fr["split"], "units": "metres",
                   "K": K.tolist(), "dist": DIST.tolist(), "bbox_xyxy": label_box_xyxy(k2).tolist(),
                   "kpts2d": k2.tolist(), "kpts3d_cam": ((R @ hubs.T).T + t).tolist(),
                   "R_gt": R.tolist(), "t_gt": (R @ centroid + t).tolist(),
                   "distance_m": fr["dist"], "px_speed": fr.get("px_speed"), "in_image": inside,
                   "annotated": None if a is None or a["skipped"] else ("holdout" if a["use"] == "holdout" else "fit")}
            f.write(json.dumps(rec) + "\n")
            fb.write(json.dumps({"frame_id": rec["frame_id"], "box": rec["bbox_xyxy"], "box_conf": 1.0, "cls": 4}) + "\n")
            n, n_in = n + 1, n_in + inside
    print(f"{n} records ({n_in} with all 4 keypoints inside the image) -> {out}, GT boxes -> {args.gt_box_out}")


if __name__ == "__main__":
    main()
