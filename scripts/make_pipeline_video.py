"""Video of the pipeline on val sequences: box and class from C (optional) -> crop keypoints -> IPPE -> `global` -> forecast.

Usage: python scripts/make_pipeline_video.py --kpts K.jsonl [--det D.jsonl] [--cls gt|frame|causal] [--split val]
           --out OUT.mp4 SEQ...
Drawn: GT keypoints (green rings), predicted keypoints (red), body axes, forecast (yellow), C box (cyan, with --det).
--cls picks the 3D model class (gt = true drone; frame/causal need --det). Original PNGs are read only; needs ffmpeg.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.keypoints.detections import class_causal_vote, class_per_frame, load_detections
from dronepose.pose.replay import check_no_test, load_keypoints, replay, sequence_key
from dronepose.viz.overlay import draw_overlay

W, H = 960, 540
BAND = 88   # text band above the frame, so the text never covers the drone
CLS_LABEL = {"gt": "classe vera (GT)", "frame": "classe C (per frame)", "causal": "classe C (voto causale)"}


def render(records, res, seqs, dets, drone_of, mode, ff) -> None:
    """Draw every frame of `seqs` and write it to the ffmpeg pipe."""
    for key in seqs:
        by_id = {r["frame_id"]: r for r in records if sequence_key(r["frame_id"]) == key}
        for x in res[key]:
            r = by_id[x["frame_id"]]
            img = cv2.imread(r["image_path"])
            if img is None:
                sys.exit(f"cannot read image {r['image_path']}")
            frame = draw_overlay(img, np.array(r["K"]), np.array(r["kpts2d"]), np.array(x["kpts_pred"]),
                                 x["R"], x["t"], x["future_pred"])
            box = dets[x["frame_id"]].box if dets and x["frame_id"] in dets else None
            if box is not None:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            c, d = drone_of[x["frame_id"]] if drone_of else r["drone"], r["drone"]
            rot = x["rot_err_sym_deg"]
            lines = [f"seq {key}  frame {x['frame_idx']}",
                     f"{CLS_LABEL[mode]}: {c}   vera: {d}" + ("" if c == d else "   SBAGLIATA"),
                     "posa scartata" if rot is None else f"err. rotazione {rot:.1f} deg   err. traslazione {x['t_err']:.3f}"]
            frame = np.vstack([np.full((BAND, W, 3), 24, np.uint8), frame])
            for i, s in enumerate(lines):
                col = (80, 80, 255) if (i == 1 and c != d) else (235, 235, 235)
                cv2.putText(frame, s, (12, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
            ff.stdin.write(frame.tobytes())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seqs", nargs="+", metavar="SEQ", help='sequence ids such as "14_03_10_000"')
    ap.add_argument("--kpts", required=True, help="keypoint JSONL")
    ap.add_argument("--det", default=None, help="detection JSONL (box and class of C)")
    ap.add_argument("--cls", default="gt", choices=["gt", "frame", "causal"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.det is None and args.cls != "gt":
        sys.exit("--cls frame/causal needs --det")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg not found: install it (e.g. `brew install ffmpeg`) and retry")

    records = [r for r in read_jsonl(Path(f"data/processed/canonical_{args.split}.jsonl"))
               if sequence_key(r["frame_id"]) in args.seqs]
    check_no_test(records, args.allow_test)
    missing = set(args.seqs) - {sequence_key(r["frame_id"]) for r in records}
    if missing:
        sys.exit(f"sequences not in {args.split}: {sorted(missing)}")
    dets = load_detections(Path(args.det)) if args.det else None
    drone_of = None
    if args.cls != "gt":
        drone_of = (class_per_frame if args.cls == "frame" else class_causal_vote)(records, dets)
    res = replay(records, load_keypoints(Path(args.kpts)), load_drone_models(Path("configs/drone_models.json")),
                 "global", model_drone=drone_of)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H + BAND}",
                           "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                          stdin=subprocess.PIPE)
    try:
        render(records, res, args.seqs, dets, drone_of, args.cls, ff)
    finally:
        ff.stdin.close()
        ff.wait()
    print(out, len(records), "frame")


if __name__ == "__main__":
    main()
