"""Per-stage latency and FPS of DroneKey in pipeline B (crop+JPEG -> network -> IPPE) at batch 1, crops built in memory.

Kaggle (T4; inputs dronepose-code, dronepose-cache, dronepose-frames, weights dataset; no ultralytics needed):
    python -u $CODE/scripts/kaggle/bench_dronekey.py --weights <dronekey/best.pt> --variant paper --n 500
Frames: same source as bench_fps.py (dronepose-frames yolo_frames/images/val, evenly spaced, upscaled to 1920x1080 untimed);
the box is the GT label box (C's latency is measured separately by bench_fps.py: FPS of B = C + crop + DroneKey + IPPE).
Stages, timed per frame: crop_jpeg (crop_from_detection), net (preprocess + forward + crop_to_frame, device synced),
ippe (CPU, K and drone model from the canonical record). The first --warmup (50) frames are dropped; n + warmup are run.
Network in float32 (training used fp16 AMP): the conservative figure.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "src"))                  # dronepose without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent))   # kaggle_paths

from dronepose.common.timing import end_to_end_fps, latency_summary  # noqa: E402
from dronepose.data.canonical import read_jsonl  # noqa: E402
from dronepose.data.crops import label_box_xyxy  # noqa: E402
from dronepose.data.drone_models import load_drone_models  # noqa: E402
from dronepose.keypoints.dronekey.predict import load_predictor  # noqa: E402
from dronepose.keypoints.dronekey.train import pick_device  # noqa: E402
from dronepose.keypoints.two_stage import crop_from_detection  # noqa: E402
from dronepose.pose.pnp import solve_ippe  # noqa: E402


def make_sync(device: torch.device):
    """Block until queued GPU work is done (no-op on CPU)."""
    if device.type == "cuda":
        return torch.cuda.synchronize
    if device.type == "mps":
        return torch.mps.synchronize
    return lambda: None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", default=None, help="name in configs/dronekey.json, cross-checked with the checkpoint")
    ap.add_argument("--n", type=int, default=500, help="measured frames (after the warmup)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default="fps_dronekey.json")
    ap.add_argument("--frames-dir", default=None, help="yolo_frames folder (default: dronepose-frames on Kaggle)")
    ap.add_argument("--canonical", default=None, help="canonical_val.jsonl (default: dronepose-cache on Kaggle)")
    ap.add_argument("--models", default=str(CODE / "configs/drone_models.json"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tiny", action="store_true", help="TinyBackbone checkpoint (smoke test only)")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    sync = make_sync(device)

    if args.frames_dir is None or args.canonical is None:
        from kaggle_paths import find_dataset
    frames_dir = Path(args.frames_dir) if args.frames_dir else find_dataset("dronepose-frames") / "yolo_frames"
    canonical = Path(args.canonical) if args.canonical else find_dataset("dronepose-cache") / "canonical_val.jsonl"
    recs = {r["frame_id"]: r for r in read_jsonl(canonical)}
    ids = [r["frame_id"] for r in read_jsonl(frames_dir / "frames_val.jsonl")]
    total = args.n + args.warmup
    ids = [ids[i] for i in np.linspace(0, len(ids) - 1, min(total, len(ids))).astype(int)]
    models = load_drone_models(Path(args.models))
    pred = load_predictor(Path(args.weights), None, device, variant=args.variant, tiny=args.tiny)

    # all frames (upscaled to the original size) and their GT label boxes in RAM before the clock starts
    frames = []
    for fid in ids:
        img = cv2.imread(str(frames_dir / "images" / "val" / f"{fid}.jpg"))
        if img is None:
            raise FileNotFoundError(fid)
        frames.append((cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR),
                       label_box_xyxy(np.array(recs[fid]["kpts2d"], float))))
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    print(f"device {device} ({gpu}), {len(ids)} frames, warmup {args.warmup}", flush=True)

    ms: dict[str, list[float]] = {"crop_jpeg": [], "net": [], "ippe": []}
    nonfinite = 0
    t_start = time.time()
    for k, (fid, (big, box_xyxy)) in enumerate(zip(ids, frames)):
        t = time.perf_counter()
        crop, box = crop_from_detection(big, box_xyxy)
        ms["crop_jpeg"].append(1000 * (time.perf_counter() - t))
        sync()
        t = time.perf_counter()
        kpts = pred.predict_crops([crop], [box], batch=1)[0]
        sync()
        ms["net"].append(1000 * (time.perf_counter() - t))
        if np.isfinite(kpts[:, :2]).all():   # otherwise no pose: ippe not timed (`n` per stage tells)
            rec = recs[fid]
            t = time.perf_counter()
            solve_ippe(models[rec["drone"]].kpts_flu, kpts[:, :2], np.array(rec["K"], float))
            ms["ippe"].append(1000 * (time.perf_counter() - t))
        else:
            nonfinite += 1
        if k % 100 == 0:
            print(f"{k}/{len(ids)} {time.time() - t_start:.0f}s", flush=True)

    stages = {s: latency_summary(v, args.warmup) for s, v in ms.items()}
    med = {s: v["median_ms"] for s, v in stages.items()}
    fps = end_to_end_fps([med["crop_jpeg"], med["net"], med["ippe"]])
    out = {"stages": stages, "fps_crop_net_ippe": fps, "frames": len(ids), "non_finite_keypoints": nonfinite,
           "warmup": args.warmup, "device": str(device), "gpu": gpu, "weights": args.weights,
           "versions": {"torch": torch.__version__, "cv2": cv2.__version__, "python": platform.python_version()}}
    for s, v in stages.items():
        print(f"{s:10s} n={v['n']:4d} median {v['median_ms']:7.2f} ms  p90 {v['p90_ms']:7.2f} ms")
    print(f"FPS crop+net+ippe: {fps:.1f}   ({nonfinite} frames with non-finite keypoints)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
