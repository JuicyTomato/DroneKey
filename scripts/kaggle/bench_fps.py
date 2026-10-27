"""Per-stage latency and end-to-end FPS of pipelines B (C -> crop -> pose) and C (frames) at batch 1.

Kaggle (T4; inputs dronepose-code, dronepose-cache, dronepose-frames, weights dataset; pip install ultralytics==8.4.178):
    python -u $CODE/scripts/kaggle/bench_fps.py --c-weights <c/best.pt> --crop-weights <crop/best.pt>
Stages, timed per frame with a device sync: c_640 (predict of C on the 640x360 frame), crop_jpeg (crop + JPEG q95 from a
frame upscaled to 1920x1080, as the original would be; the upscale itself is not timed), crop_256 (crop pose model),
ippe (CPU, K and drone from canonical_val.jsonl). B = c_640 + crop_jpeg + crop_256 + ippe; C = c_640 + ippe.
Frames without a detection (or with non-finite keypoints) skip the stages that need them: `n` per stage tells.
--cls-weights W adds stage cls_256 (crop classifier on the same crop) and B_cls = B + cls_256; without it the output
is unchanged.
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
from ultralytics import YOLO  # noqa: E402

import ultralytics  # noqa: E402
from dronepose.common.timing import end_to_end_fps, latency_summary  # noqa: E402
from dronepose.data.crops import CROP_SIZE  # noqa: E402
from dronepose.data.drone_models import load_drone_models  # noqa: E402
from dronepose.keypoints.export import crop_result_to_frame  # noqa: E402
from dronepose.keypoints.two_stage import crop_from_detection, top_detection_row  # noqa: E402
from dronepose.pose.pnp import solve_ippe  # noqa: E402


def default_device() -> str:
    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_sync(device: str):
    """Block until queued GPU work is done (no-op on CPU)."""
    if device not in ("cpu", "mps"):
        return torch.cuda.synchronize
    if device == "mps" and hasattr(torch, "mps"):
        return torch.mps.synchronize
    return lambda: None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c-weights", required=True)
    ap.add_argument("--crop-weights", required=True)
    ap.add_argument("--frames", type=int, default=550, help="evenly spaced val frames; the first 50 are warmup")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default="fps.json")
    ap.add_argument("--frames-dir", default=None, help="yolo_frames folder (default: dronepose-frames on Kaggle)")
    ap.add_argument("--canonical", default=None, help="canonical_val.jsonl (default: dronepose-cache on Kaggle)")
    ap.add_argument("--device", default=None, help="default: 0 on CUDA, else mps if available, else cpu")
    ap.add_argument("--cls-weights", default=None, help="crop classifier: adds stage cls_256 and B_cls")
    args = ap.parse_args(argv)
    device = args.device or default_device()
    sync = make_sync(device)

    if args.frames_dir is None or args.canonical is None:
        from kaggle_paths import find_dataset
    frames_dir = Path(args.frames_dir) if args.frames_dir else find_dataset("dronepose-frames") / "yolo_frames"
    canonical = Path(args.canonical) if args.canonical else find_dataset("dronepose-cache") / "canonical_val.jsonl"
    with open(frames_dir / "frames_val.jsonl") as f:
        ids = [json.loads(l)["frame_id"] for l in f if l.strip()]
    ids = [ids[i] for i in np.linspace(0, len(ids) - 1, min(args.frames, len(ids))).astype(int)]
    with open(canonical) as f:
        recs = {r["frame_id"]: r for r in map(json.loads, (l for l in f if l.strip()))}
    models = load_drone_models(CODE / "configs/drone_models.json")

    # all frames in RAM before the clock starts
    imgs = []
    for fid in ids:
        img = cv2.imread(str(frames_dir / "images" / "val" / f"{fid}.jpg"))
        if img is None:
            raise FileNotFoundError(fid)
        imgs.append(img)
    det_model, crop_model = YOLO(args.c_weights), YOLO(args.crop_weights)
    cls_model = YOLO(args.cls_weights) if args.cls_weights else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else device
    print(f"device {device} ({gpu}), {len(ids)} frames, warmup {args.warmup}", flush=True)

    ms: dict[str, list[float]] = {"c_640": [], "crop_jpeg": [], "crop_256": [], "ippe": []}
    if cls_model is not None:
        ms["cls_256"] = []
    nodet = 0
    t_start = time.time()
    for k, (fid, img) in enumerate(zip(ids, imgs)):
        # stage c_640
        sync()
        t = time.perf_counter()
        res = det_model.predict(img, imgsz=640, device=device, verbose=False)[0]
        has = res.boxes is not None and len(res.boxes) > 0
        if has:
            xyxy, conf, cls = (v.cpu().numpy() for v in (res.boxes.xyxy, res.boxes.conf, res.boxes.cls))
            kxy = res.keypoints.xy.cpu().numpy()
            kconf = None if res.keypoints.conf is None else res.keypoints.conf.cpu().numpy()
            row = top_detection_row(fid, xyxy, conf, cls, kxy, kconf, 1920 / res.orig_shape[1])
        sync()
        ms["c_640"].append(1000 * (time.perf_counter() - t))
        if not has:
            nodet += 1
            continue
        # stage crop_jpeg (original-size frame: upscaled here, untimed)
        big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        t = time.perf_counter()
        crop, box = crop_from_detection(big, row["box"])
        ms["crop_jpeg"].append(1000 * (time.perf_counter() - t))
        # stage crop_256
        sync()
        t = time.perf_counter()
        r = crop_model.predict(crop, imgsz=CROP_SIZE, device=device, verbose=False)[0]
        kp = r.keypoints
        xy = None if kp is None else kp.xy.cpu().numpy()
        kc = None if kp is None or kp.conf is None else kp.conf.cpu().numpy()
        bc = None if r.boxes is None else r.boxes.conf.cpu().numpy()
        kpts = crop_result_to_frame(xy, kc, bc, box, CROP_SIZE)
        sync()
        ms["crop_256"].append(1000 * (time.perf_counter() - t))
        # stage cls_256 (optional): class probabilities of the same crop
        if cls_model is not None:
            sync()
            t = time.perf_counter()
            cls_model.predict(crop, imgsz=CROP_SIZE, device=device, verbose=False)[0].probs.data.cpu().numpy()
            sync()
            ms["cls_256"].append(1000 * (time.perf_counter() - t))
        # stage ippe (CPU)
        if np.isfinite(kpts[:, :2]).all():
            rec = recs[fid]
            t = time.perf_counter()
            solve_ippe(models[rec["drone"]].kpts_flu, kpts[:, :2], np.array(rec["K"], float))
            ms["ippe"].append(1000 * (time.perf_counter() - t))
        if k % 100 == 0:
            print(f"{k}/{len(ids)} {time.time() - t_start:.0f}s", flush=True)

    stages = {s: latency_summary(v, args.warmup) for s, v in ms.items()}
    med = {s: v["median_ms"] for s, v in stages.items()}
    fps = {"B": end_to_end_fps([med["c_640"], med["crop_jpeg"], med["crop_256"], med["ippe"]]),
           "C": end_to_end_fps([med["c_640"], med["ippe"]])}
    if cls_model is not None:
        fps["B_cls"] = end_to_end_fps([med["c_640"], med["crop_jpeg"], med["crop_256"], med["cls_256"], med["ippe"]])
    out = {"stages": stages, "fps": fps, "frames": len(ids), "no_detection": nodet, "warmup": args.warmup,
           "device": device, "gpu": gpu, "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                                                       "cv2": cv2.__version__, "python": platform.python_version()},
           "weights": {"c": args.c_weights, "crop": args.crop_weights, **({"cls": args.cls_weights} if cls_model else {})}}
    for s, v in stages.items():
        print(f"{s:10s} n={v['n']:4d} median {v['median_ms']:7.2f} ms  p90 {v['p90_ms']:7.2f} ms")
    print(f"FPS end-to-end  B: {fps['B']:.1f}  C: {fps['C']:.1f}   ({nodet} frames without detection)")
    if "B_cls" in fps:
        print(f"FPS end-to-end  B + classifier: {fps['B_cls']:.1f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
