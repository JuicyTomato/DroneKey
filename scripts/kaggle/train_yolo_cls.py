"""Train the crop classifier (7 drone classes) on the setting-A crops; train split only, `last.pt` is the model.

Kaggle:  !pip install -q ultralytics==8.4.178
         python -u $CODE/scripts/kaggle/train_yolo_cls.py --seed 0 --name cls_s_s0
The val split is only logged by Ultralytics: the checkpoint is last.pt after a fixed number of epochs (30), so the
val stays clean for the class grid (spec 2026-10-27-classe-drone-design.md, 5.1). Augmentation as recipe w1 where it
applies to classification (horizontal flip, HSV, scale). Writes <project>/<name>/run_info.json before training.
The train and val crops (never the test ones) are first copied to --folder on local disk: the Kaggle input mount is
slow per file and the classifier reads every image at every epoch.
Local smoke run: --data-root data/processed/yolo_crops --limit 200 --epochs 1 --device mps --project <dir> --link.
"""
import argparse
import json
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]            # folder that holds src/ and scripts/
sys.path.insert(0, str(CODE / "src"))                  # dronepose without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent))   # kaggle_paths
FOLDER = Path("/tmp/cls_crops")                        # image-folder on local disk (crops copied there)
W1_CLS = {"fliplr": 0.5, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "scale": 0.3}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo26s-cls.pt")
    ap.add_argument("--fallback-model", default="yolo11s-cls.pt", help="used (and recorded) if --model fails to load")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="cls_s_s0")
    ap.add_argument("--project", default="/kaggle/working/runs")
    ap.add_argument("--data-root", default=None, help="yolo_crops folder; default: found in the Kaggle datasets")
    ap.add_argument("--folder", default=str(FOLDER), help="where the image-folder is built")
    ap.add_argument("--link", action="store_true", help="symlink instead of copying the crops (local smoke runs)")
    ap.add_argument("--limit", type=int, default=None, help="crops per split (smoke runs only)")
    ap.add_argument("--device", default=None)
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    from dronepose.common.registry import data_fingerprint, make_run_info
    from dronepose.keypoints.cls_data import build_imagefolder
    from train_yolo_pose import finish_info, read_commit, versions_and_gpu
    from ultralytics import YOLO

    if args.data_root:
        crops = Path(args.data_root)
    else:
        from kaggle_paths import find_dataset
        crops = find_dataset("dronepose-cache") / "yolo_crops"
    counts = build_imagefolder(crops, Path(args.folder), limit=args.limit, copy=not args.link)   # train + val only
    print(f"[{time.time() - t0:7.1f}s] image-folder {args.folder}: {counts}", flush=True)
    try:
        model, used = YOLO(args.model), args.model
    except Exception as e:   # noqa: BLE001 (download / unknown weights): fall back, and record it
        print(f"cannot load {args.model} ({e}); using {args.fallback_model}", flush=True)
        model, used = YOLO(args.fallback_model), args.fallback_model
    versions, gpu = versions_and_gpu()
    run_dir = Path(args.project) / args.name
    info = make_run_info({**vars(args), "model_used": used, "recipe": "w1_cls", "setting": "crop_cls"},
                         read_commit(), data_fingerprint(crops), versions, gpu)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    try:
        model.train(data=args.folder, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                    seed=args.seed, project=args.project, name=args.name, exist_ok=True,
                    **({"device": args.device} if args.device else {}), **W1_CLS)
    finally:
        finish_info(run_dir, info)
        print(f"[{time.time() - t0:7.1f}s] run_info.json -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
