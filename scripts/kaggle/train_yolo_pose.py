"""Train YOLO-Pose on the dronepose crops. Resumable: re-run with --resume-from <last.pt>.

Kaggle:  !pip install -q ultralytics
         !python /kaggle/input/dronepose-cache/code/scripts/kaggle/train_yolo_pose.py --epochs 10
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/kaggle/input/dronepose-cache/yolo_crops/data.yaml")
ap.add_argument("--model", default="yolo11n-pose.pt", help="pretrained weights (latest Ultralytics pose model available)")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--imgsz", type=int, default=256)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--project", default="/kaggle/working/runs")
ap.add_argument("--name", default="skeleton")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume-from", default=None, help="path to last.pt of an interrupted run")
args = ap.parse_args()

if args.resume_from:
    YOLO(args.resume_from).train(resume=True)
else:
    YOLO(args.model).train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project=args.project, name=args.name, exist_ok=True, seed=args.seed,
        fliplr=0.5, degrees=30.0, scale=0.3, mosaic=0.0, save_period=1,
    )
