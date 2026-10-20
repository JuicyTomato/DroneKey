"""Train YOLO-Pose on the dronepose crops (setting A/B) or full frames (setting C). Resumable: --resume-from <last.pt>.

Datasets are located automatically under /kaggle/input (whatever the mount layout).
Kaggle:  !pip install -q ultralytics==8.4.178
         %%bash  (see docs/kaggle/README.md for the CODE=... lookup)
         python -u $CODE/scripts/kaggle/train_yolo_pose.py --recipe final --seed 0 --hours 3 --name crop_s_final_s0
Writes <project>/<name>/run_info.json (commit from $CODE/COMMIT, data hash, versions, GPU) before training.
"""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]            # folder that holds src/ and scripts/
sys.path.insert(0, str(CODE / "src"))                  # dronepose without pip install
FRAMES_LOCAL = Path("/tmp/yolo_frames")                # local disk: the input mount is slow per file


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=["crop", "frame"], default="crop")
    ap.add_argument("--data", default=None, help="data.yaml; default: auto-built from the Kaggle datasets")
    ap.add_argument("--model", default="yolo26s-pose.pt", help="pretrained weights")
    ap.add_argument("--hours", type=float, default=None, help="time budget (Ultralytics time=, epochs=300, patience=1000)")
    ap.add_argument("--epochs", type=int, default=None, help="number of epochs; only without --hours (default 10)")
    ap.add_argument("--recipe", choices=["w1", "final"], default="final")
    ap.add_argument("--imgsz", type=int, default=None, help="default 256 (crop) / 640 (frame)")
    ap.add_argument("--batch", type=int, default=None, help="default 64 (crop) / 32 (frame)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-period", type=int, default=5)
    ap.add_argument("--project", default="/kaggle/working/runs")
    ap.add_argument("--name", default="skeleton")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume-from", default=None, help="path to last.pt of an interrupted run (pass the same --setting)")
    args = ap.parse_args(argv)
    if args.hours is not None and args.epochs is not None:
        ap.error("--epochs and --hours are mutually exclusive")
    args.imgsz = args.imgsz or (256 if args.setting == "crop" else 640)
    args.batch = args.batch or (64 if args.setting == "crop" else 32)
    return args


def copy_frames(src: Path, dst: Path, threads: int = 32, every: int = 10_000) -> None:
    """Copy the frame dataset to local disk in parallel; data.yaml goes last as completion marker."""
    if (dst / "data.yaml").exists():
        print(f"{dst} already copied, skipping", flush=True)
        return
    t0 = time.time()
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.relative_to(src) != Path("data.yaml"))
    print(f"copying {len(files)} files {src} -> {dst}", flush=True)

    def one(p: Path) -> None:
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out)

    with ThreadPoolExecutor(threads) as pool:
        for i, _ in enumerate(pool.map(one, files), 1):
            if i % every == 0:
                print(f"  {i}/{len(files)} files, {time.time() - t0:.0f} s", flush=True)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "data.yaml", dst / "data.yaml")
    print(f"copy done: {len(files)} files, {time.time() - t0:.0f} s", flush=True)


def prepare_data(args) -> tuple[str, Path]:
    """Return (data.yaml path, dataset folder holding the indices); rebuilt on every run, resumes included."""
    from kaggle_paths import find_dataset, write_data_yaml  # same folder as this script

    if args.data:
        return args.data, Path(args.data).parent
    if args.setting == "crop":
        cache = find_dataset("dronepose-cache")
        return str(write_data_yaml(cache, "/kaggle/working")), cache / "yolo_crops"
    src = find_dataset("dronepose-frames") / "yolo_frames"
    copy_frames(src, FRAMES_LOCAL)
    return str(write_data_yaml(FRAMES_LOCAL.parent, "/kaggle/working", sub=FRAMES_LOCAL.name, name="data_frames.yaml")), src


def read_commit() -> str:
    f = CODE / "COMMIT"
    return f.read_text().strip() if f.exists() else "unknown"


def versions_and_gpu() -> tuple[dict, str | None]:
    import torch
    import ultralytics

    v = {"python": sys.version.split()[0], "torch": torch.__version__, "ultralytics": ultralytics.__version__}
    try:
        import albumentations
        v["albumentations"] = albumentations.__version__
    except ImportError:
        pass
    return v, (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def finish_info(run_dir: Path, info: dict) -> None:
    """Add epochs_done and train_hours from results.csv and rewrite run_info.json."""
    from dronepose.common.registry import registry_row

    row = registry_row(run_dir) if (run_dir / "results.csv").exists() else {}
    info.update({k: row[k] for k in ("epochs_done", "train_hours") if k in row})
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    args = parse_args()
    t0 = time.time()
    print(f"[{0:7.1f}s] setting={args.setting} recipe={args.recipe} model={args.model} code={CODE}", flush=True)
    from dronepose.common.registry import data_fingerprint, make_run_info
    from dronepose.keypoints.recipes import RECIPES, albumentations_for
    from ultralytics import YOLO

    data, data_dir = prepare_data(args)
    print(f"[{time.time() - t0:7.1f}s] data.yaml: {data}", flush=True)
    versions, gpu = versions_and_gpu()
    data_hash = data_fingerprint(data_dir)

    if args.resume_from:
        model = YOLO(args.resume_from)
        ckpt_args = getattr(model, "ckpt", {}).get("train_args", {}) or {}
        run_dir = Path(ckpt_args.get("save_dir") or Path(args.resume_from).parent.parent)
        old = next((p for p in (Path(args.resume_from).parent.parent / "run_info.json", run_dir / "run_info.json") if p.exists()), None)
        info = json.loads(old.read_text()) if old else make_run_info(
            {**vars(args), "name": ckpt_args.get("name", args.name)}, read_commit(), data_hash, versions, gpu)
        info.setdefault("resumed", []).append(time.strftime("%Y-%m-%d"))
        train = lambda: model.train(resume=True)  # noqa: E731
    else:
        run_dir = Path(args.project) / args.name
        info = make_run_info(vars(args), read_commit(), data_hash, versions, gpu)
        kw = dict(RECIPES[args.recipe])
        aug = albumentations_for(args.recipe)
        if aug is not None:
            kw["augmentations"] = aug
        budget = dict(time=args.hours, epochs=300, patience=1000) if args.hours else dict(epochs=args.epochs or 10)
        model = YOLO(args.model)
        train = lambda: model.train(  # noqa: E731
            data=data, imgsz=args.imgsz, batch=args.batch, workers=args.workers, save_period=args.save_period,
            project=args.project, name=args.name, exist_ok=True, seed=args.seed,
            **({"cache": "ram"} if args.setting == "crop" else {}), **budget, **kw)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"[{time.time() - t0:7.1f}s] run_info.json -> {run_dir} (commit {info['commit']})", flush=True)
    try:
        train()
    finally:
        finish_info(run_dir, info)
        print(f"[{time.time() - t0:7.1f}s] run_info.json updated", flush=True)


if __name__ == "__main__":
    main()
