"""Train DroneKey (keypoint regression on the 256 px crops). Resumable: --resume-from <last.pt>.

Kaggle:  %%bash  (see docs/kaggle/README.md for the CODE=... lookup)
         python -u $CODE/scripts/kaggle/train_dronekey.py --dataset 6dronesyn --variant paper --seed 0 --name dk_paper_s0
Datasets are located under /kaggle/input (dronepose-cache/yolo_crops, dronekey-2dk) unless --data-root is given.
Writes <project>/<name>/{run_info.json, history.json, last.pt, best.pt}; run_info.json (commit from $CODE/COMMIT,
data hash, versions, GPU) is written before training and updated at the end.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]            # folder that holds src/ and scripts/
sys.path.insert(0, str(CODE / "src"))                  # dronepose without pip install
sys.path.insert(0, str(CODE / "scripts" / "kaggle"))   # kaggle_paths


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["6dronesyn", "2dronekey"], required=True)
    ap.add_argument("--variant", required=True, help="variant of configs/dronekey.json (paper, code, G0..G6)")
    ap.add_argument("--epochs", type=int, default=None, help="default: train.epochs (paper/code) or ladder_epochs (G*)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recipe", choices=["w1", "final", "none"], default=None, help="default: the variant's recipe")
    ap.add_argument("--name", default=None, help="run name (required unless --resume-from; on resume it must match the checkpoint's)")
    ap.add_argument("--project", default="/kaggle/working/runs")
    ap.add_argument("--resume-from", default=None, help="last.pt of an interrupted run, also read-only (e.g. a Kaggle dataset; best.pt, run_info.json, history.json next to it are carried over). "
                    "Output always goes to <project>/<name>, name taken from the checkpoint. Same args as the original, except --workers, --max-hours, --device")
    ap.add_argument("--max-hours", type=float, default=None, help="per-session budget: stop after the epoch that exceeds it, counted from this launch (resumable)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N crops of train and val")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=None, help="default: train.workers")
    ap.add_argument("--batch", type=int, default=None, help="default: train.batch")
    ap.add_argument("--lr", type=float, default=None, help="default: train.lr")
    ap.add_argument("--cache-ram", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="random ResNet50 init (smoke tests)")
    ap.add_argument("--data-root", default=None, help="crop folder (outside Kaggle)")
    args = ap.parse_args(argv)
    if not args.name and not args.resume_from:
        ap.error("--name is required unless --resume-from is given")
    return args


def data_root(args) -> Path:
    if args.data_root:
        return Path(args.data_root)
    from kaggle_paths import find_dataset

    if args.dataset == "6dronesyn":
        return find_dataset("dronepose-cache") / "yolo_crops"
    return find_dataset("dronekey-2dk")


def build_run_cfg(args, name: str) -> dict:
    from dronepose.keypoints.dronekey.config import load_config, resolve_variant

    cfg = load_config()
    run_cfg = resolve_variant(cfg, args.variant)
    tc = run_cfg["train"]
    tc["epochs"] = args.epochs or (cfg["ladder_epochs"] if args.variant.startswith("G") else tc["epochs"])
    for k in ("lr", "batch", "workers"):
        if getattr(args, k) is not None:
            tc[k] = getattr(args, k)
    if args.recipe:
        run_cfg["recipe"] = args.recipe
    run_cfg.update(seed=args.seed, name=name, variant=args.variant, dataset=args.dataset)
    return run_cfg


def require_albumentations(recipe: str) -> None:
    """w1/final include Albumentations filters: refuse to train without them (the recipe would silently differ from YOLO26s's)."""
    if recipe in ("w1", "final"):
        try:
            import albumentations  # noqa: F401
        except ImportError:
            sys.exit(f"albumentations mancante: la ricetta {recipe} non sarebbe quella di YOLO26s; "
                     "installarlo (pip install albumentations) o usare --recipe none")


def finish_info(run_dir: Path, info: dict) -> None:
    """Add epochs_done and train_hours from history.json and rewrite run_info.json."""
    from dronepose.common.registry import registry_row

    row = registry_row(run_dir) if (run_dir / "history.json").exists() else {}
    info.update({k: row[k] for k in ("epochs_done", "train_hours") if k in row})
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))


def copy_sidecars(src: Path, dst: Path) -> None:
    """Resume from a read-only folder (Kaggle input): carry best.pt / run_info.json / history.json to the output folder.
    last.pt is not copied (fit reads it from resume_from); files already in dst are kept (they are newer)."""
    if src.resolve() == dst.resolve():
        return
    for fname in ("best.pt", "run_info.json", "history.json"):
        if (src / fname).exists() and not (dst / fname).exists():
            shutil.copyfile(src / fname, dst / fname)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    import torch

    from dronepose.common.registry import data_fingerprint, make_run_info
    from dronepose.common.runinfo import read_commit, versions_and_gpu
    from dronepose.keypoints.dronekey import train as T
    from dronepose.keypoints.dronekey.data import CropDataset
    from dronepose.keypoints.dronekey.model import build_model

    resume = Path(args.resume_from) if args.resume_from else None
    name = args.name
    if resume:  # the checkpoint (possibly on a read-only Kaggle input) is the source of truth for the run name
        ck_cfg = torch.load(resume, map_location="cpu", weights_only=False, mmap=True)
        ck_name, ck_hash = ck_cfg["run_cfg"]["name"], ck_cfg["config_hash"]
        del ck_cfg
        if args.name and args.name != ck_name:
            sys.exit(f"--name {args.name!r} differs from the checkpoint's run name {ck_name!r}: omit --name or use the same one")
        name = ck_name
    run_dir = Path(args.project) / name  # always writable, also when resuming from /kaggle/input
    run_cfg = build_run_cfg(args, name)
    if resume and T.config_hash(run_cfg) != ck_hash:  # reject before touching the output folder
        sys.exit(f"{resume}: run config differs from the checkpoint's (config hash mismatch): repeat the original "
                 "--dataset/--variant/--seed/--epochs/--recipe/--lr/--batch flags")
    require_albumentations(run_cfg["recipe"])
    root = data_root(args)
    device = T.pick_device(args.device)
    print(f"[{0:7.1f}s] dataset={args.dataset} variant={args.variant} recipe={run_cfg['recipe']} "
          f"epochs={run_cfg['train']['epochs']} device={device} data={root} code={CODE}", flush=True)
    versions, gpu = versions_and_gpu()
    run_args = {**vars(args), "name": name, "setting": "crop", "recipe": run_cfg["recipe"], "run_cfg": run_cfg}
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        copy_sidecars(resume.parent, run_dir)
    old = run_dir / "run_info.json"
    if resume and old.exists():
        info = json.loads(old.read_text())
    else:
        info = make_run_info(run_args, read_commit(CODE), data_fingerprint(root), versions, gpu)
    if resume:
        info.setdefault("resumed", []).append(time.strftime("%Y-%m-%d"))
    old.write_text(json.dumps(info, indent=2))
    print(f"[{time.time() - t0:7.1f}s] run_info.json -> {run_dir} (commit {info['commit']})", flush=True)

    T.seed_everything(args.seed)  # head init reproducible; fit re-seeds fresh runs
    inp = run_cfg["input"]
    train_ds = CropDataset(root, "train", run_cfg["recipe"], True, inp, seed=args.seed, limit=args.limit, cache_ram=args.cache_ram)
    val_ds = CropDataset(root, "val", run_cfg["recipe"], False, inp, limit=args.limit, cache_ram=args.cache_ram)
    print(f"[{time.time() - t0:7.1f}s] train {len(train_ds)} crops, val {len(val_ds)} crops", flush=True)
    model = build_model(run_cfg["model"], pretrained=not args.no_pretrained)
    try:
        T.fit(run_cfg, model, train_ds, val_ds, run_dir, device, resume_from=resume, max_hours=args.max_hours)
    finally:
        try:
            finish_info(run_dir, info)
            print(f"[{time.time() - t0:7.1f}s] run_info.json updated", flush=True)
        except Exception as exc:  # never mask the exception raised by fit
            print(f"warning: run_info.json not updated: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
