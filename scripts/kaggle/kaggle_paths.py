"""Locate the Kaggle input datasets wherever Kaggle mounts them, and write a data.yaml with the real path."""
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


def find_dataset(name: str, root: str | Path = "/kaggle/input") -> Path:
    root = Path(root)
    for pattern in (name, f"*/{name}", f"datasets/*/{name}"):
        for hit in sorted(root.glob(pattern)):
            if hit.is_dir():
                return hit
    have = sorted(p.name for p in root.glob("*")) + sorted(str(p.relative_to(root)) for p in root.glob("datasets/*/*"))
    raise FileNotFoundError(f"dataset '{name}' non trovato sotto {root}; presenti: {have}")


def copy_tree(src: Path, dst: Path, threads: int = 32) -> Path:
    """Parallel copy to local disk (the input mount is slow per file); `.done` goes last, a finished copy is skipped."""
    src, dst = Path(src), Path(dst)
    if (dst / ".done").exists():
        return dst

    def one(p: Path) -> None:
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(one, sorted(p for p in src.rglob("*") if p.is_file())))
    dst.mkdir(parents=True, exist_ok=True)
    (dst / ".done").touch()
    return dst


def resolve_aug_dirs(masks_dir, bg_dir, sub: str, root: str | Path = "/kaggle/input",
                     local: str | Path = "/tmp") -> tuple[Path, Path]:
    """Masks (`sub` = crop or frame) and backgrounds: explicit paths as they are, else the Kaggle datasets on local disk."""
    if masks_dir and bg_dir:
        return Path(masks_dir), Path(bg_dir)
    return (copy_tree(find_dataset("dronepose-masks", root) / sub, Path(local) / f"aug_masks_{sub}"),
            copy_tree(find_dataset("dronepose-bgs", root), Path(local) / "aug_bgs"))


def write_data_yaml(cache_dir: Path, out_dir: Path, sub: str = "yolo_crops", name: str = "data.yaml") -> Path:
    """Copy a dataset's data.yaml to a writable folder with `path` set to where the images really are."""
    images = Path(cache_dir) / sub
    cfg = yaml.safe_load((images / "data.yaml").read_text())
    cfg["path"] = str(images)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out
