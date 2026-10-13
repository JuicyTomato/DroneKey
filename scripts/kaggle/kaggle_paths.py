"""Locate the Kaggle input datasets wherever Kaggle mounts them, and write a data.yaml with the real path."""
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


def write_data_yaml(cache_dir: Path, out_dir: Path) -> Path:
    """Copy the cache's data.yaml to a writable folder with `path` set to where the crops really are."""
    crops = Path(cache_dir) / "yolo_crops"
    cfg = yaml.safe_load((crops / "data.yaml").read_text())
    cfg["path"] = str(crops)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "data.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out
