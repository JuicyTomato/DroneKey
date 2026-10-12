"""Assemble data/kaggle_cache/ (code + canonical + models + crops + skeleton frames + manifest) and zip it. Must be < 20 GB."""
import shutil
from pathlib import Path

LIMIT = 20 * 1024 ** 3
cache = Path("data/kaggle_cache")
if cache.exists():
    shutil.rmtree(cache)
(cache / "code").mkdir(parents=True)
for item in ("src", "scripts", "configs", "pyproject.toml", "requirements-lock.txt"):
    src = Path(item)
    (shutil.copytree if src.is_dir() else shutil.copy2)(src, cache / "code" / item)
for item in ("canonical_train.jsonl", "canonical_val.jsonl", "canonical_test.jsonl", "yolo_crops", "frames"):
    src = Path("data/processed") / item
    (shutil.copytree if src.is_dir() else shutil.copy2)(src, cache / item)
if Path("data/manifest").exists():
    shutil.copytree("data/manifest", cache / "manifest")
size = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
print(f"cache: {size / 1024 ** 3:.2f} GB")
assert size < LIMIT, "cache oltre il limite di 20 GB del dataset privato Kaggle"
shutil.make_archive("data/dronepose-cache", "zip", cache)
print("zip: data/dronepose-cache.zip")
