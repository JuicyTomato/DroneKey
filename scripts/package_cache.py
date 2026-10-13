"""Build the two Kaggle datasets: data/dronepose-cache.zip (data, rarely changes, < 20 GB) and
data/dronepose-code.zip (code, a few hundred KB: re-upload after every code change)."""
import shutil
from pathlib import Path

LIMIT = 20 * 1024 ** 3
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "._*", ".DS_Store")

# --- data ---
cache = Path("data/kaggle_cache")
if cache.exists():
    shutil.rmtree(cache)
cache.mkdir(parents=True)
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

# --- code ---
code = Path("data/kaggle_code")
if code.exists():
    shutil.rmtree(code)
code.mkdir(parents=True)
for item in ("src", "scripts", "configs", "pyproject.toml", "requirements-lock.txt"):
    src = Path(item)
    if src.is_dir():
        shutil.copytree(src, code / item, ignore=IGNORE)
    else:
        shutil.copy2(src, code / item)
shutil.make_archive("data/dronepose-code", "zip", code)
print(f"zip: data/dronepose-code.zip ({Path('data/dronepose-code.zip').stat().st_size / 1024:.0f} KB)")
