"""Build the two Kaggle datasets: data/dronepose-cache.zip (data, rarely changes, < 20 GB) and
data/dronepose-code.zip (code, a few hundred KB: re-upload after every code change).
`--code-only` rebuilds just the code zip (it always contains a COMMIT file: git HEAD, plus -dirty if src/scripts/configs differ)."""
import argparse
import shutil
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--code-only", action="store_true", help="skip the data cache, rebuild only data/dronepose-code.zip")
args = ap.parse_args()

LIMIT = 20 * 1024 ** 3
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "._*", ".DS_Store")

# --- data ---
if not args.code_only:
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
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"],
                       capture_output=True, text=True, check=True).stdout.strip()
(code / "COMMIT").write_text(head + ("-dirty" if dirty else "") + "\n")
print("COMMIT:", (code / "COMMIT").read_text().strip())
shutil.make_archive("data/dronepose-code", "zip", code)
print(f"zip: data/dronepose-code.zip ({Path('data/dronepose-code.zip').stat().st_size / 1024:.0f} KB)")
