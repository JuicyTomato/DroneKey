"""Zip data/processed/dronekey_2dk -> data/dronekey-2dk.zip (folder content at the zip root, like dronepose-cache)."""
import shutil
from pathlib import Path

SRC = Path("data/processed/dronekey_2dk")
assert SRC.is_dir(), f"{SRC} not found: run scripts/dronekey/build_2dronekey.py first"
shutil.make_archive("data/dronekey-2dk", "zip", SRC)
print(f"zip: data/dronekey-2dk.zip ({Path('data/dronekey-2dk.zip').stat().st_size / 1024 ** 3:.2f} GB)")
