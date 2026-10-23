"""Run registry: run_info.json written by the training script, one JSONL row per run."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

_MAP = "metrics/mAP50-95(P)"


def data_fingerprint(dataset_dir: Path) -> str:
    """SHA-256 of data.yaml plus the sorted crops_*/frames_* indices; images are never read."""
    d = Path(dataset_dir)
    files = [d / "data.yaml"] + sorted([*d.glob("crops_*.jsonl"), *d.glob("frames_*.jsonl")], key=lambda p: p.name)
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def make_run_info(args: dict, commit: str, data_hash: str, versions: dict[str, str], gpu: str | None) -> dict:
    """Metadata of a training run (args must carry name, seed, recipe, setting)."""
    return {"name": args["name"], "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": commit, "data_hash": data_hash, "versions": versions, "gpu": gpu, "args": args,
            "seed": args["seed"], "recipe": args["recipe"], "setting": args["setting"]}


def _read_results(run_dir: Path) -> list[dict]:
    path = run_dir / "results.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]


def registry_row(run_dir: Path) -> dict:
    """Registry row of a run folder: run_info.json if present, else rebuilt from args.yaml (ricostruito)."""
    run_dir = Path(run_dir)
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        row = json.loads(info_path.read_text())
    else:
        args = yaml.safe_load((run_dir / "args.yaml").read_text())
        row = {"name": args.get("name", run_dir.name), "seed": args.get("seed"),
               "setting": "frame" if args.get("imgsz") == 640 else "crop", "args": args, "ricostruito": True}
    rows = _read_results(run_dir)
    row["epochs_done"] = len(rows)
    if rows:
        best = max(rows, key=lambda r: float(r[_MAP]))
        row["best_epoch"] = int(float(best["epoch"]))
        row["train_hours"] = float(rows[-1]["time"]) / 3600
    elif (run_dir / "history.json").exists():  # DroneKey runs: one row per epoch, no results.csv
        hist = json.loads((run_dir / "history.json").read_text())
        row["epochs_done"] = len(hist)
        if hist:
            row["best_epoch"] = min(range(len(hist)), key=lambda i: hist[i]["val_px_median"]) + 1
            row["train_hours"] = sum(r["seconds"] for r in hist) / 3600
    return row


def update_registry(registry: Path, rows: list[dict]) -> None:
    """Merge rows into the JSONL registry: one row per name, replaced if already present."""
    registry = Path(registry)
    old = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()] if registry.exists() else []
    merged = {r["name"]: r for r in old}
    merged.update({r["name"]: r for r in rows})
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("".join(json.dumps(r) + "\n" for r in merged.values()))
