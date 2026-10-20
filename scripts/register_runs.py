"""Add or refresh training runs in the local registry (experiments/registry.jsonl, outside git).

    python scripts/register_runs.py experiments/results/runs/* [--registry experiments/registry.jsonl]
"""
import argparse
from pathlib import Path

from dronepose.common.registry import registry_row, update_registry

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", type=Path, help="run folders (with run_info.json or args.yaml)")
ap.add_argument("--registry", type=Path, default=Path("experiments/registry.jsonl"))
args = ap.parse_args()

rows = [registry_row(r) for r in args.runs if r.is_dir()]
update_registry(args.registry, rows)
for r in rows:
    print(f"{r['name']}: epochs {r['epochs_done']}" + (" (ricostruito)" if r.get("ricostruito") else ""))
print("registry:", args.registry)
