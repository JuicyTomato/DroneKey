"""Single source of DroneKey hyper-parameters and ablation variants (configs/dronekey.json)."""
import copy
import json
from pathlib import Path

DEFAULT_PATH = Path("configs/dronekey.json")
SECTIONS = ("model", "loss", "train", "input")


def load_config(path: Path = DEFAULT_PATH) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():  # allow running from outside the repo root
        path = Path(__file__).resolve().parents[4] / path
    return json.loads(path.read_text())


def resolve_variant(cfg: dict, name: str) -> dict:
    """Base sections with the variant's overrides merged in (new dict; KeyError for an unknown variant)."""
    variant = cfg["variants"][name]
    out = {k: copy.deepcopy(cfg[k]) for k in SECTIONS}
    out["recipe"] = variant.get("recipe", cfg["recipe"])
    for k in SECTIONS:
        out[k].update(copy.deepcopy(variant.get(k, {})))
    return out
