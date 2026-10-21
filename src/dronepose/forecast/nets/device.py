"""Device choice, seeding and the network settings (configs/nets.json)."""
import json
import random
from pathlib import Path

import numpy as np
import torch


def pick_device(prefer: str = "auto") -> torch.device:
    """"auto": MPS when available, else CPU; "cpu" / "mps" forced."""
    if prefer == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(prefer)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_nets_config(path: Path = Path("configs/nets.json")) -> dict:
    return json.loads(Path(path).read_text())
