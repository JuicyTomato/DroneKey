"""Build configs/drone_models.json from the real dataset (every 3rd frame is plenty)."""
import itertools
import os
from pathlib import Path

from dronepose.data.drone_models import build_drone_models, save_drone_models
from dronepose.data.sixdronesyn import iter_frames

root = Path(os.environ["SIXDRONESYN_ROOT"])
models = build_drone_models(itertools.islice(iter_frames(root), 0, None, 3))
save_drone_models(models, Path("configs/drone_models.json"))
for d, m in models.items():
    print(d, f"spread {1000 * m.spread:.3f} mm", m.kpts_flu.round(3).tolist())
