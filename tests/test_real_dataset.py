"""Checks on the real 6DroneSyn. Run with SIXDRONESYN_ROOT set; skipped otherwise."""
import itertools
from pathlib import Path

import numpy as np
import pytest

from dronepose.data.drone_models import build_drone_models, load_drone_models
from dronepose.data.sixdronesyn import DRONE_IDS, iter_frames

pytestmark = pytest.mark.dataset
MODELS_PATH = Path("configs/drone_models.json")


def test_real_models_rigid_below_0_1_mm(dataset_root):
    models = build_drone_models(itertools.islice(iter_frames(dataset_root), 0, None, 50))
    assert sorted(models) == list(DRONE_IDS)
    assert all(m.spread < 1e-4 for m in models.values())


def test_committed_models_match_paper_table():
    m = load_drone_models(MODELS_PATH)
    # PROJECT_MEMORY §3.5, Mini 3 Pro front width 0.306, Tello diagonal 0.14 (camera units)
    assert np.linalg.norm(m["01"].kpts_flu[0] - m["01"].kpts_flu[1]) == pytest.approx(0.306, abs=0.002)
    assert np.linalg.norm(m["08"].kpts_flu[0] - m["08"].kpts_flu[2]) == pytest.approx(0.137, abs=0.003)
