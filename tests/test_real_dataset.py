"""Checks on the real 6DroneSyn. Run with SIXDRONESYN_ROOT set; skipped otherwise."""
import itertools
from pathlib import Path

import numpy as np
import pytest

from dronepose.data.drone_models import build_drone_models, load_drone_models
from dronepose.data.sixdronesyn import DRONE_IDS, iter_frames

MODELS_PATH = Path("configs/drone_models.json")


@pytest.mark.dataset
def test_real_models_rigid_below_0_1_mm(dataset_root):
    models = build_drone_models(itertools.islice(iter_frames(dataset_root), 0, None, 50))
    assert sorted(models) == list(DRONE_IDS)
    assert all(m.spread < 1e-4 for m in models.values())


def test_committed_models_match_paper_table():
    m = load_drone_models(MODELS_PATH)
    # PROJECT_MEMORY §3.5, Mini 3 Pro front width 0.306, Tello diagonal 0.14 (camera units)
    assert np.linalg.norm(m["01"].kpts_flu[0] - m["01"].kpts_flu[1]) == pytest.approx(0.306, abs=0.002)
    assert np.linalg.norm(m["08"].kpts_flu[0] - m["08"].kpts_flu[2]) == pytest.approx(0.137, abs=0.003)


from dronepose.data.canonical import read_jsonl

PROCESSED = Path("data/processed")


@pytest.mark.dataset
def test_real_canonical_counts():
    n = {s: len(read_jsonl(PROCESSED / f"canonical_{s}.jsonl")) for s in ("train", "val", "test")}
    assert n == {"train": 35160, "val": 12600, "test": 5040}


@pytest.mark.dataset
def test_real_canonical_gt_matches_corrected_euler(dataset_root):
    from dronepose.common.geometry import BLENDER_BODY_TO_FLU, geodesic_deg
    from dronepose.data.sixdronesyn import corrected_rotation
    frames = list(itertools.islice(iter_frames(dataset_root), 0, None, 500))
    recs = {r["frame_id"]: r for s in ("train", "val", "test") for r in read_jsonl(PROCESSED / f"canonical_{s}.jsonl")}
    checked = 0
    for f in frames:
        if f.frame_id in recs:
            R_euler = corrected_rotation(f) @ BLENDER_BODY_TO_FLU.T
            assert geodesic_deg(np.array(recs[f.frame_id]["R_gt"]), R_euler) < 0.05
            checked += 1
    assert checked >= 50, f"only {checked} frames matched: frame_id drift would make this test vacuous"
