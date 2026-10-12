import numpy as np
import pytest

from dronepose.data.drone_models import build_drone_models, load_drone_models, save_drone_models
from dronepose.data.sixdronesyn import iter_frames
from tests.fixtures.fake_dataset import FLU_MODELS


def test_models_match_ground_truth_flu(fake_root):
    models = build_drone_models(iter_frames(fake_root))
    for drone, flu in FLU_MODELS.items():
        np.testing.assert_allclose(models[drone].kpts_flu, flu - flu.mean(0), atol=1e-9)
        np.testing.assert_allclose(models[drone].origin_offset_flu, -flu.mean(0), atol=1e-9)
        assert models[drone].spread < 1e-9


def test_front_pair_is_front_and_key1_is_right(fake_root):
    m = build_drone_models(iter_frames(fake_root))["01"].kpts_flu
    assert m[0, 0] > 0 and m[1, 0] > 0 and m[2, 0] < 0 and m[3, 0] < 0   # 1-2 in front (+x)
    assert m[0, 1] < 0 < m[1, 1]                                          # key1 right (-y), key2 left (+y)


def test_inconsistent_rotation_raises(fake_root):
    frames = list(iter_frames(fake_root))
    bad = frames[0]
    object.__setattr__(bad, "rot_cam_euler", bad.rot_cam_euler + np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="spread"):
        build_drone_models(frames)


def test_save_load_roundtrip(fake_root, tmp_path):
    models = build_drone_models(iter_frames(fake_root))
    save_drone_models(models, tmp_path / "m.json")
    loaded = load_drone_models(tmp_path / "m.json")
    for d in models:
        np.testing.assert_allclose(loaded[d].kpts_flu, models[d].kpts_flu)
        np.testing.assert_allclose(loaded[d].origin_offset_flu, models[d].origin_offset_flu)
