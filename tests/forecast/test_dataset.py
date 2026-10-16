import copy
from pathlib import Path

import numpy as np
import pytest

from dronepose.data.drone_models import load_drone_models
from dronepose.forecast.config import load_config, metric_scales
from dronepose.forecast.dataset import build_trajectories, load_trajectories, save_trajectories
from dronepose.forecast.frames import to_gravity_positions, to_gravity_rotations

CFG = load_config()
SCALES = metric_scales(load_drone_models(Path("configs/drone_models.json")), CFG.dji_diagonal_mm)


def _records(scene, drone, backgrounds, T, split="val", degree=180):
    out = []
    for bg in backgrounds:
        for i in range(1, T + 1):
            out.append({"scene": scene, "drone": drone, "background": bg, "degree": degree, "frame_idx": i,
                        "split": split, "t_gt": [0.01 * i, 0.3 - 0.002 * i, 2.0 + 0.005 * i],
                        "R_gt": np.eye(3).tolist()})
    return out


def test_copies_are_deduplicated():
    trajs = build_trajectories(_records("03", "01", ["05", "06", "07"], T=50), CFG, SCALES)
    assert len(trajs) == 1
    assert [c[0] for c in trajs[0].copies] == ["05", "06", "07"]
    assert len(trajs[0].times) == 50
    np.testing.assert_allclose(trajs[0].times[:2], [0, 1 / 30])


def test_differing_copies_raise():
    recs = _records("03", "01", ["05", "06"], T=50)
    recs[-1]["t_gt"][0] += 1e-9
    with pytest.raises(ValueError, match="copies differ"):
        build_trajectories(recs, CFG, SCALES)


def test_missing_frame_raises():
    recs = [r for r in _records("03", "01", ["05"], T=50) if r["frame_idx"] != 20]
    with pytest.raises(ValueError, match="frames"):
        build_trajectories(recs, CFG, SCALES)


def test_gravity_frame_and_metric_units():
    recs = _records("03", "01", ["05"], T=50)
    tr = build_trajectories(recs, CFG, SCALES)[0]
    np.testing.assert_allclose(tr.p_G, to_gravity_positions(np.array([r["t_gt"] for r in recs])))
    np.testing.assert_allclose(tr.R_G, to_gravity_rotations(np.array([r["R_gt"] for r in recs])))
    np.testing.assert_allclose(tr.u, np.tile([0, 1, 0], (50, 1)))  # identity body->camera: body up = camera z = G forward
    assert tr.metric and tr.s_c == SCALES["01"]
    np.testing.assert_allclose(tr.positions, SCALES["01"] * tr.p_G)
    assert tr.motion_type == "linear" and tr.split == "val" and tr.key == "03_01"


def test_drone_08_stays_in_dataset_units():
    tr = build_trajectories(_records("03", "08", ["05"], T=50), CFG, SCALES)[0]
    assert not tr.metric and tr.s_c is None
    np.testing.assert_allclose(tr.positions, tr.p_G)


def test_trajectories_sorted_by_scene_and_drone():
    recs = _records("14", "02", ["01"], T=50) + _records("03", "08", ["05"], T=50) + _records("03", "01", ["05"], T=50)
    assert [t.key for t in build_trajectories(recs, CFG, SCALES)] == ["03_01", "03_08", "14_02"]


def test_save_load_roundtrip(tmp_path):
    recs = _records("03", "01", ["05", "06"], T=50) + _records("03", "08", ["05"], T=60)
    trajs = build_trajectories(recs, CFG, SCALES)
    save_trajectories(trajs, tmp_path, "val")
    back = load_trajectories(tmp_path, "val")
    assert len(back) == 2
    for a, b in zip(trajs, back):
        assert (a.scene, a.drone, a.split, a.motion_type, a.copies, a.s_c) == (b.scene, b.drone, b.split, b.motion_type, b.copies, b.s_c)
        np.testing.assert_array_equal(a.times, b.times)
        np.testing.assert_array_equal(a.p_G, b.p_G)
        np.testing.assert_array_equal(a.R_G, b.R_G)
