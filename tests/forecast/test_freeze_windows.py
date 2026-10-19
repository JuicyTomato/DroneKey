import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.forecast.dataset import Trajectory, load_trajectories, save_trajectories
from dronepose.forecast.windows import freeze_windows, make_windows


def tr(T=60, freeze_idx=29, meta=None, offset=0.0, scene="gen", drone="01"):
    t = np.arange(T) / 30
    p = np.stack([offset + t, 2 * t, 0.1 * t ** 2], axis=1)
    R = Rotation.from_euler("z", 10 * t[:, None], degrees=True).as_matrix()
    return Trajectory(scene=scene, drone=drone, split="val", motion_type="normale", copies=[], s_c=1.0,
                      times=t, p_G=p, R_G=R, freeze_idx=freeze_idx, meta=dict(meta or {}))


def test_old_index_files_still_load(tmp_path):
    save_trajectories([tr(freeze_idx=None)], tmp_path, "val")
    index = json.loads((tmp_path / "index_val.json").read_text())
    for e in index:
        e.pop("freeze_idx", None)
        e.pop("meta", None)
    (tmp_path / "index_val.json").write_text(json.dumps(index))
    back = load_trajectories(tmp_path, "val")
    assert back[0].freeze_idx is None and back[0].meta == {}


def test_freeze_and_meta_roundtrip(tmp_path):
    meta = {"mode": "acro", "s": 0.25, "flip": True, "fz_v": [1.0, -2.5, 0.125], "level": 1}
    save_trajectories([tr(meta=meta)], tmp_path, "val")
    back = load_trajectories(tmp_path, "val")[0]
    assert back.freeze_idx == 29
    assert back.meta == meta


def test_freeze_windows_one_per_trajectory():
    w = freeze_windows([tr(), tr(offset=5.0), tr(offset=9.0)], 15, 30, 30)
    assert len(w) == 3
    assert list(w.start) == [29 - 15 + 1] * 3
    np.testing.assert_allclose(w.obs_p[:, -1], 0)
    np.testing.assert_allclose(w.last_p[:, 0], np.array([0.0, 5.0, 9.0]) + 29 / 30)
    assert w.fut_p.shape == (3, 30, 3)


def test_freeze_windows_requires_freeze_idx():
    with pytest.raises(ValueError):
        freeze_windows([tr(), tr(freeze_idx=None)], 15, 30, 30)


def test_freeze_windows_rejects_short_history():
    with pytest.raises(ValueError):
        freeze_windows([tr()], 31, 30, 30)
    with pytest.raises(ValueError):
        freeze_windows([tr()], 15, 31, 30)


def test_windows_meta_columns():
    w = freeze_windows([tr(meta={"mode": "acro", "s": 0.5, "fz_v": [1, 2, 3]}),
                        tr(meta={"mode": "angle", "s": 0.1, "fz_v": [4, 5, 6], "only_here": 1})], 15, 30, 30)
    assert list(w.meta["mode"]) == ["acro", "angle"]
    np.testing.assert_allclose(w.meta["s"], [0.5, 0.1])
    assert w.meta["fz_v"].shape == (2, 3)
    assert "only_here" not in w.meta


def test_make_windows_unchanged_on_6dronesyn_like():
    trajs = [tr(80, freeze_idx=None, scene="03"), tr(70, freeze_idx=None, scene="09", offset=3.0)]
    w = make_windows(trajs, 15, 30, 30)
    assert w.meta == {}
    assert len(w) == (80 - 44) + (70 - 44)
    np.testing.assert_allclose(w.obs_p[:, -1], 0)


def test_generated_samples_get_unique_keys(tmp_path):
    trajs = [tr(meta={"sample_id": i}, offset=float(i)) for i in range(3)]
    save_trajectories(trajs, tmp_path, "val")
    back = load_trajectories(tmp_path, "val")
    assert len({t.key for t in back}) == 3
    np.testing.assert_allclose([t.p_G[0, 0] for t in back], [0.0, 1.0, 2.0])


def test_mixing_weighted_and_unweighted_trajectories_is_an_error():
    a = tr(meta={"mode": "acro", "weight": 2.0})
    b = tr(meta={"mode": "angle"})
    with pytest.raises(ValueError, match="weight"):
        freeze_windows([a, b], 15, 30, 30)
