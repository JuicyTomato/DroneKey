import pytest

from dronepose.eval.bootstrap import by_trajectory, paired_bootstrap, trajectory_of


def test_trajectory_of():
    assert trajectory_of("14_03_10_000_0005") == "14_03"


def test_by_trajectory_skips_none():
    res = [{"frame_id": "14_03_10_000_0005", "k": 1.0}, {"frame_id": "14_03_10_000_0006", "k": None},
           {"frame_id": "14_04_10_000_0001", "k": 2.0}]
    assert by_trajectory(res, "k") == {"14_03": [1.0], "14_04": [2.0]}


def test_identical_groups_give_zero():
    g = {"a": [1.0, 2.0], "b": [3.0]}
    out = paired_bootstrap({"x": g, "y": g}, ref="x", n_boot=200)
    assert out["y"] == (0.0, 0.0, 0.0)


def test_constant_offset_is_exact():
    x = {"a": [1.0, 2.0, 5.0], "b": [3.0, 4.0]}
    y = {k: [v + 1.0 for v in vs] for k, vs in x.items()}
    d, lo, hi = paired_bootstrap({"x": x, "y": y}, ref="x", n_boot=200)["y"]
    assert d == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_deterministic_with_seed():
    x = {"a": [1.0, 9.0], "b": [2.0], "c": [7.0, 3.0]}
    y = {"a": [2.0, 5.0], "b": [8.0], "c": [1.0, 1.0]}
    assert paired_bootstrap({"x": x, "y": y}, "x", 300, 5) == paired_bootstrap({"x": x, "y": y}, "x", 300, 5)


def test_trajectories_are_intersection():
    x = {"a": [1.0], "b": [2.0]}
    y = {"a": [3.0], "z": [9.0]}
    d, _, _ = paired_bootstrap({"x": x, "y": y}, "x", 50)["y"]
    assert d == pytest.approx(2.0)
