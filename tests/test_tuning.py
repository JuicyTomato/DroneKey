import pytest

from dronepose.pose.tuning import Counts, add_counts, counts_from_results, scene_of, select_best


def _res(err):
    return {"rot_err_deg": err}


def test_scene_of():
    assert scene_of("14_03_09_000_0001") == "14"


def test_counts_from_results_split_by_drone():
    by_seq = {"03_01_05_090": [_res(5.0), _res(170.0), _res(None)],        # drone 01: 1 flip, 2 posed, 3 frames
              "03_08_04_090": [_res(120.0), _res(3.0)]}                       # drone 08
    c = counts_from_results(by_seq)
    assert c["all"] == Counts(flips=2, posed=4, frames=5)
    assert c["no08"] == Counts(flips=1, posed=2, frames=3)


def test_add_counts():
    assert add_counts([Counts(1, 10, 12), Counts(0, 5, 6)]) == Counts(1, 15, 18)
    assert add_counts([]) == Counts(0, 0, 0)


def row(rc, zc, flips, posed, frames):
    return {"params": {"reject_cost": rc, "z_cap": zc}, "counts": Counts(flips, posed, frames)}


def test_select_best_prefers_low_flip_among_enough_coverage():
    rows = [row(3, 4, 1, 960, 1000),          # flip 0.10%, coverage 96%
            row(3, 6, 0, 940, 1000),          # flip 0, but coverage 94% < 95%
            row(6, 6, 0, 955, 1000)]          # flip 0, coverage 95.5%  <- best
    best, ok = select_best(rows, min_coverage=0.95)
    assert ok and best["params"] == {"reject_cost": 6, "z_cap": 6}


def test_select_best_ties_are_deterministic():
    rows = [row(6, 8, 0, 955, 1000), row(6, 6, 0, 955, 1000), row(3, 8, 0, 955, 1000)]
    best, _ = select_best(rows)
    assert best["params"] == {"reject_cost": 3, "z_cap": 8}      # same flip and coverage: lowest reject_cost, then z_cap
    best2, _ = select_best(list(reversed(rows)))
    assert best2["params"] == best["params"]


def test_select_best_when_nothing_reaches_the_coverage_floor():
    rows = [row(3, 4, 0, 900, 1000), row(6, 4, 5, 930, 1000)]
    best, ok = select_best(rows, min_coverage=0.95)
    assert not ok and best["params"] == {"reject_cost": 6, "z_cap": 4}   # falls back to the highest coverage


def test_select_best_rejects_empty_input():
    with pytest.raises(ValueError):
        select_best([])
