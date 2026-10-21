import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("generate_trajectories",
                                               Path(__file__).parents[2] / "scripts/forecast/generate_trajectories.py")
gt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gt)


def test_nets_sets_present():
    assert gt.SEEDS["select"] == 5 and gt.SEEDS["train_big"] == 6
    assert gt.SIZES["select"] == 2000 and gt.SIZES["train_big"] == 50000


def test_seeds_distinct_and_existing_unchanged():
    assert len(set(gt.SEEDS.values())) == len(gt.SEEDS)
    assert {k: gt.SEEDS[k] for k in ("train", "val", "test", "active")} == {"train": 1, "val": 2, "test": 3, "active": 4}
    assert {k: gt.SIZES[k] for k in ("train", "val", "test", "active")} == {"train": 10000, "val": 2000, "test": 2000,
                                                                          "active": 2000}
