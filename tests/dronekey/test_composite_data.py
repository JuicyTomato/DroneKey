import importlib.util
from pathlib import Path

import numpy as np
import torch

from dronepose.keypoints.dronekey.config import load_config
from dronepose.keypoints.dronekey.data import CropDataset
from tests.dronekey.test_data import _make

_spec = importlib.util.spec_from_file_location("train_dronekey", Path("scripts/kaggle/train_dronekey.py"))
td = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(td)
INPUT = load_config()["input"]


def test_no_composer_keeps_old_behaviour(tmp_path):
    _make(tmp_path)
    a = CropDataset(tmp_path, "train", "none", True, INPUT, seed=0)[0][0]
    b = CropDataset(tmp_path, "train", "none", True, INPUT, seed=0, composer=None)[0][0]
    assert torch.equal(a, b)


def test_composer_runs_before_recipe_and_only_in_train(tmp_path):
    _make(tmp_path, "train")
    _make(tmp_path, "val")
    calls = []

    def fake(img, fid, rng):
        calls.append(fid)
        return np.zeros_like(img)

    train = CropDataset(tmp_path, "train", "none", True, INPUT, seed=0, composer=fake)
    val = CropDataset(tmp_path, "val", "none", False, INPUT, composer=fake)
    expected = CropDataset(tmp_path, "train", "none", True, INPUT, seed=0)
    assert not torch.equal(train[0][0], expected[0][0]) and len(calls) == 1
    val[0]
    assert len(calls) == 1


def test_composer_rng_changes_with_epoch(tmp_path):
    _make(tmp_path)
    seen = []

    def spy(img, fid, rng):
        seen.append(rng.random())
        return img

    ds = CropDataset(tmp_path, "train", "none", True, INPUT, seed=0, composer=spy)
    ds[0]
    ds.set_epoch(1)
    ds[0]
    assert seen[0] != seen[1]


def test_train_dronekey_accepts_new_recipes():
    a = td.parse_args(["--dataset", "6dronesyn", "--variant", "paper", "--recipe", "c90b20", "--name", "x",
                       "--masks-dir", "m", "--bg-dir", "b"])
    assert a.recipe == "c90b20" and a.masks_dir == "m" and a.bg_dir == "b"


def test_composition_recipes_need_albumentations(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_alb(name, *a, **k):
        if name == "albumentations":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_alb)
    import pytest
    with pytest.raises(SystemExit):
        td.require_albumentations("cp50")
    td.require_albumentations("none")
