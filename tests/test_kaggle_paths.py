import importlib.util
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location("kaggle_paths", Path("scripts/kaggle/kaggle_paths.py"))
kp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kp)


@pytest.mark.parametrize("layout", ["{name}", "{user}/{name}", "datasets/{user}/{name}"])
def test_find_dataset_in_every_kaggle_layout(tmp_path, layout):
    d = tmp_path / layout.format(user="someone", name="dronepose-cache")
    d.mkdir(parents=True)
    assert kp.find_dataset("dronepose-cache", root=tmp_path) == d


def test_find_dataset_missing_lists_what_exists(tmp_path):
    (tmp_path / "other").mkdir()
    with pytest.raises(FileNotFoundError, match="dronepose-cache"):
        kp.find_dataset("dronepose-cache", root=tmp_path)


def test_write_data_yaml_points_to_real_location(tmp_path):
    cache = tmp_path / "datasets" / "u" / "dronepose-cache"
    (cache / "yolo_crops").mkdir(parents=True)
    (cache / "yolo_crops" / "data.yaml").write_text(
        yaml.safe_dump({"path": "/wrong/place", "train": "images/train", "kpt_shape": [4, 3]}))
    out = kp.write_data_yaml(cache, tmp_path / "work")
    cfg = yaml.safe_load(out.read_text())
    assert cfg["path"] == str(cache / "yolo_crops") and cfg["train"] == "images/train" and cfg["kpt_shape"] == [4, 3]
