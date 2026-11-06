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


def test_write_data_yaml_other_subfolder(tmp_path):
    frames = tmp_path / "dronepose-frames"
    (frames / "yolo_frames").mkdir(parents=True)
    (frames / "yolo_frames" / "data.yaml").write_text(yaml.safe_dump({"path": "/wrong", "val": "images/val"}))
    out = kp.write_data_yaml(frames, tmp_path / "work", sub="yolo_frames", name="data_frames.yaml")
    assert out.name == "data_frames.yaml"
    assert yaml.safe_load(out.read_text())["path"] == str(frames / "yolo_frames")


def test_copy_tree_marker_and_skip(tmp_path):
    (tmp_path / "s" / "sub").mkdir(parents=True)
    (tmp_path / "s" / "sub" / "a.png").write_bytes(b"x")
    out = kp.copy_tree(tmp_path / "s", tmp_path / "d")
    assert (out / ".done").exists() and (out / "sub" / "a.png").read_bytes() == b"x"
    (tmp_path / "s" / "b.png").write_bytes(b"y")
    kp.copy_tree(tmp_path / "s", tmp_path / "d")
    assert not (tmp_path / "d" / "b.png").exists()


def test_resolve_aug_dirs_explicit_paths_are_kept(tmp_path):
    assert kp.resolve_aug_dirs("m", "b", "crop") == (Path("m"), Path("b"))


def test_resolve_aug_dirs_from_kaggle(tmp_path):
    root = tmp_path / "in"
    (root / "dronepose-masks" / "crop").mkdir(parents=True)
    (root / "dronepose-masks" / "crop" / "x.png").write_bytes(b"1")
    (root / "dronepose-bgs").mkdir()
    (root / "dronepose-bgs" / "b.jpg").write_bytes(b"2")
    m, b = kp.resolve_aug_dirs(None, None, "crop", root=root, local=tmp_path / "local")
    assert (m / "x.png").exists() and (b / "b.jpg").exists() and str(m).startswith(str(tmp_path / "local"))


def test_crop_and_frame_masks_do_not_share_a_folder(tmp_path):
    root = tmp_path / "in"
    for sub, name in (("crop", "c.png"), ("frame", "f.png")):
        (root / "dronepose-masks" / sub).mkdir(parents=True)
        (root / "dronepose-masks" / sub / name).write_bytes(b"1")
    (root / "dronepose-bgs").mkdir()
    (root / "dronepose-bgs" / "b.jpg").write_bytes(b"2")
    kp.resolve_aug_dirs(None, None, "crop", root=root, local=tmp_path / "local")
    m, _ = kp.resolve_aug_dirs(None, None, "frame", root=root, local=tmp_path / "local")
    assert (m / "f.png").exists() and not (m / "c.png").exists()
