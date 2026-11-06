import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ultralytics")
from ultralytics.data import YOLODataset  # noqa: E402
from ultralytics.models.yolo.pose import PoseTrainer  # noqa: E402

from dronepose.keypoints.yolo_composite import CompositeDataset, make_trainer, train_kwargs  # noqa: E402

sys.path.insert(0, "scripts/kaggle")
_spec = importlib.util.spec_from_file_location("train_yolo_pose", Path("scripts/kaggle/train_yolo_pose.py"))
ty = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ty)


def _stub_dataset():
    ds = CompositeDataset.__new__(CompositeDataset)
    def in_place(img, fid, rng):
        img += 1
        return img

    ds.im_files, ds.composer = ["/x/01_05_13_000_0100.jpg"], in_place
    return ds


def test_load_image_does_not_touch_cached_array(monkeypatch):
    cached = np.zeros((4, 4, 3), np.uint8)
    monkeypatch.setattr(YOLODataset, "load_image", lambda self, i, *a, **k: (cached, (4, 4), (4, 4)), raising=False)
    ds = _stub_dataset()
    assert ds.load_image(0)[0].max() == 1 and cached.max() == 0


def test_composer_gets_the_frame_id(monkeypatch):
    seen = []
    monkeypatch.setattr(YOLODataset, "load_image", lambda self, i, *a, **k: (np.zeros((4, 4, 3), np.uint8), (4, 4), (4, 4)),
                        raising=False)
    ds = _stub_dataset()
    ds.composer = lambda img, fid, rng: (seen.append(fid), img)[1]
    ds.load_image(0)
    assert seen == ["01_05_13_000_0100"]


def test_trainer_swaps_only_train_dataset(monkeypatch):
    monkeypatch.setattr(PoseTrainer, "build_dataset", lambda self, p, mode="train", batch=None: YOLODataset.__new__(YOLODataset))
    T = make_trainer(lambda *a: None)
    t = T.__new__(T)
    assert isinstance(t.build_dataset("p", "train"), CompositeDataset)
    assert not isinstance(t.build_dataset("p", "val"), CompositeDataset)


def test_resume_keeps_the_trainer():
    kw = train_kwargs(object())
    assert issubclass(kw["trainer"], PoseTrainer)
    assert train_kwargs(None) == {}


def test_cli_recipe_and_blur_scale():
    a = ty.parse_args(["--recipe", "c90b20", "--setting", "frame"])
    assert a.recipe == "c90b20" and a.blur_scale == pytest.approx(0.4)
    assert ty.parse_args(["--recipe", "cp50"]).blur_scale == 1.0


def test_fresh_run_defaults_to_final():
    assert ty.run_settings(ty.parse_args([]), None) == ("final", 1.0)


def test_resume_reads_recipe_and_blur_scale_from_run_info():
    info = {"recipe": "c50b10", "args": {"blur_scale": 0.7}}
    a = ty.parse_args(["--resume-from", "last.pt"])
    assert ty.run_settings(a, info) == ("c50b10", 0.7)


def test_resume_without_run_info_needs_explicit_recipe():
    with pytest.raises(SystemExit):
        ty.run_settings(ty.parse_args(["--resume-from", "last.pt"]), None)
    a = ty.parse_args(["--resume-from", "last.pt", "--recipe", "cp90"])
    assert ty.run_settings(a, None) == ("cp90", 1.0)


def test_resume_refuses_a_different_recipe():
    a = ty.parse_args(["--resume-from", "last.pt", "--recipe", "cp90"])
    with pytest.raises(SystemExit):
        ty.run_settings(a, {"recipe": "cp50", "args": {}})
