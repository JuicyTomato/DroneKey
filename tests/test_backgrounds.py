import importlib.util
import zipfile
from pathlib import Path

import pytest


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bb = _load("scripts/aug/build_backgrounds.py", "build_backgrounds")
pa = _load("scripts/aug/package_aug.py", "package_aug")


def test_pick_images_stable_and_sized():
    p = [Path(f"{i}.jpg") for i in range(100)]
    assert bb.pick_images(p, 10, 0) == bb.pick_images(p[::-1], 10, 0) and len(bb.pick_images(p, 10, 0)) == 10
    assert bb.pick_images(p, 10, 0) != bb.pick_images(p, 10, 1)


def test_refuses_mav6d_source():
    with pytest.raises(ValueError, match="MAV6D"):
        bb.check_source(Path("/Volumes/MakeMePasta/MAV6D/frames640"))
    bb.check_source(Path("/Volumes/MakeMePasta/DroneKey_aug/coco/train2017"))


def test_zip_skips_appledouble_and_keeps_structure(tmp_path):
    (tmp_path / "s" / "crop").mkdir(parents=True)
    (tmp_path / "s" / "crop" / "a.png").write_bytes(b"x")
    (tmp_path / "s" / "crop" / "._a.png").write_bytes(b"y")
    out = pa.zip_folder(tmp_path / "s", tmp_path / "o.zip", ["crop"])
    assert zipfile.ZipFile(out).namelist() == ["crop/a.png"]
