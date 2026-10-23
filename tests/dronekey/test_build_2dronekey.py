import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from dronepose.data.crops import CropBox, crop_to_frame

_spec = importlib.util.spec_from_file_location("build_2dronekey", Path("scripts/dronekey/build_2dronekey.py"))
b2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b2)

# propeller index (category_id 1-4) -> pixel keypoint, per image
KPTS = {
    "00000": {1: (900, 500), 2: (1000, 480), 3: (980, 600), 4: (880, 590)},
    "00001": {1: (300, 200), 2: (420, 210), 3: (410, 330), 4: (290, 320)},
}


def _write_json(path: Path, kpts: dict, shuffle: bool = False, drop: tuple = ()) -> None:
    images, anns = [], []
    for i, (name, ks) in enumerate(kpts.items()):
        images.append({"id": i, "file_name": f"{name}.png", "width": 1920, "height": 1080, "6DoF": [0.0] * 7})
        items = sorted(ks.items(), reverse=shuffle)
        anns += [{"id": 10 * i + c, "image_id": i, "category_id": c, "keypoints": list(p)}
                 for c, p in items if (name, c) not in drop]
    path.write_text(json.dumps({"images": images, "annotations": anns}))


def _make_src(root: Path, splits=(("train", "train"), ("valid", "val"))) -> Path:
    src = root / "src"
    for folder, js in splits:
        (src / folder).mkdir(parents=True)
        for name in KPTS:
            img = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(str(src / folder / f"{name}.png"), img)
        _write_json(src / f"instances_{js}2017.json", KPTS, shuffle=True)
    return src


def test_load_annotations_orders_by_category(tmp_path):
    _write_json(tmp_path / "a.json", KPTS, shuffle=True)
    ann = b2.load_2dk_annotations(tmp_path / "a.json")
    assert set(ann) == {"00000", "00001"}
    for name, ks in KPTS.items():
        assert ann[name].shape == (4, 2)
        np.testing.assert_array_equal(ann[name], [ks[c] for c in (1, 2, 3, 4)])


def test_load_annotations_rejects_three_keypoints(tmp_path):
    _write_json(tmp_path / "a.json", KPTS, drop=(("00001", 3),))
    with pytest.raises(ValueError, match="00001"):
        b2.load_2dk_annotations(tmp_path / "a.json")


def test_build_writes_crops_labels_and_index(tmp_path):
    src, out = _make_src(tmp_path), tmp_path / "out"
    b2.build(src, out, splits=("train", "valid"))
    for fid, ks in KPTS.items():
        img = cv2.imread(str(out / "images" / "train" / f"{fid}.jpg"))
        assert img.shape == (256, 256, 3)
        vals = np.array((out / "labels" / "train" / f"{fid}.txt").read_text().split(), float)
        assert vals.size == 17 and vals[0] == 0
        kp = vals[5:].reshape(4, 3)
        assert (kp[:, 2] == 2).all()
        meta = {m["frame_id"]: m for m in map(json.loads, (out / "crops_train.jsonl").read_text().splitlines())}
        box = CropBox(*meta[fid]["box"])
        gt = {g["frame_id"]: g for g in map(json.loads, (out / "gt_train.jsonl").read_text().splitlines())}
        want = np.array([ks[c] for c in (1, 2, 3, 4)], float)
        np.testing.assert_allclose(gt[fid]["kpts2d"], want)
        np.testing.assert_allclose(crop_to_frame(kp[:, :2] * 256, box), want, atol=0.5)
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["kpt_shape"] == [4, 3] and cfg["flip_idx"] == [1, 0, 3, 2] and cfg["names"] == {0: "drone"}
    assert cfg["path"] and cfg["val"] == "images/val"


def test_build_maps_valid_to_val(tmp_path):
    src, out = _make_src(tmp_path), tmp_path / "out"
    b2.build(src, out, splits=("train", "valid"))
    assert sorted(p.stem for p in (out / "images" / "val").glob("*.jpg")) == ["00000", "00001"]
    assert (out / "crops_val.jsonl").is_file() and (out / "gt_val.jsonl").is_file()
    assert not (out / "images" / "valid").exists()


def test_build_limit(tmp_path):
    src, out = _make_src(tmp_path), tmp_path / "out"
    b2.build(src, out, limit=1, splits=("train", "valid"))
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 1
