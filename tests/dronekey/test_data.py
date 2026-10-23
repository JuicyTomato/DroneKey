import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dronepose.keypoints.dronekey.config import load_config  # noqa: E402
from dronepose.keypoints.dronekey.data import CropDataset  # noqa: E402
from dronepose.keypoints.recipes import RECIPES  # noqa: E402

INPUT = load_config()["input"]
KP = [(0.5, 0.2), (0.8, 0.5), (0.5, 0.8), (0.97, 0.5)]  # last one close to the right edge


def _make(root, split="train", n=2):
    (root / "images" / split).mkdir(parents=True)
    (root / "labels" / split).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        fid = f"f_{i:04d}"
        cv2.imwrite(str(root / "images" / split / f"{fid}.jpg"), rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
        vals = " ".join(f"{x} {y} 2" for x, y in KP)
        (root / "labels" / split / f"{fid}.txt").write_text(f"0 0.5 0.5 0.9 0.9 {vals}\n")
    return root


def test_dataset_item_shapes_and_range(tmp_path):
    ds = CropDataset(_make(tmp_path), "train", "final", True, INPUT)
    assert len(ds) == 2
    img, kp, fid = ds[0]
    assert fid == "f_0000" and img.shape == (3, 224, 224) and img.dtype == torch.float32
    assert kp.shape == (8,) and kp.dtype == torch.float32
    ev = CropDataset(tmp_path, "train", "final", False, INPUT)
    _, kp, _ = ev[1]
    np.testing.assert_allclose(kp.numpy(), np.array(KP, np.float32).reshape(-1), atol=1e-6)


def test_dataset_eval_normalisation(tmp_path):
    root = _make(tmp_path, n=1)
    cv2.imwrite(str(root / "images/train/f_0000.jpg"), np.full((256, 256, 3), 255, np.uint8))
    img, _, _ = CropDataset(root, "train", "none", False, INPUT)[0]
    mean, std = np.array(INPUT["mean"]), np.array(INPUT["std"])
    np.testing.assert_allclose(img[:, 5, 5].numpy(), (1 - mean) / std, atol=0.05)


def test_dataset_deterministic_per_epoch(tmp_path):
    ds = CropDataset(_make(tmp_path), "train", "final", True, INPUT, seed=7)
    ds.set_epoch(3)
    a, ka, _ = ds[1]
    b, kb, _ = ds[1]
    assert torch.equal(a, b) and torch.equal(ka, kb)
    ds.set_epoch(4)
    c, kc, _ = ds[1]
    assert not torch.equal(ka, kc) and not torch.equal(a, c)
    other = CropDataset(tmp_path, "train", "final", True, INPUT, seed=7)
    other.set_epoch(3)
    assert torch.equal(other[1][1], ka)


def test_dataset_keeps_out_of_crop_keypoints(tmp_path, monkeypatch):
    monkeypatch.setitem(RECIPES, "shift", {"fliplr": 0.0, "degrees": 0.0, "scale": 0.0, "translate": 1.0,
                                           "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0})
    ds = CropDataset(_make(tmp_path), "train", "shift", True, INPUT)
    vals = []
    for ep in range(20):
        ds.set_epoch(ep)
        _, kp, fid = ds[0]
        assert kp.shape == (8,) and fid == "f_0000"
        vals.append(kp.numpy())
    vals = np.stack(vals)
    assert vals.min() < 0 and vals.max() > 1


def test_dataset_refuses_test(tmp_path):
    root = _make(tmp_path, split="test")
    with pytest.raises(ValueError):
        CropDataset(root, "test", "none", False, INPUT)
    assert len(CropDataset(root, "test", "none", False, INPUT, allow_test=True)) == 2


def test_dataset_limit_and_cache_ram(tmp_path):
    root = _make(tmp_path, n=3)
    ds = CropDataset(root, "train", "none", False, INPUT, limit=2, cache_ram=True)
    assert len(ds) == 2
    ref = CropDataset(root, "train", "none", False, INPUT, limit=2)
    assert torch.equal(ds[1][0], ref[1][0])


def test_dataset_rejects_bad_labels_at_construction(tmp_path):
    root = _make(tmp_path)
    lab = root / "labels/train/f_0001.txt"
    lab.write_text("0 0.5 0.5 0.9 0.9 0.1 0.1 2\n")  # only one keypoint
    with pytest.raises(ValueError, match="f_0001"):
        CropDataset(root, "train", "none", False, INPUT)
    lab.write_text("0 0.5 0.5 0.9 0.9 " + "0.1 0.2 2 " * 4 + "\n0 0 0 0 0\n")  # two lines
    with pytest.raises(ValueError, match="one label line"):
        CropDataset(root, "train", "none", False, INPUT)
    lab.unlink()
    with pytest.raises(FileNotFoundError, match="f_0001"):
        CropDataset(root, "train", "none", False, INPUT)
