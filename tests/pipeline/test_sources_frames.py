import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronepose.eval.units import GRAVITY_FROM_CAMERA
from dronepose.pipeline.frames import reduce_frame
from dronepose.pipeline.sources import SixDroneSynSource

CANON = Path("data/processed/canonical_val.jsonl")
CACHED = Path("data/processed/yolo_frames/images/val")


def _write(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _rec(idx, split="val", seq="14_02_07_000"):
    return {"frame_id": f"{seq}_{idx:04d}", "image_path": f"/nowhere/{idx}.png", "drone": "02", "frame_idx": idx,
            "split": split, "K": np.eye(3).tolist()}


def test_reduce_frame_shape():
    img = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    assert reduce_frame(img).shape == (360, 640, 3)


@pytest.mark.dataset
def test_reduce_frame_matches_cached_jpeg():
    recs = [json.loads(l) for _, l in zip(range(3), open(CANON))]
    for r in recs:
        live = reduce_frame(cv2.imread(r["image_path"]))
        cached = cv2.imread(str(CACHED / f"{r['frame_id']}.jpg"))
        assert np.array_equal(live, cached), r["frame_id"]


def test_source_orders_frames_and_rejects_test(tmp_path, monkeypatch):
    p = tmp_path / "c.jsonl"
    _write(p, [_rec(3), _rec(1), _rec(2)])
    src = SixDroneSynSource(p)
    monkeypatch.setattr(cv2, "imread", lambda *a, **k: np.zeros((1080, 1920, 3), np.uint8))
    assert src.sequences() == ["14_02_07_000"]
    assert [f.frame_idx for f in src.frames("14_02_07_000")] == [1, 2, 3]
    _write(p, [_rec(1), _rec(2, split="test")])
    with pytest.raises(ValueError):
        SixDroneSynSource(p)


def test_source_missing_image_raises(tmp_path):
    p = tmp_path / "c.jsonl"
    _write(p, [_rec(1)])
    with pytest.raises(FileNotFoundError):
        next(SixDroneSynSource(p).frames("14_02_07_000"))


def test_source_info_sixdronesyn(tmp_path):
    p = tmp_path / "c.jsonl"
    _write(p, [_rec(1)])
    info = SixDroneSynSource(p).info("14_02_07_000")
    assert np.array_equal(info.G, GRAVITY_FROM_CAMERA)
    assert info.fixed_class is None
    assert len(info.models) == 7
