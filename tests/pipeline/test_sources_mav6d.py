import json
from pathlib import Path

import numpy as np
import pytest

from dronepose.data import mav6d
from dronepose.data.mav6d import planar
from dronepose.eval.units import METERS_PER_UNIT
from dronepose.pipeline.sources import MAV6DSource

KPTS = [[0.15, 0.15, 0.0], [0.15, -0.15, 0.0], [-0.15, -0.15, 0.001], [-0.15, 0.15, 0.001]]
# VICON z up -> camera y down (like the real calibration)
CAM_FROM_VICON = [[0.6685859, -0.74342, 0.01787715, -0.38], [0.01558769, -0.01002444, -0.99982825, 1.6],
                  [0.74347153, 0.66874974, 0.004886, 3.0]]
STEP = 41_666_667


def _rec(seq, split, i, in_image=True):
    ts = 1_600_000_000_000_000_000 + i * STEP
    return {"frame_id": f"{seq}_{ts}", "image_path": f"/nonexistent/{i}.jpg", "sequence": seq, "frame_idx": i,
            "authors_split": split, "K": mav6d.K.tolist(), "dist": mav6d.DIST.tolist(),
            "kpts2d": [[10.0, 20.0]] * 4, "R_gt": np.eye(3).tolist(), "t_gt": [0.1, 0.2, 2.0 + i],
            "drone": "05", "in_image": in_image}


@pytest.fixture
def src(tmp_path):
    recs = [_rec("01_0101", "train", i, in_image=(i != 1)) for i in (2, 0, 1, 3)]   # shuffled on purpose
    recs += [_rec("02_0201", "test", i) for i in range(3)]
    gt = tmp_path / "gt.jsonl"
    gt.write_text("\n".join(json.dumps(r) for r in recs))
    hub = tmp_path / "hub.json"
    hub.write_text(json.dumps({"kpts_flu": KPTS, "cam_from_vicon": CAM_FROM_VICON}))
    return MAV6DSource(gt, hub)


def test_lists_only_authors_train(src):
    assert src.sequences() == ["01_0101"]


def test_refuses_test_sequence(src):
    for call in (src.records, src.info, lambda s: list(src.frames(s)), src.fps):
        with pytest.raises(ValueError, match="test"):
            call("02_0201")


def test_unknown_sequence_is_key_error(src):
    with pytest.raises(KeyError):
        src.records("nope")


def test_frames_in_order_including_out_of_image(src, monkeypatch):
    import dronepose.pipeline.sources as S
    monkeypatch.setattr(S.cv2, "imread", lambda p, flags=None: np.zeros((4, 4, 3), np.uint8))
    frames = list(src.frames("01_0101"))
    assert [f.frame_idx for f in frames] == [0, 1, 2, 3]
    assert frames[1].record["in_image"] is False
    assert all(np.array_equal(f.dist, mav6d.DIST) for f in frames)


def test_units_of_model_05(src):
    m = METERS_PER_UNIT["05"]
    r0 = src.records("01_0101")[0]
    assert np.allclose(r0["t_gt"], np.array([0.1, 0.2, 2.0]) / m)
    assert np.allclose(src.info("01_0101").models["05"].kpts_flu, planar(np.array(KPTS)) / m)
    assert src.info("01_0101").fixed_class == "05"


def test_gravity_up_matches_vicon(src):
    G = src.info("01_0101").G
    A = np.array(CAM_FROM_VICON)[:, :3]
    assert np.allclose(G @ (A @ [0, 0, 1]), [0, 0, 1], atol=1e-3)


def test_fps_from_timestamps(src):
    assert src.fps("01_0101") == pytest.approx(24.0, abs=1e-3)


_REAL_GT = Path("data/processed/mav6d_gt.jsonl")
_REAL_HUB = Path("experiments/real/mav6d/hub_model.json")


@pytest.mark.skipif(not (_REAL_GT.exists() and _REAL_HUB.exists()), reason="real MAV6D files missing")
def test_real_gravity_is_rotation_with_y_down():
    s = MAV6DSource(_REAL_GT, _REAL_HUB)
    G = s.info(s.sequences()[0]).G
    assert np.linalg.det(G) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(G @ G.T, np.eye(3), atol=1e-6)
    assert (G.T @ np.array([0.0, 0.0, 1.0]))[1] < 0
