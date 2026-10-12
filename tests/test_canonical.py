import numpy as np

from dronepose.common.geometry import BLENDER_BODY_TO_FLU, geodesic_deg
from dronepose.data.canonical import model_reprojection_px, read_jsonl, to_canonical, write_jsonl
from dronepose.data.drone_models import build_drone_models
from dronepose.data.sixdronesyn import iter_frames
from tests.fixtures.fake_dataset import default_sequences


def _records(root):
    frames = list(iter_frames(root))
    models = build_drone_models(frames)
    return [to_canonical(f, models) for f in frames], models


def test_pose_gt_is_true_pose_in_flu_and_centroid(fake_root):
    truth = {(s.scene, s.drone): s for s in default_sequences()}
    records, _ = _records(fake_root)
    for r in records:
        s = truth[(r["scene"], r["drone"])]
        expected_R = s.R_true @ BLENDER_BODY_TO_FLU.T
        assert geodesic_deg(np.array(r["R_gt"]), expected_R) < 1e-4
        np.testing.assert_allclose(r["t_gt"], np.mean(r["kpts3d_cam"], axis=0), atol=1e-12)


def test_keypoints_reordered_and_split_assigned(fake_root):
    records, _ = _records(fake_root)
    r = next(x for x in records if x["scene"] == "03")
    assert r["split"] == "val" and r["units"] == "camera"
    k = np.array(r["kpts3d_cam"])
    R, t = np.array(r["R_gt"]), np.array(r["t_gt"])
    body = (R.T @ (k - t).T).T
    assert body[0, 0] > 0 and body[1, 0] > 0 and body[0, 1] < 0 < body[1, 1]   # 1 front-right, 2 front-left


def test_model_reprojects_on_keypoints(fake_root):
    records, models = _records(fake_root)
    assert max(model_reprojection_px(r, models) for r in records) < 0.1


def test_jsonl_roundtrip(fake_root, tmp_path):
    records, _ = _records(fake_root)
    n = write_jsonl(records, tmp_path / "c.jsonl")
    assert n == len(records)
    back = read_jsonl(tmp_path / "c.jsonl")
    assert back[0]["frame_id"] == records[0]["frame_id"]
    np.testing.assert_allclose(back[0]["R_gt"], records[0]["R_gt"])


import subprocess
import sys


def test_build_canonical_skips_missing_images_and_reports(fake_root, tmp_path):
    out = tmp_path / "processed"
    res = subprocess.run([sys.executable, "scripts/build_canonical.py", "--root", str(fake_root),
                          "--models", str(tmp_path / "m.json"), "--out", str(out)],
                         capture_output=True, text=True, check=True)
    assert "immagini mancanti: 1" in res.stdout
    total = sum(len(read_jsonl(p)) for p in out.glob("canonical_*.jsonl"))
    assert total == 10 - 1
    assert {p.name for p in out.glob("canonical_*.jsonl")} == {"canonical_train.jsonl", "canonical_val.jsonl",
                                                             "canonical_test.jsonl"}
