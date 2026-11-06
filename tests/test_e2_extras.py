import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dronepose.eval.detection import detection_metrics
from dronepose.keypoints.detections import Detection

_spec = importlib.util.spec_from_file_location("eval_e2", Path("scripts/real/eval_e2.py"))
e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2)
KP = [[100, 100], [200, 100], [200, 200], [100, 200]]


def test_recall_counts_missing_frames():
    from dronepose.data.crops import label_box_xyxy
    recs = [{"frame_id": "a", "kpts2d": KP, "drone": "05"}, {"frame_id": "b", "kpts2d": KP, "drone": "05"}]
    dets = {"a": Detection(label_box_xyxy(np.array(KP, float)).tolist(), 0.9, "05")}
    m = detection_metrics(dets, recs)
    assert m["recall_05"] == pytest.approx(0.5) and m["iou_ge_05"] == pytest.approx(1.0)


def test_recall_with_no_frames_is_none():
    assert detection_metrics({}, [])["recall_05"] is None


def test_empty_subset_is_skipped():
    recs = [{"frame_id": "a", "authors_split": "train", "distance_m": 3.5, "px_speed": 200.0}]
    s = e2.subsets(recs)
    assert "authors_test" not in s and "dist3-4" in s and s["all"] == {"a"}


def test_only_frames_of_canonical_keeps_ids():
    assert set(e2.restrict({"a": 1, "b": 2}, {"a": {}})) == {"a"}


def test_syn_models_scaled_to_metres():
    m = e2.syn_models(Path("configs/drone_models.json"))
    assert set(m) == {"01", "02", "03", "04", "05", "06", "08"}
    from dronepose.data.drone_models import load_drone_models
    from dronepose.eval.units import METERS_PER_UNIT
    raw = load_drone_models(Path("configs/drone_models.json"))
    assert np.allclose(m["05"].kpts_flu, raw["05"].kpts_flu * METERS_PER_UNIT["05"])
    assert np.allclose(m["08"].kpts_flu, raw["08"].kpts_flu)


def test_vote_survives_a_sequence_without_detections():
    recs = [{"frame_id": "a_1_1", "frame_idx": 1, "drone": "05"}, {"frame_id": "b_1_1", "frame_idx": 1, "drone": "05"}]
    v = e2.vote_or_true(recs, {"a_1_1": Detection((0, 0, 1, 1), 0.9, "03")})
    assert v == {"a_1_1": "03", "b_1_1": "05"}


def test_ids_in(tmp_path):
    from dronepose.keypoints.detections import ids_in
    p = tmp_path / "gt.jsonl"
    p.write_text(json.dumps({"frame_id": "a"}) + "\n")
    assert ids_in(p) == {"a"}


def test_detection_line_survives_zero_detections():
    line = e2.detection_line("New", detection_metrics({}, [{"frame_id": "a", "kpts2d": KP, "drone": "05"}]))
    assert "recall" in line and "no-det 1.000" in line


def test_detection_line_shows_recall():
    from dronepose.data.crops import label_box_xyxy
    recs = [{"frame_id": "a", "kpts2d": KP, "drone": "05"}, {"frame_id": "b", "kpts2d": KP, "drone": "05"}]
    dets = {"a": Detection(label_box_xyxy(np.array(KP, float)).tolist(), 0.9, "05")}
    assert "recall@0.5 0.500" in e2.detection_line("New", detection_metrics(dets, recs))


def test_dump_refuses_the_test_split():
    with pytest.raises(ValueError, match="authors_test"):
        e2.check_dump({"all": {"a"}, "authors_test": {"a"}}, "dir")
    e2.check_dump({"all": {"a"}}, "dir")
    e2.check_dump({"all": {"a"}, "authors_test": {"a"}}, None)
