import json

import pytest

from dronepose.keypoints.detections import Detection, class_causal_vote, class_per_frame, load_detections


def _seq(classes, seq="03_01_05_090"):
    recs = [{"frame_id": f"{seq}_{i + 1:04d}", "frame_idx": i + 1, "drone": "01"} for i in range(len(classes))]
    dets = {r["frame_id"]: Detection((0.0, 0.0, 1.0, 1.0), 0.9, c) for r, c in zip(recs, classes) if c is not None}
    return recs, dets


def _vote(classes):
    recs, dets = _seq(classes)
    out = class_causal_vote(recs, dets)
    return [out[r["frame_id"]] for r in recs]


def test_causal_vote_majority_and_ties():
    # 03, 06, 06, 03, None -> ties go to the class seen first in the sequence
    assert _vote(["03", "06", "06", "03", None]) == ["03", "03", "06", "03", "03"]


def test_causal_vote_starts_without_detection():
    assert _vote([None, None, "05"]) == ["05", "05", "05"]


def test_causal_vote_single_frame():
    assert _vote(["02"]) == ["02"]


def test_causal_vote_no_detection_raises():
    recs, dets = _seq([None, None])
    with pytest.raises(ValueError, match="no detection"):
        class_causal_vote(recs, dets)


def test_causal_vote_is_per_sequence_and_sorted_by_frame_idx():
    a, da = _seq(["03", "03"], "03_01_05_090")
    b, db = _seq(["06", "06"], "03_01_07_090")
    recs = [b[1], a[1], b[0], a[0]]
    out = class_causal_vote(recs, {**da, **db})
    assert out == {a[0]["frame_id"]: "03", a[1]["frame_id"]: "03",
                   b[0]["frame_id"]: "06", b[1]["frame_id"]: "06"}


def test_per_frame_fills_missing_with_first_seen():
    recs, dets = _seq([None, "05", "03", None])
    out = class_per_frame(recs, dets)
    assert [out[r["frame_id"]] for r in recs] == ["05", "05", "03", "05"]


def test_load_detections_maps_class_index_to_drone(tmp_path):
    rows = [{"frame_id": "a_0001", "kpts_pred": [], "box": [1, 2, 3, 4.5], "box_conf": 0.93, "cls": 2},
            {"frame_id": "a_0002", "kpts_pred": [], "box": None, "box_conf": 0.0, "cls": None}]
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    dets = load_detections(p)
    assert dets["a_0001"] == Detection((1.0, 2.0, 3.0, 4.5), 0.93, "03")
    assert dets["a_0002"] == Detection(None, 0.0, None)
