import json

import numpy as np
import pytest

from pathlib import Path

from scipy.spatial.transform import Rotation

from dronepose.common.geometry import project
from dronepose.data.drone_models import load_drone_models
from dronepose.keypoints.class_scores import combine, fuse_causal, load_probs, log_scores, probs_row, \
    reproj_per_class, scores_c1, scores_cm, scores_geom, scores_k
from tests.fixtures.fake_dataset import FAKE_K
from dronepose.keypoints.detections import Detection, class_causal_vote


def _seq(classes, seq="03_01_05_090", confs=None):
    recs = [{"frame_id": f"{seq}_{i + 1:04d}", "frame_idx": i + 1, "drone": "01"} for i in range(len(classes))]
    confs = confs or [0.9] * len(classes)
    dets = {r["frame_id"]: Detection((0.0, 0.0, 1.0, 1.0), k, c)
            for r, c, k in zip(recs, classes, confs) if c is not None}
    return recs, dets


def _vals(out, recs):
    return [out[r["frame_id"]] for r in recs]


def test_sum_of_unweighted_c1_equals_majority_vote():
    recs, dets = _seq(["03", "06", "06", "03", None, "06"])
    assert fuse_causal(recs, scores_c1(dets, weighted=False), "S") == class_causal_vote(recs, dets)


def test_sum_of_unweighted_c1_equals_majority_vote_on_ties_and_gaps():
    recs, dets = _seq([None, "05", "03", "03", "05", None, "02"])
    assert fuse_causal(recs, scores_c1(dets, weighted=False), "S") == class_causal_vote(recs, dets)


def test_weighted_c1_uses_confidence():
    recs, dets = _seq(["03", "06", "06"], confs=[0.9, 0.3, 0.3])
    assert _vals(fuse_causal(recs, scores_c1(dets, weighted=True), "S"), recs) == ["03", "03", "03"]


def test_frame_rule_holds_last_class_and_backfills_start():
    recs, dets = _seq([None, "05", None, "03"])
    assert _vals(fuse_causal(recs, scores_c1(dets, False), "F"), recs) == ["05", "05", "05", "03"]


def test_sequence_without_scores_raises():
    recs, _ = _seq([None, None])
    with pytest.raises(ValueError, match="03_01_05_090"):
        fuse_causal(recs, {}, "S")


def test_unknown_temporal_rule_raises():
    recs, dets = _seq(["03"])
    with pytest.raises(ValueError, match="temporal"):
        fuse_causal(recs, scores_c1(dets, False), "V")


def test_log_sum_ties_go_to_first_seen():
    # log scores: both frames equal on 03/06 after two frames -> 03 (first argmax seen) wins
    recs = [{"frame_id": f"03_01_05_090_{i:04d}", "frame_idx": i, "drone": "01"} for i in (1, 2)]
    s = {recs[0]["frame_id"]: {"03": -1.0, "06": -2.0}, recs[1]["frame_id"]: {"03": -2.0, "06": -1.0}}
    assert _vals(fuse_causal(recs, s, "S"), recs) == ["03", "03"]


def test_combine_and_log():
    s = combine([({"f": {"03": 1.0}}, 2.0), ({"f": {"06": 1.0}, "g": {"01": 1.0}}, 1.0)])
    assert s == {"f": {"03": 2.0, "06": 1.0}, "g": {"01": 1.0}}
    lg = log_scores({"f": {"03": 1.0}}, 0.1)
    assert lg["f"]["06"] == pytest.approx(np.log(0.1)) and lg["f"]["03"] == 0.0 and len(lg["f"]) == 7


MODELS = load_drone_models(Path(__file__).resolve().parents[1] / "configs" / "drone_models.json")


def _projected(drone):
    R = Rotation.from_euler("xz", [20, 30], degrees=True).as_matrix()
    img = project(FAKE_K, (R @ MODELS[drone].kpts_flu.T).T + np.array([0.1, -0.05, 3.0]))
    return np.hstack([img, np.ones((4, 1))])


def test_reproj_true_model_is_zero():
    e = reproj_per_class({"K": FAKE_K.tolist()}, _projected("03"), MODELS)
    assert set(e) == set(MODELS) and e["03"] < 1e-3 and min(e, key=e.get) == "03"


def test_reproj_nan_keypoints_gives_none():
    kp = _projected("03")
    kp[2, 0] = np.nan
    assert reproj_per_class({"K": FAKE_K.tolist()}, kp, MODELS) is None


def test_geom_log_and_quad():
    r = {"f": {"03": 0.0, "06": 100.0}, "g": None}
    lg = scores_geom(r, "log")
    assert lg["f"]["06"] == pytest.approx(-np.log(50.1)) and lg["f"]["03"] == pytest.approx(-np.log(0.1))
    assert "g" not in lg
    assert scores_geom({"f": {"03": 2.0}}, "quad", sigma=1.0)["f"]["03"] == pytest.approx(-2.0)
    with pytest.raises(ValueError, match="form"):
        scores_geom(r, "abs")


def test_scores_cm_keeps_overlapping_classes():
    # class index 2 = "03", 5 = "06", 0 = "01" (CLASS_IDS order)
    boxes = {"f": [[0, 0, 10, 10, 0.8, 2], [0, 0, 10, 9, 0.5, 5], [0, 0, 10, 10, 0.4, 2], [50, 50, 60, 60, 0.7, 0]],
             "g": []}
    assert scores_cm(boxes, 0.5) == {"f": {"03": 0.8, "06": 0.5}}


NAMES = {0: "08", 1: "01", 2: "02", 3: "03", 4: "04", 5: "05", 6: "06"}


def test_probs_row_maps_by_name():
    p = np.array([0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03])
    row = probs_row("f", p, NAMES)
    assert row["frame_id"] == "f" and row["probs"]["08"] == pytest.approx(0.7) and row["probs"]["01"] == pytest.approx(0.1)
    assert probs_row("g", None, NAMES) == {"frame_id": "g", "probs": None}


def test_probs_row_rejects_unknown_names():
    with pytest.raises(ValueError, match="names"):
        probs_row("f", np.ones(7) / 7, {i: str(i) for i in range(7)})


def test_load_probs_and_scores_k(tmp_path):
    path = tmp_path / "p.jsonl"
    rows = [probs_row("f", np.array([0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03]), NAMES), probs_row("g", None, NAMES)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = scores_k(load_probs(path))
    assert set(s) == {"f"} and max(s["f"], key=s["f"].get) == "08"


def test_scores_cm_ignores_frames_without_a_detection_above_threshold():
    boxes = {"f": [[0, 0, 10, 10, 0.02, 2], [0, 0, 10, 10, 0.01, 5]],
             "g": [[0, 0, 10, 10, 0.30, 2], [0, 0, 10, 10, 0.05, 5]]}
    assert scores_cm(boxes, 0.5) == {"g": {"03": 0.30, "06": 0.05}}
