import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.data.canonical import to_canonical
from dronepose.data.drone_models import build_drone_models
from dronepose.data.sixdronesyn import iter_frames
from dronepose.keypoints.predictors import GTPredictor
from dronepose.pipeline.skeleton import run_sequence
from dronepose.pose.replay import (check_no_test, gt_rotation_steps_deg, load_keypoints, replay, sequence_key,
                                   strategy_table)
from tests.fixtures.fake_dataset import FakeSequence, write_fake_dataset


def _two_sequences(tmp_path):
    R = Rotation.from_euler("xyz", [0.3, 0.1, 0.5]).as_matrix()
    seqs = [FakeSequence("03", "01", "05", 90, 20, R, np.array([-0.3, 0.0, 3.0]), np.array([0.01, 0.0, 0.0])),
            FakeSequence("03", "01", "07", 90, 20, R, np.array([0.2, 0.1, 3.5]), np.array([0.0, 0.01, 0.0]))]
    write_fake_dataset(tmp_path / "ds", seqs)
    frames = list(iter_frames(tmp_path / "ds"))
    models = build_drone_models(frames)
    return [to_canonical(f, models) for f in frames], models


def _write_kpts(path, recs, sigma=0.0, drop=()):
    pred = GTPredictor(sigma_px=sigma, seed=0)
    with open(path, "w") as f:
        for r in recs:
            if r["frame_id"] in drop:
                continue
            f.write(json.dumps({"frame_id": r["frame_id"], "kpts_pred": pred.predict(None, r).tolist()}) + "\n")


def test_sequence_key():
    assert sequence_key("14_03_09_000_0001") == "14_03_09_000"


def test_groups_sequences_and_matches_online_pipeline(tmp_path):
    recs, models = _two_sequences(tmp_path)
    _write_kpts(tmp_path / "k.jsonl", recs)
    by_seq = replay(recs, load_keypoints(tmp_path / "k.jsonl"), models, strategy="temporal")
    assert sorted(by_seq) == ["03_01_05_090", "03_01_07_090"]
    seq = sorted((r for r in recs if r["frame_id"].startswith("03_01_05_")), key=lambda r: r["frame_idx"])
    online = run_sequence(seq, lambda r: None, GTPredictor(), models, strategy="temporal")
    assert [x["rot_err_deg"] for x in by_seq["03_01_05_090"]] == pytest.approx([x["rot_err_deg"] for x in online])


def test_missing_keypoints_count_as_no_detection(tmp_path):
    recs, models = _two_sequences(tmp_path)
    drop = {recs[3]["frame_id"]}
    _write_kpts(tmp_path / "k.jsonl", recs, drop=drop)
    table = strategy_table(replay(recs, load_keypoints(tmp_path / "k.jsonl"), models))
    assert table["n_missing_kpts"] == 1
    assert table["all"]["n_pose"] == len(recs) - 1
    assert "01" in table["by_drone"]


def test_test_split_is_guarded():
    check_no_test([{"split": "val"}], allow_test=False)
    with pytest.raises(ValueError, match="test"):
        check_no_test([{"split": "val"}, {"split": "test"}], allow_test=False)
    check_no_test([{"split": "test"}], allow_test=True)


def test_gt_rotation_steps_are_zero_for_constant_attitude(tmp_path):
    recs, _ = _two_sequences(tmp_path)
    steps = gt_rotation_steps_deg(recs)
    assert len(steps) == 2 * 19 and steps.max() < 1e-3


def test_global_replay_matches_online_pipeline_and_handles_two_sequences(tmp_path):
    recs, models = _two_sequences(tmp_path)
    _write_kpts(tmp_path / "k.jsonl", recs, sigma=3.0)
    by_seq = replay(recs, load_keypoints(tmp_path / "k.jsonl"), models, strategy="global")
    assert sorted(by_seq) == ["03_01_05_090", "03_01_07_090"]
    seq = sorted((r for r in recs if r["frame_id"].startswith("03_01_05_")), key=lambda r: r["frame_idx"])
    online = run_sequence(seq, lambda r: None, GTPredictor(sigma_px=3.0, seed=0), models, strategy="global")
    got = [x["rot_err_deg"] for x in by_seq["03_01_05_090"]]
    assert [g is None for g in got] == [x["rot_err_deg"] is None for x in online]
    assert [g for g in got if g is not None] == pytest.approx([x["rot_err_deg"] for x in online if x["rot_err_deg"] is not None])


def _drone02_sequence(tmp_path):
    R = Rotation.from_euler("xyz", [0.3, 0.1, 0.5]).as_matrix()
    seqs = [FakeSequence("03", "02", "05", 90, 20, R, np.array([-0.3, 0.0, 3.0]), np.array([0.01, 0.0, 0.0])),
            FakeSequence("03", "01", "05", 90, 3, R, np.array([0.2, 0.1, 3.5]))]   # only to build the 01 model
    write_fake_dataset(tmp_path / "ds", seqs)
    frames = list(iter_frames(tmp_path / "ds"))
    models = build_drone_models(frames)
    return [to_canonical(f, models) for f in frames if f.drone == "02"], models


def test_replay_model_drone_changes_pose_not_metric_symmetry(tmp_path):
    recs, models = _drone02_sequence(tmp_path)
    _write_kpts(tmp_path / "k.jsonl", recs)
    kpts = load_keypoints(tmp_path / "k.jsonl")
    base = replay(recs, kpts, models)["03_02_05_090"]
    same = replay(recs, kpts, models, model_drone={r["frame_id"]: "02" for r in recs})["03_02_05_090"]
    assert same == base
    wrong = replay(recs, kpts, models, model_drone={r["frame_id"]: "01" for r in recs})["03_02_05_090"]
    got = [(b, w) for b, w in zip(base, wrong) if b["t"] is not None and w["t"] is not None]
    assert got and all(not np.allclose(b["t"], w["t"], atol=1e-3) for b, w in got)
    assert [w["frame_id"] for w in wrong] == [b["frame_id"] for b in base]
