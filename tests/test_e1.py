import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dronepose.eval.e1 import e1_metrics, e1_table, mean_std, subsets
from dronepose.eval.units import tilt_deg

_spec = importlib.util.spec_from_file_location("eval_e1", Path("scripts/eval_e1.py"))
eval_e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_e1)

R_LEVEL = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])   # FLU -> camera, level drone
GT_KPTS = np.array([[100.0, 100.0], [140.0, 100.0], [140.0, 140.0], [100.0, 140.0]])


def rec(fid, drone, scene="14", split="val", R=R_LEVEL):
    return {"frame_id": fid, "drone": drone, "scene": scene, "split": split, "frame_idx": 1,
            "kpts2d": GT_KPTS.tolist(), "bbox_xyxy": [90.0, 90.0, 150.0, 150.0], "R_gt": R.tolist()}


def res(fid, rot=1.0, t_err=1.0, posed=True):
    return {"frame_id": fid, "rot_err_deg": rot if posed else None, "rot_err_sym_deg": rot if posed else None,
            "t_err": t_err if posed else None}


def kp(offset=0.0, conf=0.9):
    return np.c_[GT_KPTS + offset, np.full(4, conf)]


def tilted(deg):
    a = np.radians(deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return Rx @ R_LEVEL


def test_tilt_level_drone_is_zero():
    assert tilt_deg(R_LEVEL) == pytest.approx(0, abs=1e-6)
    assert tilt_deg(tilted(20)) == pytest.approx(20, abs=1e-6)


def test_subsets_breakdown_names():
    records = [rec("14_01_a_1", "01", "14", R=tilted(5)), rec("03_08_a_1", "08", "03", R=tilted(40))]
    base = subsets(records, False)
    assert set(base) == {"all", "no08", "08"}
    assert base["no08"] == {"14_01_a_1"} and base["08"] == {"03_08_a_1"}
    full = subsets(records, True)
    assert {"drone_01", "drone_08", "scene_14", "scene_03", "tilt_0-15", "tilt_30+"} <= set(full)
    assert full["tilt_0-15"] == {"14_01_a_1"} and full["tilt_30+"] == {"03_08_a_1"}


def test_meters_excludes_08_and_uses_true_drone():
    records = [rec("14_01_a_1", "01"), rec("14_08_a_1", "08")]
    results = [res("14_01_a_1"), res("14_08_a_1")]
    kpts = {r["frame_id"]: kp() for r in records}
    m = e1_metrics(results, {r["frame_id"]: r for r in records}, kpts)
    assert m["median_t_meters"] == pytest.approx(0.7463)
    assert m["median_t_world"] == pytest.approx(1.718)
    assert m["median_t_camera"] == pytest.approx(1.0)
    only08 = e1_metrics(results[1:], {r["frame_id"]: r for r in records}, kpts)
    assert only08["median_t_meters"] is None


def test_coverage_and_missing_keypoints():
    records = [rec(f"14_01_a_{i}", "01") for i in range(4)]
    by_id = {r["frame_id"]: r for r in records}
    results = [res(r["frame_id"]) for r in records[:3]] + [res(records[3]["frame_id"], posed=False)]
    kpts = {r["frame_id"]: kp(offset=3.0) for r in records[:3]}
    m = e1_metrics(results, by_id, kpts)
    assert m["coverage"] == pytest.approx(0.75)
    assert m["px_median"] == pytest.approx(np.hypot(3, 3))        # only the 3 frames with keypoints
    assert m["pck_2px"] == pytest.approx(0.0)
    assert m["pck_5px"] == pytest.approx(0.75)                      # the missing frame fails
    assert m["sr90"] == pytest.approx(0.75)                         # OKS 0.975 on 3 of 4 frames


def test_e1_has_ap_paper_and_old_keys_unchanged():
    records = [rec(f"14_01_a_{i}", "01") for i in range(4)]
    by_id = {r["frame_id"]: r for r in records}
    results = ([res(r["frame_id"], rot=1.0 + i, t_err=1.0 + i) for i, r in enumerate(records[:3])]
               + [res(records[3]["frame_id"], posed=False)])
    kpts = {r["frame_id"]: kp(offset=3.0 * i) for i, r in enumerate(records[:3])}
    m = e1_metrics(results, by_id, kpts)
    golden = {"n_frames": 4, "coverage": 0.75, "median_rot_sym": 2.0, "p90_rot_sym": 2.8, "median_t_camera": 2.0,
              "median_t_world": 3.436, "median_t_meters": 1.4926, "flip_rate": 0.0, "px_median": 4.242640687119285,
              "px_p90": 7.636753236814712, "pck_2px": 0.25, "pck_5px": 0.5, "pck_10px": 0.75,
              "pck_0.02diag": 0.25, "pck_0.05diag": 0.5, "identity": 1.0, "oks_ap": 0.7277227722772277,
              "sr90": 0.75, "sr95": 0.5}                          # recorded before ap_paper was added
    assert set(m) == set(golden) | {"ap_paper"}
    for k, v in golden.items():
        assert m[k] == pytest.approx(v), k
    # OKS = [1, 0.9753, 0.9048, 0 (missing frame)]: 0.75 up to 0.90 (9 thresholds), 0.50 at 0.95
    assert m["ap_paper"] == pytest.approx(0.725)


def test_perfect_keypoints_and_scores():
    records = [rec("14_01_a_1", "01"), rec("14_01_a_2", "01")]
    by_id = {r["frame_id"]: r for r in records}
    kpts = {r["frame_id"]: kp() for r in records}
    m = e1_metrics([res(r["frame_id"]) for r in records], by_id, kpts, {"14_01_a_1": 0.3, "14_01_a_2": 0.8})
    assert m["px_median"] == 0 and m["identity"] == 1.0 and m["oks_ap"] == pytest.approx(1.0)
    assert m["pck_0.02diag"] == 1.0 and m["sr95"] == 1.0 and m["flip_rate"] == 0.0


def test_e1_table_rows():
    records = [rec("14_01_a_1", "01"), rec("03_08_a_1", "08", "03")]
    kpts = {r["frame_id"]: kp() for r in records}
    table = e1_table([res(r["frame_id"]) for r in records], records, kpts, breakdown=True)
    assert table["all"]["n_frames"] == 2 and table["no08"]["n_frames"] == 1 and table["08"]["n_frames"] == 1
    assert "drone_01" in table and "scene_03" in table


def test_mean_std():
    rows = {"all": {"median_rot_sym": 1.0, "median_t_meters": None}}
    other = {"all": {"median_rot_sym": 3.0, "median_t_meters": None}}
    ms = mean_std([rows, other])
    assert ms["all"]["median_rot_sym"]["mean"] == pytest.approx(2.0)
    assert ms["all"]["median_rot_sym"]["std"] == pytest.approx(1.4142, 1e-4)
    assert ms["all"]["median_t_meters"] == {"mean": None, "std": None}


def test_parse_spec():
    p = eval_e1.parse_spec("Bc=k.jsonl,det=d.jsonl,cls=causal")
    assert p == {"name": "Bc", "kpts": "k.jsonl", "det": "d.jsonl", "cls": "causal"}
    assert eval_e1.parse_spec("A=k.jsonl")["cls"] == "gt"
    with pytest.raises(ValueError):
        eval_e1.parse_spec("A=k.jsonl,cls=frame")       # class vote needs detections


def test_eval_rejects_test_split(tmp_path):
    records = [rec("06_01_a_1", "01", "06", split="test")]
    with pytest.raises(ValueError, match="test split"):
        eval_e1.run(["A=missing.jsonl"], split="test", allow_test=False, records=records)


@pytest.mark.parametrize("gravity", [False, True])
def test_gravity_flag_reaches_replay(tmp_path, monkeypatch, gravity):
    seen = {}

    def fake_replay(*args, **kwargs):
        seen["up"] = kwargs.get("up")
        raise RuntimeError("stop")

    monkeypatch.setattr(eval_e1, "replay", fake_replay)
    path = tmp_path / "k.jsonl"
    path.write_text(json.dumps({"frame_id": "14_01_a_1", "kpts_pred": kp().tolist()}) + "\n")
    with pytest.raises(RuntimeError, match="stop"):
        eval_e1.run([f"A={path}"], records=[rec("14_01_a_1", "01")], gravity=gravity)
    if gravity:
        assert np.array_equal(seen["up"], [0.0, -1.0, 0.0])
    else:
        assert seen["up"] is None
