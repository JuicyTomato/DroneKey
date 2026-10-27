from dronepose.eval.class_eval import class_metrics, correct_rows, scene_of


def _recs(seq, drone, n):
    return [{"frame_id": f"{seq}_{i:04d}", "frame_idx": i, "drone": drone} for i in range(1, n + 1)]


def test_scene_of():
    assert scene_of("14_03_10_000_0005") == "14"


def test_class_metrics_counts_all_frames():
    recs = _recs("03_04_15_180", "04", 4)
    m = class_metrics(recs, {r["frame_id"]: c for r, c in zip(recs, ["05", "04", "04", "04"])})
    assert m["frame_acc"] == 0.75 and m["by_drone"] == {"04": 0.75}
    assert m["by_scene"] == {"03": 0.75} and m["seq_final_acc"] == 1.0 and m["n_frames"] == 4


def test_seq_final_acc_uses_last_frame():
    a, b = _recs("09_03_01_000", "03", 2), _recs("14_01_05_000", "01", 2)
    cls = {a[0]["frame_id"]: "03", a[1]["frame_id"]: "06", b[0]["frame_id"]: "05", b[1]["frame_id"]: "01"}
    m = class_metrics(a + b, cls)
    assert m["seq_final_acc"] == 0.5 and m["by_scene"] == {"09": 0.5, "14": 0.5}


def test_correct_rows():
    recs = _recs("03_04_15_180", "04", 2)
    rows = correct_rows(recs, {recs[0]["frame_id"]: "04", recs[1]["frame_id"]: "05"})
    assert [r["correct"] for r in rows] == [1.0, 0.0]
