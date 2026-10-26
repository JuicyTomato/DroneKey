import json

import numpy as np
import pytest

from dronepose.common.geometry import rot_z
from dronepose.eval.units import GRAVITY_FROM_CAMERA as G
from dronepose.pose.error_archive import archive_rows, header, load_archive, windows, write_archive

R_GT = rot_z(30.0)


def rec(fid, idx, drone="01", t_gt=(0.0, 0.0, 3.0)):
    return {"frame_id": fid, "frame_idx": idx, "drone": drone, "scene": "city", "R_gt": R_GT.tolist(),
            "t_gt": list(t_gt)}


def res(fid, idx, R=None, t=None, rejected=False, reason=None, rot_err=None):
    return {"frame_id": fid, "frame_idx": idx, "R": None if R is None else R.tolist(),
            "t": None if t is None else list(t), "rejected": rejected, "reject_reason": reason,
            "rot_err_deg": rot_err}


def test_dp_and_dR_exact():
    r = rec("s_00000", 0)
    row = archive_rows([res("s_00000", 0, R_GT @ rot_z(5.0), (0.1, 0.0, 3.0), rot_err=5.0)],
                       {"s_00000": r}, None, seed=7)[0]
    assert row["valid"] and row["meters"] and row["seed"] == 7 and row["drone_model"] == "01"
    np.testing.assert_allclose(row["dp_g"], G @ np.array([0.1, 0, 0]) * 0.7463, atol=1e-12)
    np.testing.assert_allclose(row["dR"], rot_z(5.0), atol=1e-12)
    np.testing.assert_allclose(row["rotvec_deg"], [0, 0, 5], atol=1e-9)
    np.testing.assert_allclose(row["distance_m"], 3.0 * 0.7463)
    assert row["sequence"] == "s" and not row["flip"] and not row["symmetric"]
    json.dumps(row, allow_nan=False)


def test_gravity_consistency():
    R_est = R_GT @ rot_z(12.0)
    row = archive_rows([res("s_00000", 0, R_est, (0.2, -0.1, 3.3))], {"s_00000": rec("s_00000", 0)}, None, 0)[0]
    np.testing.assert_allclose(G @ R_GT @ np.array(row["dR"]), G @ R_est, atol=1e-12)


def test_drone08_dataset_units():
    r = rec("t_00000", 0, drone="08", t_gt=(1.0, 2.0, 4.0))
    row = archive_rows([res("t_00000", 0, R_GT, (1.5, 2.0, 4.0), rot_err=0.0)], {"t_00000": r}, None, 0)[0]
    assert row["meters"] is False and row["symmetric"] is True
    np.testing.assert_allclose(row["dp_g"], G @ np.array([0.5, 0, 0]), atol=1e-12)
    np.testing.assert_allclose(row["distance_m"], np.linalg.norm([1.0, 2.0, 4.0]))


def test_model_drone_override():
    row = archive_rows([res("s_00000", 0, R_GT, (0, 0, 3.0))], {"s_00000": rec("s_00000", 0)},
                       {"s_00000": "02"}, 0)[0]
    assert row["drone"] == "01" and row["drone_model"] == "02"


def test_invalid_frames_kept_in_order():
    recs = {f"s_{i:05d}": rec(f"s_{i:05d}", i) for i in range(3)}
    results = [res("s_00002", 2, R_GT, (0, 0, 3.0), rot_err=0.0),
               res("s_00001", 1, rejected=True, reason="reproj"),
               res("s_00000", 0, R_GT, (0, 0, 3.0), rot_err=0.0)]
    rows = archive_rows(results, recs, None, 0)
    assert [r["frame_idx"] for r in rows] == [0, 1, 2]
    bad = rows[1]
    assert bad["valid"] is False and bad["reason"] == "reproj" and bad["dp_g"] is None and bad["dR"] is None
    assert bad["rotvec_deg"] is None and bad["rot_err_deg"] is None and bad["flip"] is False
    assert bad["drone"] == "01" and bad["scene"] == "city" and bad["distance_m"] > 0


def test_all_invalid_sequence(tmp_path):
    recs = {f"s_{i:05d}": rec(f"s_{i:05d}", i) for i in range(4)}
    results = [res(f"s_{i:05d}", i, rejected=True, reason="no_detection") for i in range(4)]
    rows = archive_rows(results, recs, None, 0)
    p = tmp_path / "a.jsonl"
    write_archive(p, header("c", 0, "abc"), rows)
    _, arch = load_archive(p)
    se = arch["s"]
    assert not se.valid.any() and np.isnan(se.dp_g).all() and np.isnan(se.dR).all()
    assert np.isfinite(se.distance_m).all()
    assert len(windows(arch, length=3)) == 2


def test_load_roundtrip(tmp_path):
    recs, results = {}, []
    for i in range(5):
        fid = f"s_{i:05d}"
        recs[fid] = rec(fid, i)
        results.append(res(fid, i, R_GT @ rot_z(i), (0.1 * i, 0, 3.0), rot_err=float(i)) if i != 2
                       else res(fid, i, rejected=True, reason="x"))
    rows = archive_rows(results, recs, None, 1)
    p = tmp_path / "a.jsonl"
    write_archive(p, header("combo", 1, "abc"), list(reversed(rows)))   # load must not assume order
    head, arch = load_archive(p)
    assert head["kind"] == "header" and head["version"] == 1 and head["combo"] == "combo"
    se = arch["s"]
    assert se.drone == "01" and se.meters and list(se.frame_idx) == [0, 1, 2, 3, 4]
    assert list(se.valid) == [True, True, False, True, True]
    for i, r in enumerate(rows):
        if r["valid"]:
            np.testing.assert_allclose(se.dp_g[i], r["dp_g"])
            np.testing.assert_allclose(se.dR[i], r["dR"])
        np.testing.assert_allclose(se.distance_m[i], r["distance_m"])
    assert np.isnan(se.dp_g[2]).all()


def test_windows_contiguous():
    def se(idx):
        n = len(idx)
        return type("S", (), {"frame_idx": np.array(idx)})()
    full = {"a": se(range(20))}
    w = windows(full, 15)
    assert w == [("a", i) for i in range(6)]
    assert windows(full, 15, step=2) == [("a", 0), ("a", 2), ("a", 4)]
    gap = {"a": se([i for i in range(20) if i != 10])}
    assert windows(gap, 15) == []
    gap2 = {"a": se([i for i in range(40) if i != 10])}
    assert all(s >= 10 for _, s in windows(gap2, 15)) and len(windows(gap2, 15)) == 15


def test_flip_flag():
    row = archive_rows([res("s_00000", 0, R_GT @ rot_z(175.0), (0, 0, 3.0), rot_err=175.0)],
                       {"s_00000": rec("s_00000", 0)}, None, 0)[0]
    assert row["flip"] is True
