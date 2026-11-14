import json
from pathlib import Path

import numpy as np
import pytest

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.eval.units import GRAVITY_FROM_CAMERA
from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.e4 import load_pose_windows
from dronepose.pipeline.decide import decide_sequence, redecided_windows, write_pose_file
from dronepose.pipeline.perception import Perception
from dronepose.pipeline.sources import SceneInfo
from dronepose.pose.replay import load_keypoints, replay

YOLO_KPTS = Path("experiments/results/keypoints/keypoints_val_b_cbest_yolo26s.jsonl")
CANON = Path("data/processed/canonical_val.jsonl")
SEQ = "14_02_07_000"
P = {"a_min": 0.1, "w_g": 1.0, "form": "log"}
MODELS = load_drone_models(Path("configs/drone_models.json"))
needs_files = pytest.mark.skipif(not (YOLO_KPTS.exists() and CANON.exists()), reason="exported files missing")


def _records(n=None):
    recs = sorted((r for r in read_jsonl(CANON) if r["frame_id"].startswith(SEQ + "_")), key=lambda r: r["frame_idx"])
    return recs[:n]


def _percs(recs, kp):
    out = []
    for r in recs:
        k = kp.get(r["frame_id"], np.full((4, 3), np.nan))
        out.append(Perception(r["frame_id"], r["frame_idx"], [0, 0, 1, 1], 0.9, np.asarray(k, float), None, None, {}))
    return out


@needs_files
def test_offline_matches_export_poses():
    recs, kp = _records(), load_keypoints(YOLO_KPTS)
    info = SceneInfo(GRAVITY_FROM_CAMERA, "02", MODELS)
    res, red = decide_sequence(_percs(recs, kp), recs, info, P, "offline")
    ref = replay(recs, kp, MODELS, "global", model_drone={r["frame_id"]: "02" for r in recs}, solver="ippe",
                 refine=True)[SEQ]
    assert red == {}
    for a, b in zip(res, ref):
        assert a["rejected"] == b["rejected"], a["frame_id"]
        if not a["rejected"]:
            assert np.allclose(a["R"], b["R"], atol=1e-9) and np.allclose(a["t"], b["t"], atol=1e-9)


@needs_files
def test_streaming_emitted_are_causal():
    recs, kp = _records(80), load_keypoints(YOLO_KPTS)
    info = SceneInfo(GRAVITY_FROM_CAMERA, "02", MODELS)
    pa = _percs(recs, kp)
    pb = _percs(recs, kp)
    for p in pb[50:]:
        p.kpts = p.kpts[[2, 3, 0, 1]]                       # garbage after frame 50
    ra, _ = decide_sequence(pa, recs, info, P, "streaming", W=30)
    rb, _ = decide_sequence(pb, recs, info, P, "streaming", W=30)
    for a, b in zip(ra[:50], rb[:50]):
        assert a["rejected"] == b["rejected"] and (a["R"] == b["R"])


def test_class_none_means_no_pose():
    recs = [{"frame_id": f"{SEQ}_{i:04d}", "frame_idx": i, "drone": "02", "K": np.eye(3).tolist()} for i in (1, 2)]
    percs = [Perception(r["frame_id"], r["frame_idx"], None, 0.0, np.full((4, 3), np.nan), None, None, {})
             for r in recs]
    info = SceneInfo(GRAVITY_FROM_CAMERA, None, MODELS)
    res, _ = decide_sequence(percs, recs, info, P, "streaming", W=15)
    assert all(r["rejected"] and r["reject_reason"] == "no_class" for r in res)


def _fake_results(tr, seq, rng, drop=()):
    res = []
    for i in range(len(tr.times)):
        rej = i in drop
        R = np.eye(3)
        t = rng.normal(0, 1, 3) + [0, 0, 10]
        res.append({"frame_id": f"{seq}_{i + 1:04d}", "frame_idx": i + 1, "rejected": rej,
                    "reject_reason": "cost" if rej else None, "R": None if rej else R.tolist(),
                    "t": None if rej else t.tolist(), "drone_model": tr.drone})
    return res


def test_pose_file_and_redecided_windows(tmp_path):
    cfg = load_config(Path("configs/forecast.json"))
    trajs = [t for t in load_trajectories(Path("data/processed/forecast"), "val") if t.drone != "08"][:1]
    tr = trajs[0]
    rng = np.random.default_rng(0)
    by_seq, recs_by_id, red_by_seq = {}, {}, {}
    for bg, deg in tr.copies:
        seq = f"{tr.scene}_{tr.drone}_{bg}_{int(deg):03d}"
        res = _fake_results(tr, seq, rng, drop=(3, 20))
        by_seq[seq] = res
        for r in res:
            recs_by_id[r["frame_id"]] = {"frame_id": r["frame_id"], "drone": tr.drone}
        red_by_seq[seq] = {r["frame_idx"]: [x for x in res if r["frame_idx"] - 15 < x["frame_idx"] <= r["frame_idx"]]
                           for r in res}
    path = tmp_path / "poses.jsonl"
    write_pose_file(by_seq, recs_by_id, "val", 0, "test", path)
    w_ref, _ = load_pose_windows(path, trajs, cfg)
    assert not w_ref.obs_valid.all()
    w = redecided_windows(red_by_seq, by_seq, recs_by_id, trajs, cfg)
    assert np.allclose(w.obs_p, w_ref.obs_p) and np.allclose(w.obs_R, w_ref.obs_R)
    assert np.array_equal(w.obs_valid, w_ref.obs_valid)
    assert np.array_equal(w.meta["gt_index"], w_ref.meta["gt_index"])
