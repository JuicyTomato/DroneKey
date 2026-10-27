import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import Trajectory, load_trajectories
from dronepose.forecast.e4 import load_pose_windows, split_nets_scenes
from dronepose.forecast.windows import make_windows

CFG = load_config()
S01, S02 = 0.7463, 0.8053


def _traj(scene="03", drone="01", s_c=S01, T=60, copies=(("05", 180), ("06", 180))):
    t = np.arange(T) / 30
    p = np.stack([0.3 + 0.8 * t, 2 - 0.5 * t, 0.2 * t], axis=1)
    R = Rotation.from_euler("xz", np.stack([5 * np.sin(t), 20 * t], axis=1), degrees=True).as_matrix()
    return Trajectory(scene=scene, drone=drone, split="val", motion_type="linear", copies=list(copies), s_c=s_c,
                      times=t, p_G=p, R_G=R)


def _pose_file(tmp_path, trajs, split="val", pred=None, invalid=(), offset=0.0, seed=0, name="poses.jsonl"):
    rows = []
    for tr in trajs:
        for bg, deg in tr.copies:
            seq = f"{tr.scene}_{tr.drone}_{bg}_{int(deg):03d}"
            for i in range(len(tr.times)):
                ok = i not in invalid
                rows.append({"frame_id": f"{seq}_{i + 1:04d}", "split": split, "sequence": seq, "frame_idx": i + 1,
                             "drone": tr.drone, "drone_model": pred or tr.drone, "valid": ok,
                             "R_g": tr.R_G[i].tolist() if ok else None,
                             "t_g": (tr.positions[i] + offset).tolist() if ok else None,
                             "meters": tr.metric, "seed": seed})
    path = Path(tmp_path) / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_exact_poses_reproduce_gt_windows(tmp_path):
    trajs = [_traj(), _traj(drone="02", s_c=S02, copies=(("05", 180),))]
    w, info = load_pose_windows(_pose_file(tmp_path, trajs), trajs, CFG)
    gt = make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)
    n0 = int((gt.traj == 0).sum())
    assert len(w) == 2 * n0 + int((gt.traj == 1).sum())                 # one window per copy
    assert w.obs_valid.all() and info["n_rescaled"] == 0
    np.testing.assert_allclose(w.obs_p[:n0], gt.obs_p[:n0], atol=1e-12)
    np.testing.assert_allclose(w.obs_R[:n0], gt.obs_R[:n0], atol=1e-12)
    np.testing.assert_array_equal(w.fut_p[:n0], gt.fut_p[:n0])
    assert set(w.meta["copy"]) == {"05_180", "06_180"} and set(w.meta["pose_seed"]) == {0}


def test_offset_moves_only_observed(tmp_path):
    trajs = [_traj()]
    w, _ = load_pose_windows(_pose_file(tmp_path, trajs, offset=0.1), trajs, CFG)
    gt = make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)
    np.testing.assert_allclose(w.obs_p[: len(gt)] - gt.obs_p, 0.1, atol=1e-12)
    np.testing.assert_array_equal(w.fut_p[: len(gt)], gt.fut_p)
    np.testing.assert_array_equal(w.last_p[: len(gt)], gt.last_p)


def test_pose_rescale_predicted_class(tmp_path):
    trajs = [_traj(copies=(("05", 180),))]
    w, info = load_pose_windows(_pose_file(tmp_path, trajs, pred="02"), trajs, CFG)
    gt = make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)
    expected = (gt.obs_p + gt.last_p[:, None]) * (S02 / S01) - gt.last_p[:, None]
    np.testing.assert_allclose(w.obs_p, expected, atol=1e-12)
    assert info["n_rescaled"] == len(trajs[0].times)
    w2, _ = load_pose_windows(_pose_file(tmp_path, trajs, pred="02", name="b.jsonl"), trajs, CFG, rescale=False)
    np.testing.assert_allclose(w2.obs_p, gt.obs_p, atol=1e-12)


def test_pose_pred08_lost_and_invalid_frames(tmp_path):
    trajs = [_traj(copies=(("05", 180),))]
    w, info = load_pose_windows(_pose_file(tmp_path, trajs, pred="08"), trajs, CFG)
    assert not w.obs_valid.any() and info["n_lost_pred08"] == len(trajs[0].times)
    w2, _ = load_pose_windows(_pose_file(tmp_path, trajs, invalid=(14,), name="c.jsonl"), trajs, CFG)
    assert not w2.obs_valid[0, 14] and w2.obs_valid[0, :14].all() and not w2.obs_valid[1, 13]


def test_true_08_not_rescaled(tmp_path):
    trajs = [_traj(drone="08", s_c=None, copies=(("05", 180),))]
    w, info = load_pose_windows(_pose_file(tmp_path, trajs, pred="01"), trajs, CFG)
    gt = make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)
    np.testing.assert_allclose(w.obs_p, gt.obs_p, atol=1e-12)
    assert info["n_rescaled"] == 0 and not w.metric.any()


def test_test_split_refused(tmp_path):
    trajs = [_traj()]
    path = _pose_file(tmp_path, trajs, split="test")
    with pytest.raises(ValueError):
        load_pose_windows(path, trajs, CFG)
    load_pose_windows(path, trajs, CFG, allow_test=True)


def test_missing_sequence_raises(tmp_path):
    trajs = [_traj()]
    path = _pose_file(tmp_path, [_traj(copies=(("05", 180),))])
    with pytest.raises(ValueError):
        load_pose_windows(path, trajs, CFG)


def test_split_nets_scenes():
    trajs = [_traj(scene=s) for s in ("01", "02", "05", "11")] + [_traj(scene="04", drone="08", s_c=None)]
    w = make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)
    tr, sel = split_nets_scenes(w)
    assert set(tr.scene) == {"01", "11"} and set(sel.scene) == {"02", "05"}
    assert tr.metric.all() and sel.metric.all()


VAL_POSES = Path("experiments/poses/poses_val_s0_causal.jsonl")


@pytest.mark.skipif(not VAL_POSES.exists(), reason="pose export not present")
def test_real_val_poses_close_to_gt():
    trajs = [t for t in load_trajectories(Path("data/processed/forecast"), "val") if t.metric]
    w, info = load_pose_windows(VAL_POSES, trajs, CFG)
    gt = make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)
    rep = np.repeat(np.arange(len(gt)), 1)  # windows per copy follow the GT order inside each copy
    v = w.obs_valid
    d = np.linalg.norm(w.obs_p - gt.subset(w.meta["gt_index"]).obs_p, axis=-1)[v]
    assert 0.01 < np.median(d) < 0.08
    rel = np.einsum("...ji,...jk->...ik", gt.subset(w.meta["gt_index"]).obs_R[v], w.obs_R[v])
    ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    assert 0.5 < np.median(ang) < 3.0
    assert info["n_rescaled"] > 0 and len(rep) > 0
