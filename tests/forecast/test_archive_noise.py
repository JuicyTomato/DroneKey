import dataclasses
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.forecast.archive_noise import (Tracts, apply_tracts, archive_noise_rng, load_tracts, sample_tracts,
                                              yaw_angles)
from tests.forecast.test_tuning import _traj, _windows

S01, S02 = 0.7463, 0.8053


def _fake_archive(tmp_path, true="01", pred=None, scene="03", n=15, gap_at=None, meters=True, seed=0,
                  dp=(0.0, 0.01, 0.0), los=(0.0, 1.0, 0.0), distance=2.0, valid=True, name=None):
    pred = pred or true
    path = tmp_path / (name or f"arch_{scene}_{true}_{pred}_{seed}.jsonl")
    rows = [{"kind": "header", "version": 1}]
    seq = f"{scene}_{true}_05_180"
    frames = [i for i in range(n + (gap_at is not None)) if i != gap_at]
    for f in frames:
        rows.append({"frame_id": f"{seq}_{f:04d}", "sequence": seq, "frame_idx": f, "scene": scene, "drone": true,
                     "drone_model": pred, "seed": seed, "valid": valid, "dp_g": list(dp) if valid else None,
                     "meters": meters, "dR": np.eye(3).tolist() if valid else None, "distance_m": distance,
                     "los_g": list(los)})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_rescale_predicted_class(tmp_path):
    tr = load_tracts([_fake_archive(tmp_path, true="01", pred="02")])
    k = S02 / S01
    los, d = np.array([0, 1.0, 0]), 2.0
    np.testing.assert_allclose(tr.dp[0, 0], (np.array([0, .01, 0]) + d * los) * k - d * los)
    assert tr.n_rescaled == 15 and tr.valid.all()


def test_predicted_08_is_lost(tmp_path):
    tr = load_tracts([_fake_archive(tmp_path, true="03", pred="08")])
    assert not tr.valid.any() and tr.n_lost_pred08 == 15
    np.testing.assert_array_equal(tr.dp[0], 0)
    np.testing.assert_array_equal(tr.dR[0], np.broadcast_to(np.eye(3), (15, 3, 3)))


def test_true_08_unchanged(tmp_path):
    tr = load_tracts([_fake_archive(tmp_path, true="08", pred="01", meters=False)])
    np.testing.assert_allclose(tr.dp[0, 0], [0, .01, 0])
    assert not tr.meters[0] and tr.n_rescaled == 0


def test_no_rescale_flag(tmp_path):
    tr = load_tracts([_fake_archive(tmp_path, true="01", pred="02")], rescale=False)
    np.testing.assert_allclose(tr.dp[0, 0], [0, .01, 0])


def test_invalid_rows_are_lost_frames(tmp_path):
    tr = load_tracts([_fake_archive(tmp_path, valid=False)])
    assert len(tr.dp) == 1 and not tr.valid.any()
    np.testing.assert_allclose(tr.distance[0], 2.0)


def test_tracts_contiguous_only(tmp_path):
    assert len(load_tracts([_fake_archive(tmp_path, n=20, gap_at=10)]).dp) == 0
    assert len(load_tracts([_fake_archive(tmp_path, n=20, name="b.jsonl")]).dp) == 6


def test_select_scenes_and_meters(tmp_path):
    paths = [_fake_archive(tmp_path, scene=s) for s in ("03", "09", "14")]
    paths.append(_fake_archive(tmp_path, true="08", meters=False, scene="03"))
    tr = load_tracts(paths)
    sub = tr.select(("03", "09"), meters=True)
    assert set(sub.scene) == {"03", "09"} and sub.meters.all() and len(sub.dp) == 2
    assert set(tr.select(("03",), meters=False).drone) == {"08"}


def test_seeds_are_merged(tmp_path):
    tr = load_tracts([_fake_archive(tmp_path, seed=s) for s in (0, 1, 2)])
    assert sorted(tr.seed) == [0, 1, 2]


def _tracts_with_distances(tmp_path, distances):
    return load_tracts([_fake_archive(tmp_path, distance=d, seed=i, name=f"d{i}.jsonl") for i, d in enumerate(distances)])


def test_distance_bins(tmp_path):
    tr = _tracts_with_distances(tmp_path, [2.0, 4.0])
    idx, n_fb = sample_tracts(tr, 50, archive_noise_rng(0), distance=np.r_[np.full(25, 1.0), np.full(25, 5.0)])
    assert n_fb == 0
    assert (tr.distance[idx[:25], -1] <= 3).all() and (tr.distance[idx[25:], -1] > 3).all()


def test_distance_bin_fallback(tmp_path):
    tr = _tracts_with_distances(tmp_path, [2.0, 2.5])           # no tract beyond 3 m
    idx, n_fb = sample_tracts(tr, 10, archive_noise_rng(0), distance=np.full(10, 4.0))
    assert n_fb == 10 and len(idx) == 10


def test_uniform_sampling_reproducible(tmp_path):
    tr = _tracts_with_distances(tmp_path, [2.0, 2.5, 4.0])
    a, _ = sample_tracts(tr, 20, archive_noise_rng(3, 1))
    b, _ = sample_tracts(tr, 20, archive_noise_rng(3, 1))
    np.testing.assert_array_equal(a, b)


def test_yaw_alignment(tmp_path):
    tr = load_tracts([_fake_archive(tmp_path, dp=(0, 0.1, 0), los=(0, 1, 0))])
    w = _windows([_traj()])
    idx = np.zeros(len(w), int)
    yaw = yaw_angles(tr, idx, archive_noise_rng(0), target_los=np.tile([1.0, 0, 0], (len(w), 1)))
    w2 = apply_tracts(w, tr, idx, yaw)
    np.testing.assert_allclose(w2.obs_p - w.obs_p, np.broadcast_to([0.1, 0, 0], w.obs_p.shape), atol=1e-12)


def test_random_yaw_keeps_norm_and_vertical(tmp_path):
    tr = load_tracts([_fake_archive(tmp_path, dp=(0.03, 0.04, 0.02))])
    w = _windows([_traj()])
    idx = np.zeros(len(w), int)
    yaw = yaw_angles(tr, idx, archive_noise_rng(1))
    d = apply_tracts(w, tr, idx, yaw).obs_p - w.obs_p
    np.testing.assert_allclose(np.linalg.norm(d, axis=-1), np.sqrt(0.03 ** 2 + 0.04 ** 2 + 0.02 ** 2))
    np.testing.assert_allclose(d[..., 2], 0.02)
    assert len(np.unique(np.round(yaw, 9))) > 1


def test_apply_sets_mask_rotation_and_keeps_future(tmp_path):
    rot = Rotation.from_rotvec([0.02, -0.01, 0.03]).as_matrix()
    path = _fake_archive(tmp_path)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    for r in rows[1:]:
        r["dR"] = rot.tolist()
    rows[5]["valid"], rows[5]["dp_g"], rows[5]["dR"] = False, None, None
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tr = load_tracts([path])
    w = _windows([_traj()])
    idx = np.zeros(len(w), int)
    w2 = apply_tracts(w, tr, idx, np.zeros(len(w)))
    assert w2.obs_valid.shape == w.obs_p.shape[:2] and not w2.obs_valid[:, 4].any() and w2.obs_valid[:, 3].all()
    np.testing.assert_allclose(w2.obs_R[0, 0], w.obs_R[0, 0] @ rot)
    np.testing.assert_allclose(w2.obs_u, w2.obs_R[..., :, 2])
    np.testing.assert_array_equal(w2.fut_p, w.fut_p)
    np.testing.assert_array_equal(w2.fut_R, w.fut_R)


def test_apply_keeps_existing_mask(tmp_path):
    tr = load_tracts([_fake_archive(tmp_path)])
    w = _windows([_traj()])
    v = np.ones(w.obs_p.shape[:2], bool)
    v[:, 2] = False
    w2 = apply_tracts(dataclasses.replace(w, obs_valid=v), tr, np.zeros(len(w), int), np.zeros(len(w)))
    assert not w2.obs_valid[:, 2].any() and w2.obs_valid[:, 3].all()


def test_apply_length_mismatch_raises(tmp_path):
    tr = load_tracts([_fake_archive(tmp_path)])
    w = _windows([_traj()])
    with pytest.raises(ValueError):
        apply_tracts(w, tr, np.zeros(len(w) - 1, int), np.zeros(len(w) - 1))
