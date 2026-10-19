import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.forecast.metrics import ade_at, fde_at, position_errors
from dronepose.forecast.noise import (RETUNE_GRIDS, add_pose_noise, chosen_params, evaluate_positions, noise_rng,
                                      parse_grid, retune_level, summarize_retuned, with_grids)
from dronepose.forecast.tuning import predict_all, tune
from dronepose.forecast.windows import make_windows
from tests.forecast.test_tuning import CFG, _traj


def _weighted(trajs, seed=0):
    rng = np.random.default_rng(seed)
    for t in trajs:
        t.meta["weight"] = float(rng.uniform(0.2, 3.0))
    return trajs


@pytest.fixture(scope="module")
def train():
    trajs = _weighted([_traj(T=50, chaos=True, seed=s) for s in range(2)] + [_traj(T=50, circle=True, seed=2)])
    return make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)


@pytest.fixture(scope="module")
def val():
    trajs = _weighted([_traj(T=48, chaos=True, seed=7), _traj(T=48, circle=True, seed=8)], seed=1)
    return make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)


def test_zero_noise_leaves_windows_unchanged(val):
    w = add_pose_noise(val, 0, 0, noise_rng(0, 0))
    np.testing.assert_allclose(w.obs_p, val.obs_p)
    np.testing.assert_allclose(w.obs_R, val.obs_R, atol=1e-12)
    assert w.fut_p is val.fut_p and w.fut_R is val.fut_R


def test_noise_matches_original_script_draws(val):
    """noise.json was produced with these exact draws: positions first, then rotation vectors, key [seed, 10 sr]."""
    sp, sr, seed = 2, 1, 1
    rng = np.random.default_rng([seed, int(10 * sr)])
    obs_p = val.obs_p + rng.normal(scale=sp / 100, size=val.obs_p.shape)
    W, N = val.obs_R.shape[:2]
    dR = Rotation.from_rotvec(rng.normal(scale=np.radians(sr), size=(W * N, 3))).as_matrix().reshape(W, N, 3, 3)
    obs_R = np.asarray(val.obs_R, float) @ dR
    w = add_pose_noise(val, sp, sr, noise_rng(seed, sr))
    np.testing.assert_array_equal(w.obs_p, obs_p)
    np.testing.assert_array_equal(w.obs_R, obs_R)
    np.testing.assert_array_equal(w.obs_u, obs_R[..., :, 2])


def test_noise_scale_and_rotations(val):
    w = add_pose_noise(val, 5, 3, noise_rng(0, 3))
    d = (w.obs_p - val.obs_p).ravel()
    assert 0.04 < d.std() < 0.06
    R = np.asarray(w.obs_R)
    np.testing.assert_allclose(R @ np.swapaxes(R, -1, -2), np.broadcast_to(np.eye(3), R.shape), atol=1e-9)
    ang = np.degrees(Rotation.from_matrix((np.swapaxes(val.obs_R, -1, -2) @ R).reshape(-1, 3, 3)).magnitude())
    assert 3 < ang.mean() < 7                      # |N(0, s^2 I3)| has mean ~1.6 s


def test_train_stream_differs_from_val_stream():
    a = noise_rng(0, 1).normal(size=5)
    b = noise_rng(0, 1, salt=1).normal(size=5)
    assert not np.allclose(a, b)
    np.testing.assert_array_equal(noise_rng(0, 1, salt=1).normal(size=5), b)


def test_parse_grid_and_with_grids():
    assert parse_grid("kf_r_grid=1e-6,1e-1") == ("kf_r_grid", (1e-6, 0.1))
    assert parse_grid("acro_k_grid=3,15") == ("acro_k_grid", (3, 15))
    with pytest.raises(ValueError):
        parse_grid("nope_grid=1")
    cfg = with_grids(CFG, {"kf_r_grid": (1e-6, 0.1), "acro_k_grid": (3, 15)})
    assert cfg.kf_r_grid == (1e-6, 0.1) and cfg.acro_k_grid == (3, 15)
    assert all(isinstance(k, int) for k in cfg.acro_k_grid)
    assert cfg.cv_k_grid == CFG.cv_k_grid
    assert max(RETUNE_GRIDS["kf_r_grid"]) >= 1e-1 and max(RETUNE_GRIDS["acro_k_grid"]) == 15
    assert max(RETUNE_GRIDS["ctr_k_grid"]) == 15


def test_evaluate_positions_weighted(val):
    params = tune(val, CFG)
    res = evaluate_positions(val, params, CFG)
    preds = predict_all(val, params, CFG)
    assert "rot_hold" not in res and {"cv", "kf_ca", "physics_acro_gen"} <= set(res)
    e = position_errors(preds["kf_ca"], val.fut_p)
    wts = np.asarray(val.meta["weight"], float)
    assert res["kf_ca"]["ade_1s"] == pytest.approx(np.average(ade_at(e, 30), weights=wts))
    assert res["kf_ca"]["fde_1s"] == pytest.approx(np.average(fde_at(e, 30), weights=wts))


def test_retune_level_at_zero_noise_is_clean_tuning(train, val):
    run = retune_level(train, val, 0, 0, 0, CFG)
    params = tune(train, CFG)
    assert run["params"] == chosen_params(params)
    assert run["val"] == evaluate_positions(val, params, CFG)
    assert run["train_scores"] == pytest.approx(params["scores"])


def test_retune_level_tunes_on_noisy_train(train, val):
    run = retune_level(train, val, 3, 2, 0, CFG)
    noisy = add_pose_noise(train, 3, 2, noise_rng(0, 2, salt=1))
    assert run["params"] == chosen_params(tune(noisy, CFG))
    assert run["sigma_p_cm"] == 3 and run["sigma_r_deg"] == 2 and run["seed"] == 0


def test_summarize_retuned_means_over_seeds_and_skill():
    def run(seed, a, b):
        return {"sigma_p_cm": 1, "sigma_r_deg": 0.5, "seed": seed, "params": {"cv_k": seed},
                "val": {"kf_ca": {"ade_1s": b, "fde_1s": 2 * b}, "physics_acro_gen": {"ade_1s": a, "fde_1s": 2 * a}}}
    s = summarize_retuned([run(0, 0.2, 0.4), run(1, 0.4, 0.4)])
    lvl = s["1cm_0.5deg"]
    assert lvl["physics_acro_gen"]["ade_1s"] == pytest.approx(0.3)
    assert lvl["kf_ca"]["fde_1s"] == pytest.approx(0.8)
    assert lvl["skill_acro_gen_vs_kf_ca"]["ade_1s"] == pytest.approx(1 - 0.3 / 0.4)
    assert lvl["skill_acro_gen_vs_kf_ca"]["per_seed_ade_1s"] == pytest.approx([0.5, 0.0])
    assert lvl["params"] == [{"cv_k": 0}, {"cv_k": 1}]
