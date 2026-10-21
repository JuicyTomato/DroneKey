import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.forecast.config import load_config
from dronepose.forecast.nets.features import (BASES, Normaliser, base_predictions, build_batch, build_pool, load_pool,
                                              rotate_yaw, save_pool)
from dronepose.forecast.tuning import predict_all
from dronepose.forecast.windows import make_windows
from tests.forecast.test_tuning import _traj

CFG = load_config()
PARAMS = {"cv_k": 2, "ca_k": 3, "kf_cv": {"q": 100.0, "r": 1e-6}, "kf_ca": {"q": 1000.0, "r": 1e-6}, "ctr_k": 3,
          "acro_k": 3, "acro_kd": "estimate", "acro_gen": {"lam": 5.0, "thrust": "hold_vz", "k": 3, "kd": "estimate"},
          "acro_mode": {"acro": {"lam": 0.0, "thrust": "constant", "k": 3, "kd": "estimate"},
                        "angle": {"lam": 8.0, "thrust": "hold_vz", "k": 5, "kd": "estimate"},
                        "pmode": {"lam": 5.0, "thrust": "hold_vz", "k": 3, "kd": 0.7}}}
MODES = ("acro", "angle", "pmode")


def _gen_trajs(n=6):
    trajs = []
    for i in range(n):
        tr = _traj(T=48, chaos=i % 2 == 0, circle=i % 2 == 1, seed=i)
        tr.meta = {"mode": MODES[i % 3], "drone_class": ("01", "02", "08")[i % 3], "weight": 0.5 + i, "sample_id": i}
        trajs.append(tr)
    return trajs


def _windows(trajs):
    return make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)


@pytest.fixture(scope="module")
def win():
    return _windows(_gen_trajs())


@pytest.fixture(scope="module")
def batch(win):
    return build_batch(win, PARAMS, CFG)


def test_base_predictions_match_predict_all(win):
    ours, ref = base_predictions(win, PARAMS, CFG), predict_all(win, PARAMS, CFG)
    for k, r in (("cv", "cv"), ("phys", "physics_acro_gen"), ("phys_mode", "physics_acro_mode"), ("kf_ca", "kf_ca")):
        np.testing.assert_array_equal(ours[k], ref[r])


def test_shapes_and_bases(win, batch):
    b = batch
    W = len(win)
    assert b.x.shape == (W, 15, 9) and b.target.shape == (W, 30, 3) and b.weight.shape == (W,)
    assert set(b.bases) == set(BASES) and all(v.shape == (W, 30, 3) for v in b.bases.values())
    assert b.mode_onehot.shape == (W, 3) and b.class_onehot.shape == (W, 7)
    np.testing.assert_allclose(b.mode_onehot.sum(1), 1)
    np.testing.assert_allclose(b.class_onehot.sum(1), 1)
    np.testing.assert_allclose(b.weight, win.meta["weight"])
    np.testing.assert_allclose(b.x[:, -1, :3], 0)                       # positions relative to the last observation
    np.testing.assert_allclose(b.x[:, :, 6:], win.obs_u, atol=1e-6)


def test_target_is_future_relative(win, batch):
    np.testing.assert_allclose(batch.target, win.fut_p, atol=1e-6)


def test_noisy_window_reanchors_on_noisy_last_observation(win, batch):
    shifted = dataclasses.replace(win, obs_p=win.obs_p + np.array([0.1, -0.2, 0.05]))
    b, b0 = build_batch(shifted, PARAMS, CFG), batch
    np.testing.assert_allclose(b.target, win.fut_p - np.array([0.1, -0.2, 0.05]), atol=1e-6)
    np.testing.assert_allclose(b.x, b0.x, atol=1e-6)
    np.testing.assert_allclose(b.bases["cv"], b0.bases["cv"], atol=1e-5)


def _rotate_windows(w, angles):
    Rz = Rotation.from_euler("z", angles[:, None]).as_matrix()                   # (W,3,3)
    rot = lambda p: np.einsum("wij,w...j->w...i", Rz, p)
    R = np.einsum("wij,wnjk->wnik", Rz, np.asarray(w.obs_R, float))
    return dataclasses.replace(w, obs_p=rot(w.obs_p), fut_p=rot(w.fut_p), obs_R=R, obs_u=R[..., :, 2],
                               fut_R=np.einsum("wij,wnjk->wnik", Rz, np.asarray(w.fut_R, float)))


def test_yaw_rotation_consistent_with_recomputed_bases(win, batch):
    angles = np.random.default_rng(0).uniform(0, 2 * np.pi, len(win))
    a = rotate_yaw(batch, angles)
    b = build_batch(_rotate_windows(win, angles), PARAMS, CFG)
    np.testing.assert_allclose(a.x, b.x, atol=1e-5)
    # 1e-6 m plus the float32 storage resolution (~1e-6 relative at a few metres)
    np.testing.assert_allclose(a.target, b.target, atol=1e-6, rtol=1e-6)
    # the Kalman CA at q / r = 1e9 amplifies round-off (~2e-5 m); an inconsistent rotation would be off by metres
    for k in BASES:
        np.testing.assert_allclose(a.bases[k], b.bases[k], atol=1e-4 if k == "kf_ca" else 1e-6, rtol=1e-6, err_msg=k)


def test_normalisation_fit_on_train_only(batch):
    tr = batch
    other = rotate_yaw(batch.subset(np.arange(len(batch))[::-1]), np.full(len(batch), 0.7))
    other.x[..., 2] += 0.3                                              # a set with other statistics
    n = Normaliser.fit(tr)
    state = n.to_json()
    out = n.apply(other)
    assert n.to_json() == state                                         # apply does not refit
    m = Normaliser.from_json(state)
    assert m.to_json() == state
    np.testing.assert_allclose(m.apply(other).x, out.x)
    np.testing.assert_allclose(out.target, other.target / n.pos_scale, rtol=1e-6)
    np.testing.assert_allclose(out.bases["phys"], other.bases["phys"] / n.pos_scale, rtol=1e-6)
    # horizontal channels: zero mean and a shared scale (yaw rotation commutes with the normalisation)
    assert np.all(n.x_mean[[0, 1, 3, 4, 6, 7]] == 0)
    assert n.x_std[0] == n.x_std[1] and n.x_std[3] == n.x_std[4] and n.x_std[6] == n.x_std[7]
    ang = np.linspace(0, 3, len(tr))
    np.testing.assert_allclose(n.apply(rotate_yaw(tr, ang)).x, rotate_yaw(n.apply(tr), ang).x, atol=1e-5)


def test_pool_copies_differ_and_are_reproducible(tmp_path):
    win = _windows(_gen_trajs(3))
    p1 = build_pool(win, PARAMS, CFG, K=3, seed=0)
    p2 = build_pool(win, PARAMS, CFG, K=3, seed=0)
    assert len(p1) == 3
    assert not np.allclose(p1[0].x, p1[1].x) and not np.allclose(p1[1].x, p1[2].x)
    for a, b in zip(p1, p2):
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.bases["phys"], b.bases["phys"])
    save_pool(p1, tmp_path / "pool.npz")
    q = load_pool(tmp_path / "pool.npz")
    assert len(q) == 3
    for a, b in zip(p1, q):
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.target, b.target)
        np.testing.assert_array_equal(a.weight, b.weight)
        for k in BASES:
            np.testing.assert_array_equal(a.bases[k], b.bases[k])


def test_6dronesyn_windows_have_zero_mode():
    trajs = [_traj(T=60, chaos=True, seed=1), _traj(T=60, drone="08", s_c=None, seed=2)]
    w = _windows(trajs)
    b = build_batch(w, PARAMS, CFG)
    assert np.all(b.mode_onehot == 0) and np.all(b.class_onehot == 0)
    np.testing.assert_allclose(b.weight, 1)
    np.testing.assert_array_equal(b.bases["phys_mode"], b.bases["phys"])    # no mode: the mode-blind physics
    np.testing.assert_array_equal(b.metric, w.metric)


def test_load_pool_reads_only_requested_bases(batch, tmp_path):
    save_pool([batch, batch], tmp_path / "pool.npz")
    q = load_pool(tmp_path / "pool.npz", bases=("phys",))
    assert len(q) == 2 and set(q[0].bases) == {"phys"}
    np.testing.assert_array_equal(q[1].bases["phys"], batch.bases["phys"])
