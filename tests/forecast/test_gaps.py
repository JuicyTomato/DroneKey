import dataclasses

import numpy as np
import pytest

from dronepose.forecast.baselines import ca, cv
from dronepose.forecast.acro import acro_predict
from dronepose.forecast.gaps import (acro_g, ca_g, cv_g, drop_sparse, last_k_valid, last_valid_index, predict_gapped,
                                     restrict_frames)
from dronepose.forecast.kalman import kalman_predict
from dronepose.forecast.tuning import predict_all, tune
from tests.forecast.test_tuning import CFG, _traj, _windows

PARAMS = {"cv_k": 2, "ca_k": 3, "kf_cv": {"q": 100.0, "r": 1e-6}, "kf_ca": {"q": 1000.0, "r": 1e-6},
          "ctr_k": {"ctrv": 3, "ctra": 5}, "acro_k": 3, "acro_kd": "estimate",
          "acro_gen": {"lam": 5.0, "thrust": "hold_vz", "k": 3, "kd": "estimate"}}


def _w():
    return _windows([_traj(), _traj(seed=1, circle=True)])


def _lost(w, frames, corrupt=10.0):
    """Mark `frames` lost in every window and corrupt them (the result must not depend on them)."""
    valid = np.ones(w.obs_p.shape[:2], bool)
    valid[:, frames] = False
    obs_p = w.obs_p.copy()
    obs_p[:, frames] += corrupt
    return dataclasses.replace(w, obs_p=obs_p, obs_valid=valid)


def test_default_bit_identical():
    w = _w()
    a = predict_all(w, PARAMS, CFG)
    b = predict_all(dataclasses.replace(w, obs_valid=np.ones(w.obs_p.shape[:2], bool)), PARAMS, CFG)
    assert a.keys() == b.keys()
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_cv_ignores_lost_last_frame():
    w = _windows([_traj()])                                    # uniform motion: CV-2 is exact
    w2 = _lost(w, [-1])
    pred = predict_gapped(w2, lambda s: cv(s.obs_t, s.obs_p, s.fut_t, min(2, len(s.obs_t))))
    np.testing.assert_allclose(pred, w.fut_p, atol=1e-9)


def test_kalman_skips_lost_frames():
    w = _windows([_traj()])
    w2 = _lost(w, [3, 7, -1])
    pred = predict_gapped(w2, lambda s: kalman_predict(s.obs_t, s.obs_p, s.fut_t, "ca", 1e3, 1e-6))
    np.testing.assert_allclose(pred, w.fut_p, atol=1e-4)


def test_predict_all_ignores_corrupted_lost_frames():
    w = _windows([_traj()])
    clean = predict_all(_lost(w, [4, -1], corrupt=0.0), PARAMS, CFG)
    dirty = predict_all(_lost(w, [4, -1], corrupt=10.0), PARAMS, CFG)
    for k in clean:
        np.testing.assert_allclose(clean[k], dirty[k], atol=1e-9, err_msg=k)


def test_few_valid_frames_fall_back():
    # only 2 valid frames: CA, CTR and acro cannot fit, predict_all still returns finite predictions
    w = _windows([_traj()])
    valid = np.zeros(w.obs_p.shape[:2], bool)
    valid[:, [-3, -1]] = True
    out = predict_all(dataclasses.replace(w, obs_valid=valid), PARAMS, CFG)
    assert all(np.isfinite(v).all() for v in out.values())
    np.testing.assert_allclose(out["ca"], w.fut_p, atol=1e-9)    # CV through the two valid frames, uniform motion


def test_k_clamped_to_valid_count():
    w = _windows([_traj()])
    valid = np.zeros(w.obs_p.shape[:2], bool)
    valid[:, -4:] = True
    pred = predict_gapped(dataclasses.replace(w, obs_valid=valid),
                          lambda s: ca(s.obs_t, s.obs_p, s.fut_t, min(10, len(s.obs_t))))
    np.testing.assert_allclose(pred, w.fut_p, atol=1e-9)


def test_mixed_patterns_keep_window_order():
    w = _w()
    valid = np.ones(w.obs_p.shape[:2], bool)
    valid[::2, -1] = False                                    # every other window loses its last frame
    w2 = dataclasses.replace(w, obs_valid=valid)
    pred = predict_gapped(w2, lambda s: cv(s.obs_t, s.obs_p, s.fut_t, 2))
    ref_odd = cv(w.obs_t, w.obs_p[1::2], w.fut_t, 2)
    ref_even = cv(w.obs_t[:-1], w.obs_p[::2, :-1], w.fut_t, 2)
    np.testing.assert_array_equal(pred[1::2], ref_odd)
    np.testing.assert_array_equal(pred[::2], ref_even)


def test_too_few_valid_raises():
    w = _windows([_traj()])
    valid = np.zeros(w.obs_p.shape[:2], bool)
    valid[:, -1] = True
    with pytest.raises(ValueError):
        predict_gapped(dataclasses.replace(w, obs_valid=valid), lambda s: cv(s.obs_t, s.obs_p, s.fut_t, 2))


def test_drop_sparse():
    w = _windows([_traj()])
    valid = np.ones(w.obs_p.shape[:2], bool)
    valid[0, :-1] = False
    kept, m = drop_sparse(dataclasses.replace(w, obs_valid=valid))
    assert len(kept) == len(w) - 1 and not m[0] and m[1:].all()
    assert kept.obs_valid.shape == (len(w) - 1, w.obs_p.shape[1])
    np.testing.assert_array_equal(kept.obs_p, w.obs_p[1:])


def test_last_valid_index():
    v = np.array([[True, True, False], [True, False, False], [True, True, True]])
    np.testing.assert_array_equal(last_valid_index(v), [1, 0, 2])


def test_tune_with_gaps_runs():
    w = _lost(_windows([_traj(chaos=True), _traj(chaos=True, seed=3)]), [5, -1])
    p = tune(w, CFG)
    assert p["cv_k"] in CFG.cv_k_grid and np.isfinite(p["scores"]["kf_ca"])


def _random_mask(w, p_lost=0.15, seed=0):
    v = np.random.default_rng(seed).random(w.obs_p.shape[:2]) > p_lost
    v[:, -3:] |= np.random.default_rng(seed + 1).random((len(w), 3)) > 0.5   # keep >= a few valid frames
    v[:, -1] |= ~v[:, :-1].any(1)
    v[v.sum(1) < 3, -3:] = True
    return dataclasses.replace(w, obs_valid=v)


def _per_pattern(w, fn):
    """Reference: one call per full pattern of valid frames, no last-k grouping, no batched rollout."""
    out = np.empty(w.fut_p.shape)
    pats, inv = np.unique(w.obs_valid, axis=0, return_inverse=True)
    for j, pat in enumerate(pats):
        sel = np.flatnonzero(inv.reshape(-1) == j)
        out[sel] = fn(restrict_frames(w.subset(sel), pat))
    return out


def test_acro_g_matches_per_pattern_reference():
    w = _random_mask(_windows([_traj(chaos=True, seed=s) for s in range(4)]))
    got = acro_g(w, 5, 8.0, "hold_vz", "estimate")
    ref = _per_pattern(w, lambda s: acro_predict(s.obs_t, s.obs_p, np.asarray(s.obs_R, float), s.fut_t,
                                                 min(5, len(s.obs_t)), 8.0, "hold_vz", "estimate"))
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_last_k_grouping_matches_full_patterns():
    w = _random_mask(_windows([_traj(chaos=True, seed=s) for s in range(4)]), seed=3)
    for k in (2, 3, 5):
        np.testing.assert_allclose(cv_g(w, k), _per_pattern(w, lambda s: cv(s.obs_t, s.obs_p, s.fut_t,
                                                                            min(k, len(s.obs_t)))), atol=1e-12)
    np.testing.assert_allclose(ca_g(w, 5), _per_pattern(w, lambda s: ca(s.obs_t, s.obs_p, s.fut_t,
                                                                        min(5, len(s.obs_t)))), atol=1e-12)


def test_last_k_valid():
    v = np.array([[True, False, True, True, False, True]])
    np.testing.assert_array_equal(last_k_valid(v, 2), [[False, False, False, True, False, True]])
    np.testing.assert_array_equal(last_k_valid(v, None), v)
