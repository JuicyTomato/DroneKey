import numpy as np

from dronepose.forecast.config import load_config
from dronepose.forecast.kalman import kalman_predict

N, H = 15, 30
OBS_T = (np.arange(N) - (N - 1)) / 30
FUT_T = np.arange(1, H + 1) / 30
ALL_T = np.concatenate([OBS_T, FUT_T])


def _motion(W, acc=False, seed=0):
    rng = np.random.default_rng(seed)
    p0, v0 = rng.normal(size=(W, 1, 3)), rng.normal(size=(W, 1, 3))
    a0 = rng.normal(size=(W, 1, 3)) if acc else 0
    P = p0 + v0 * ALL_T[None, :, None] + 0.5 * a0 * ALL_T[None, :, None] ** 2
    return P[:, :N], P[:, N:]


def test_kf_cv_exact_on_uniform_motion():
    obs, fut = _motion(4)
    np.testing.assert_allclose(kalman_predict(OBS_T, obs, FUT_T, "cv", q=1e-12, r=1e-12), fut, atol=1e-8)


def test_kf_ca_exact_on_constant_acceleration():
    obs, fut = _motion(4, acc=True)
    np.testing.assert_allclose(kalman_predict(OBS_T, obs, FUT_T, "ca", q=1e-12, r=1e-12), fut, atol=1e-6)


def test_per_window_noise_shape_accepted():
    obs = np.random.default_rng(1).normal(size=(5, N, 3))
    scalar = kalman_predict(OBS_T, obs, FUT_T, "ca", q=1.0, r=1e-4)
    full = kalman_predict(OBS_T, obs, FUT_T, "ca", q=1.0, r=np.full((5, N, 3), 1e-4))
    np.testing.assert_allclose(full, scalar, atol=1e-12)


def test_per_axis_noise_changes_only_that_axis():
    obs = np.random.default_rng(2).normal(size=(3, N, 3))
    base = kalman_predict(OBS_T, obs, FUT_T, "cv", q=1.0, r=1e-4)
    r = np.full((3, N, 3), 1e-4)
    r[..., 2] = 1e-1
    out = kalman_predict(OBS_T, obs, FUT_T, "cv", q=1.0, r=r)
    np.testing.assert_allclose(out[..., :2], base[..., :2], atol=1e-12)
    assert np.abs(out[..., 2] - base[..., 2]).max() > 1e-6


def test_finite_over_tuning_grid():
    cfg = load_config()
    obs = np.cumsum(np.random.default_rng(3).normal(scale=0.05, size=(6, N, 3)), axis=1)
    for model, qs in (("cv", cfg.kf_cv_q_grid), ("ca", cfg.kf_ca_q_grid)):
        for q in qs:
            for r in cfg.kf_r_grid:
                assert np.isfinite(kalman_predict(OBS_T, obs, FUT_T, model, q=q, r=r)).all(), (model, q, r)


def test_large_r_tends_to_smoother_prediction():
    obs, fut = _motion(200, seed=4)
    noisy = obs + np.random.default_rng(5).normal(scale=0.05, size=obs.shape)
    fde = lambda r: np.linalg.norm(kalman_predict(OBS_T, noisy, FUT_T, "cv", q=1e-3, r=r)[:, -1] - fut[:, -1], axis=-1).mean()
    assert fde(1e-2) < fde(1e-6)
