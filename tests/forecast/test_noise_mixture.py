import numpy as np
import pytest

from dronepose.forecast.noise import add_pose_noise, add_pose_noise_per_window, noise_rng, sample_mixture_sigmas
from dronepose.forecast.windows import make_windows
from tests.forecast.test_tuning import CFG, _traj


@pytest.fixture(scope="module")
def windows():
    return make_windows([_traj(T=48, chaos=True, seed=7), _traj(T=48, circle=True, seed=8)], CFG.obs_len,
                        CFG.pred_len, CFG.fps)


def test_per_window_matches_scalar_when_constant(windows):
    W = len(windows)
    a = add_pose_noise(windows, 2.0, 1.0, noise_rng(3, 1.0))
    b = add_pose_noise_per_window(windows, np.full(W, 2.0), np.full(W, 1.0), noise_rng(3, 1.0))
    np.testing.assert_array_equal(a.obs_p, b.obs_p)
    np.testing.assert_array_equal(a.obs_R, b.obs_R)
    np.testing.assert_array_equal(a.obs_u, b.obs_u)
    assert b.fut_p is windows.fut_p


def test_per_window_sigma_is_per_window(windows):
    W = len(windows)
    sp = np.zeros(W)
    sp[0] = 5.0
    b = add_pose_noise_per_window(windows, sp, np.zeros(W), noise_rng(0, 0))
    assert np.abs(b.obs_p[0] - windows.obs_p[0]).max() > 1e-3
    np.testing.assert_array_equal(b.obs_p[1:], windows.obs_p[1:])


def test_per_window_zero_sigma_is_identity(windows):
    W = len(windows)
    b = add_pose_noise_per_window(windows, np.zeros(W), np.zeros(W), noise_rng(0, 0))
    np.testing.assert_array_equal(b.obs_p, windows.obs_p)
    np.testing.assert_allclose(b.obs_R, windows.obs_R, atol=1e-12)


def test_mixture_sigmas_ranges():
    sp, sr = sample_mixture_sigmas(20000, np.random.default_rng(0))
    assert sp.shape == sr.shape == (20000,)
    assert sp.min() >= 0 and sp.max() <= 5 and sr.min() >= 0 and sr.max() <= 3
    assert abs(sp.mean() - 2.5) < 0.05 and abs(sr.mean() - 1.5) < 0.05
    assert abs(np.corrcoef(sp, sr)[0, 1]) < 0.05          # independent
