import numpy as np
import pytest

from dronepose.forecast.baselines import ca, constant_position, cv, fit_state, poly_fit_predict
from dronepose.trajectory.baselines import constant_velocity

N, H = 15, 30
OBS_T = (np.arange(N) - (N - 1)) / 30
FUT_T = np.arange(1, H + 1) / 30
ALL_T = np.concatenate([OBS_T, FUT_T])
RNG = np.random.default_rng(0)


def _motion(W, acc=False):
    p0, v0 = RNG.normal(size=(W, 1, 3)), RNG.normal(size=(W, 1, 3))
    a0 = RNG.normal(size=(W, 1, 3)) if acc else 0
    P = p0 + v0 * ALL_T[None, :, None] + 0.5 * a0 * ALL_T[None, :, None] ** 2
    return P[:, :N], P[:, N:]


@pytest.mark.parametrize("k", [2, 3, 5, 10, 15])
def test_cv_exact_on_uniform_straight_motion(k):
    obs, fut = _motion(4)
    np.testing.assert_allclose(cv(OBS_T, obs, FUT_T, k), fut, atol=1e-10)


@pytest.mark.parametrize("k", [3, 5, 10, 15])
def test_ca_exact_on_constant_acceleration(k):
    obs, fut = _motion(4, acc=True)
    np.testing.assert_allclose(ca(OBS_T, obs, FUT_T, k), fut, atol=1e-9)


def test_constant_position_repeats_last():
    obs, _ = _motion(3)
    pred = constant_position(OBS_T, obs, FUT_T)
    assert pred.shape == (3, H, 3)
    np.testing.assert_allclose(pred, np.repeat(obs[:, -1:], H, axis=1))


def test_fit_rejects_k_larger_than_window():
    obs, _ = _motion(2)
    with pytest.raises(ValueError):
        cv(OBS_T[-10:], obs[:, -10:], FUT_T, k=15)
    with pytest.raises(ValueError):
        poly_fit_predict(OBS_T, obs, FUT_T, k=2, degree=2)


def test_vectorized_equals_loop():
    obs = RNG.normal(size=(7, N, 3))
    batch = ca(OBS_T, obs, FUT_T, 5)
    for i in range(7):
        np.testing.assert_allclose(batch[i], ca(OBS_T, obs[i:i + 1], FUT_T, 5)[0], atol=1e-12)


def test_cv_matches_skeleton_baseline():
    obs = RNG.normal(size=(1, N, 3))
    np.testing.assert_allclose(cv(OBS_T, obs, FUT_T, N)[0], constant_velocity(OBS_T, obs[0], FUT_T), atol=1e-12)


def test_fit_state_returns_last_position_and_velocity():
    obs, _ = _motion(3)
    p0, v0 = fit_state(OBS_T, obs, 5, 1)
    np.testing.assert_allclose(p0, obs[:, -1], atol=1e-12)
    np.testing.assert_allclose(v0, (obs[:, -1] - obs[:, -2]) * 30, atol=1e-10)
