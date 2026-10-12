import numpy as np
import pytest

from dronepose.trajectory.baselines import constant_velocity


def test_exact_on_uniform_straight_motion():
    t = np.arange(15) / 30
    p = np.stack([1 + 0.5 * t, -2 * t, 3 + 0 * t], axis=1)
    tf = 14 / 30 + np.arange(1, 31) / 30
    expected = np.stack([1 + 0.5 * tf, -2 * tf, 3 + 0 * tf], axis=1)
    np.testing.assert_allclose(constant_velocity(t, p, tf), expected, atol=1e-12)


def test_handles_gaps_in_time():
    t = np.array([0, 1, 4, 5]) / 30                          # frames 2 and 3 missing
    p = (t * 2.0)[:, None]
    np.testing.assert_allclose(constant_velocity(t, p, np.array([6 / 30])), [[12 / 30]], atol=1e-12)


def test_single_point_repeats_it():
    np.testing.assert_allclose(constant_velocity(np.array([0.0]), np.array([[1.0, 2.0]]), np.array([0.1, 0.2])),
                               [[1.0, 2.0], [1.0, 2.0]])


def test_no_history_raises():
    with pytest.raises(ValueError):
        constant_velocity(np.zeros(0), np.zeros((0, 3)), np.array([0.1]))
