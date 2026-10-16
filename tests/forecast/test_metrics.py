import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.forecast.metrics import (ade_at, fde_at, geodesic_deg_batch, metric_group, paired_bootstrap_skill,
                                        position_errors, skill, up_angle_deg)

B = 200   # tests keep the bootstrap small; scripts use the configured 10 000


def test_position_errors_shape():
    pred, gt = np.zeros((2, 4, 3)), np.ones((2, 4, 3))
    np.testing.assert_allclose(position_errors(pred, gt), np.full((2, 4), np.sqrt(3)))


def test_ade_fde_hand_case():
    err = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert ade_at(err, 2)[0] == 1.5 and fde_at(err, 4)[0] == 4 and fde_at(err, 1)[0] == 1


def test_geodesic_and_up_angle():
    Rz = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    np.testing.assert_allclose(geodesic_deg_batch(Rz[None, None], np.eye(3)[None, None]), [[30]])
    t = np.radians(10)
    np.testing.assert_allclose(up_angle_deg(np.array([0, 0, 1.0]), np.array([np.sin(t), 0, np.cos(t)])), 10)
    assert np.isfinite(geodesic_deg_batch(np.eye(3), np.eye(3)))


def test_skill():
    assert skill(np.array([1.0, 1.0]), np.array([2.0, 2.0])) == 0.5


def _data(seed=0):
    rng = np.random.default_rng(seed)
    traj = np.repeat(np.arange(6), 10)
    strata = np.where(traj < 3, "03", "14")
    ref = rng.uniform(1, 2, size=60)
    return ref, traj, strata


def test_bootstrap_same_method_is_zero():
    ref, traj, strata = _data()
    assert paired_bootstrap_skill(ref, ref, traj, strata, B, 0) == (0.0, 0.0, 0.0)


def test_bootstrap_reproducible_with_seed():
    ref, traj, strata = _data()
    m = ref * np.random.default_rng(1).uniform(0.5, 1.2, size=60)
    a = paired_bootstrap_skill(m, ref, traj, strata, B, 7)
    assert a == paired_bootstrap_skill(m, ref, traj, strata, B, 7)
    assert a[1] <= a[0] <= a[2] and a[1] < a[2]


def test_bootstrap_single_traj_per_stratum_is_degenerate():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    m = np.array([0.5, 1.0, 3.0, 2.0])
    point, lo, hi = paired_bootstrap_skill(m, ref, np.array([0, 0, 1, 1]), np.array(["a", "a", "b", "b"]), B, 0)
    assert np.isclose(point, 1 - 6.5 / 10) and np.isclose(lo, point) and np.isclose(hi, point)


def test_metric_group_excludes_non_metric():
    class W:
        metric = np.array([True, False, True])
    np.testing.assert_array_equal(metric_group(W), [True, False, True])
