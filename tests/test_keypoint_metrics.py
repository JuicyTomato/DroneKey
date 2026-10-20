import numpy as np
import pytest

from dronepose.eval.keypoint_metrics import frame_px_error, identity_ok, oks, oks_ap, pck, success_rate
from dronepose.keypoints.predictors import no_detection


def test_frame_px_error_median_of_four():
    gt = np.zeros((4, 2)); pred = np.array([[1, 0, .9], [2, 0, .9], [3, 0, .9], [10, 0, .9]], float)
    assert frame_px_error(pred, gt) == 2.5


def test_frame_px_error_missing_is_none():
    assert frame_px_error(no_detection(), np.zeros((4, 2))) is None


def test_pck_counts_keypoints_and_missing_fail():
    gt = np.zeros((4, 2))
    good = np.array([[1, 0, 1], [1, 0, 1], [6, 0, 1], [20, 0, 1]], float)
    assert pck([good, no_detection()], [gt, gt], [2, 10]) == {2: 2 / 8, 10: 3 / 8}


def test_pck_relative_threshold_uses_scale():
    gt = np.zeros((4, 2)); pred = np.c_[np.full(4, 3.0), np.zeros(4), np.ones(4)]
    assert pck([pred], [gt], [0.05], scales=[100.0]) == {0.05: 1.0}


def test_identity_detects_swap():
    gt = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert identity_ok(np.c_[gt, np.ones(4)], gt) is True
    assert identity_ok(np.c_[gt[[1, 0, 2, 3]], np.ones(4)], gt) is False


def test_oks_formula():
    gt = np.zeros((4, 2)); pred = np.c_[np.full(4, 2.0), np.zeros(4), np.ones(4)]
    assert oks(pred, gt, area=100.0) == pytest.approx(np.exp(-4 / 20))
    assert oks(no_detection(), gt, area=100.0) == 0.0


def test_success_rate():
    assert success_rate([0.95, 0.91, 0.5, 0.0], 0.90) == 0.5


def test_oks_ap_perfect_and_missing():
    assert oks_ap([1.0, 1.0], [0.9, 0.8]) == pytest.approx(1.0)
    assert oks_ap([1.0, 0.0], [0.9, float("-inf")]) == pytest.approx(0.5, abs=0.01)


def test_missing_everywhere_does_not_crash():
    assert identity_ok(no_detection(), np.zeros((4, 2))) is None
    assert oks_ap([0.0], [float("-inf")]) == 0.0
    assert oks_ap([], []) == 0.0
    assert pck([], [], [2]) == {2: 0.0}


def test_oks_ap_false_positive_ranked_first_lowers_precision():
    # FP first (precision 0.5 at recall 0.5): AP ~ 0.5 * 0.505
    assert oks_ap([0.0, 1.0], [0.9, 0.8]) == pytest.approx(0.2525, abs=0.01)
