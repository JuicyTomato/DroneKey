import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.common.geometry import geodesic_deg, rot_z
from dronepose.pose.disambiguation import STRATEGIES, TemporalParams, disambiguate, reprojection_ratio
from dronepose.pose.pnp import PnPSolution

FLIP = Rotation.from_euler("x", 180, degrees=True).as_matrix()   # the "other" IPPE solution, far from the truth


def truth(k):
    return rot_z(2.0 * k)                                          # slow, smooth rotation


def frame(k, rep_true, rep_wrong):
    sols = [PnPSolution(truth(k), np.zeros(3), rep_true), PnPSolution(truth(k) @ FLIP, np.zeros(3), rep_wrong)]
    return sorted(sols, key=lambda s: s.reproj_px)


def picked_true(choices, k):
    return geodesic_deg(choices[k].solution.R, truth(k)) < 1e-3


def flip_block():
    # frames 8..12: reprojection slightly favours the wrong solution (ratio 1.5), elsewhere the true one (ratio 2)
    return [frame(k, 1.5, 1.0) if 8 <= k <= 12 else frame(k, 1.0, 2.0) for k in range(20)]


def test_reproj_follows_reprojection_and_flips():
    ch = disambiguate(flip_block(), "reproj")
    assert [picked_true(ch, k) for k in range(20)] == [not (8 <= k <= 12) for k in range(20)]


def test_temporal_keeps_the_true_solution_through_the_block():
    ch = disambiguate(flip_block(), "temporal")
    assert all(picked_true(ch, k) for k in range(20))
    assert not any(c.rejected for c in ch)


def test_wrong_start_is_recovered_by_reanchoring():
    # frames 0-1: wrong solution slightly better; from frame 2 the true one is clearly better (ratio 5)
    cands = [frame(k, 1.4, 1.0) if k < 2 else frame(k, 1.0, 5.0) for k in range(10)]
    ch = disambiguate(cands, "temporal", TemporalParams(reanchor_ratio=3.0, reanchor_frames=3))
    assert [picked_true(ch, k) for k in range(10)] == [False, False, False, False] + [True] * 6


def test_long_gap_forgets_the_reference():
    cands = [frame(k, 1.0, 2.0) for k in range(5)] + [[] for _ in range(7)] + [frame(12, 1.2, 1.0)]
    ch = disambiguate(cands, "temporal", TemporalParams(max_gap=5))
    assert ch[5].reason == "no_solution" and ch[5].index is None
    assert not picked_true(ch, 12)              # reference forgotten -> back to reprojection (wrong here)
    short = [frame(k, 1.0, 2.0) for k in range(5)] + [[] for _ in range(3)] + [frame(8, 1.2, 1.0)]
    assert picked_true(disambiguate(short, "temporal", TemporalParams(max_gap=5)), 8)


def test_reject_when_both_solutions_jump_away():
    side = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    cands = [frame(k, 1.0, 2.0) for k in range(10)]
    for k in (10, 11, 12):                      # both candidates ~90 deg from the previous accepted pose
        cands.append([PnPSolution(truth(k) @ side, np.zeros(3), 1.0),
                      PnPSolution(truth(k) @ side.T, np.zeros(3), 1.3)])
    cands.append(frame(13, 1.0, 2.0))
    ch = disambiguate(cands, "temporal_reject", TemporalParams(max_jump_deg=20.0))
    assert [ch[k].reason for k in (10, 11, 12)] == ["jump"] * 3
    assert all(ch[k].rejected for k in (10, 11, 12))
    assert not ch[13].rejected and picked_true(ch, 13)     # 3 frames later: allowed jump 80 deg, true is ~8 deg away


def test_reject_on_large_reprojection():
    cands = [frame(0, 1.0, 2.0), frame(1, 50.0, 60.0)]
    ch = disambiguate(cands, "temporal_reject", TemporalParams(max_reproj_px=20.0))
    assert ch[1].rejected and ch[1].reason == "reproj"
    assert not disambiguate(cands, "temporal")[1].rejected


def test_single_and_missing_solutions():
    one = [[PnPSolution(np.eye(3), np.zeros(3), 1.0)], []]
    ch = disambiguate(one, "temporal")
    assert ch[0].index == 0 and math.isinf(ch[0].ratio)
    assert ch[1].rejected and ch[1].reason == "no_solution"
    assert math.isinf(reprojection_ratio([]))


def test_unknown_strategy():
    with pytest.raises(ValueError):
        disambiguate([], "magic")


def test_global_strategy_is_dispatched_with_its_own_params():
    from dronepose.pose.global_disambiguation import GlobalParams

    assert "global" in STRATEGIES
    cands = flip_block()
    assert all(picked_true(disambiguate(cands, "global"), k) for k in range(20))     # default params
    assert all(picked_true(disambiguate(cands, "global", GlobalParams()), k) for k in range(20))
    with pytest.raises(TypeError):
        disambiguate(cands, "global", TemporalParams())
    with pytest.raises(TypeError):
        disambiguate(cands, "temporal", GlobalParams())


def test_existing_strategies_ignore_frame_idx_and_up():
    cands = flip_block()
    plain = [c.index for c in disambiguate(cands, "temporal")]
    assert [c.index for c in disambiguate(cands, "temporal", frame_idx=list(range(20)), up=[0, 0, 1])] == plain
