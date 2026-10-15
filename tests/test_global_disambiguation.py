import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.common.geometry import geodesic_deg, rot_z
from dronepose.pose.disambiguation import disambiguate
from dronepose.pose.global_disambiguation import GlobalParams, disambiguate_global
from dronepose.pose.pnp import PnPSolution

FLIP = Rotation.from_euler("x", 180, degrees=True).as_matrix()   # the "other" IPPE solution, far from the truth


def truth(k):
    return rot_z(2.0 * k)                                          # slow, smooth rotation, 2 deg per frame


def frame(k, rep_true, rep_wrong):
    sols = [PnPSolution(truth(k), np.zeros(3), rep_true), PnPSolution(truth(k) @ FLIP, np.zeros(3), rep_wrong)]
    return sorted(sols, key=lambda s: s.reproj_px)


def picked_true(choices, k):
    c = choices[k]
    return (not c.rejected) and geodesic_deg(c.solution.R, truth(k)) < 1e-3


def test_reprojection_block_favouring_the_wrong_solution_is_overruled():
    cands = [frame(k, 1.5, 1.0) if 8 <= k <= 12 else frame(k, 1.0, 2.0) for k in range(20)]
    ch = disambiguate_global(cands)
    assert all(picked_true(ch, k) for k in range(20))
    assert not any(c.rejected for c in ch)


def test_coin_flip_at_the_start_is_settled_by_later_frames():
    # frames 0-1: wrong solution slightly better; from frame 2 on the true one is clearly better
    cands = [frame(k, 1.4, 1.0) if k < 2 else frame(k, 1.0, 5.0) for k in range(10)]
    assert all(picked_true(disambiguate_global(cands), k) for k in range(10))
    greedy = disambiguate(cands, "temporal")
    assert not picked_true(greedy, 0)                          # the greedy rule starts on the wrong one


def burst_sequence(n_burst=14):
    # frames 0-9 fine; n_burst frames with reprojection far above the limit; then ratio-1.09 frames where reprojection
    # prefers the WRONG solution (the real failure of temporal_reject on 14_02_07_000)
    cands = [frame(k, 1.0, 2.0) for k in range(10)]
    cands += [frame(k, 40.0, 45.0) for k in range(10, 10 + n_burst)]
    cands += [frame(k, 1.09, 1.0) for k in range(10 + n_burst, 10 + n_burst + 16)]
    return cands


def test_long_burst_of_rejections_does_not_lose_the_reference():
    cands = burst_sequence(14)
    ch = disambiguate_global(cands)
    assert [c.reason for c in ch[10:24]] == ["reproj"] * 14
    assert all(picked_true(ch, k) for k in list(range(10)) + list(range(24, 40)))
    greedy = disambiguate(cands, "temporal_reject")             # documents the defect this strategy fixes
    assert not picked_true(greedy, 30)


def test_burst_longer_than_max_span_restarts_without_crashing():
    cands = burst_sequence(14)
    ch = disambiguate_global(cands, GlobalParams(max_span=10))
    assert len(ch) == 40 and all(c.reason == "reproj" for c in ch[10:24])
    assert not ch[24].rejected                                   # the chain restarts after the burst
    assert not ch[9].rejected                                    # and the part before the burst is kept


def test_frames_without_solutions_or_with_one_solution():
    one = [PnPSolution(truth(0), np.zeros(3), 1.0)]
    cands = [one, [], frame(2, 1.0, 2.0), [PnPSolution(truth(3) @ FLIP, np.zeros(3), 30.0)], frame(4, 1.0, 2.0)]
    ch = disambiguate_global(cands)
    assert ch[1].rejected and ch[1].reason == "no_solution" and math.isinf(ch[1].ratio)
    assert ch[0].index == 0 and math.isinf(ch[0].ratio) and not ch[0].rejected
    assert ch[3].rejected and ch[3].reason == "reproj"
    json.dumps([None if math.isinf(c.ratio) else c.ratio for c in ch], allow_nan=False)


def test_all_frames_without_solution():
    ch = disambiguate_global([[] for _ in range(6)])
    assert all(c.rejected and c.reason == "no_solution" for c in ch)


def test_real_time_between_frames_is_used():
    cands = [frame(30 * k, 1.0, 2.0) for k in range(10)]        # 60 deg of rotation per record
    spaced = disambiguate_global(cands, frame_idx=[30 * k for k in range(10)])
    assert not any(c.rejected for c in spaced)                   # 60 deg / (5 deg * 30 frames) is a normal step
    squeezed = disambiguate_global(cands)                        # same records taken as consecutive: 60 deg per frame
    assert sum(c.rejected for c in squeezed) > 0
    with pytest.raises(ValueError):
        disambiguate_global(cands, frame_idx=[0, 1, 1, 3, 4, 5, 6, 7, 8, 9])


def test_gravity_gate_only_with_up():
    up = np.array([0.0, 0.0, 1.0])                               # body up = R[:, 2]
    upright = PnPSolution(np.eye(3), np.zeros(3), 2.0)
    upside_down = PnPSolution(FLIP, np.zeros(3), 1.0)            # better reprojection but z axis points down
    cands = [[upside_down, upright] for _ in range(5)]
    free = disambiguate_global(cands)
    gated = disambiguate_global(cands, up=up)
    assert all(c.index == 0 for c in free)                       # no gravity: lowest reprojection
    assert all(c.index == 1 for c in gated)                      # gravity on: the upright one
    assert [c.index for c in disambiguate_global(cands, up=None)] == [c.index for c in free]
    with pytest.raises(ValueError):
        disambiguate_global(cands, up=np.zeros(3))


def test_small_inputs_and_determinism():
    assert disambiguate_global([]) == []
    one = disambiguate_global([frame(0, 1.0, 2.0)])
    assert one[0].index == 0 and not one[0].rejected
    cands = [frame(k, 1.4, 1.0) if k < 2 else frame(k, 1.0, 5.0) for k in range(10)]
    a, b = disambiguate_global(cands), disambiguate_global(cands)
    assert [c.index for c in a] == [c.index for c in b]


def test_unambiguous_sequence_matches_reprojection():
    cands = [frame(k, 1.0, 9.0) for k in range(30)]
    assert [c.index for c in disambiguate_global(cands)] == [c.index for c in disambiguate(cands, "reproj")]
