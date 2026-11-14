import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.pipeline.choice import WindowChooser, candidates, choose_offline
from dronepose.pose.global_disambiguation import GlobalParams, disambiguate_global
from dronepose.pose.pnp import PnPSolution


def _seq(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        R = Rotation.from_euler("zyx", [2.0 * i, 10 + np.sin(i / 5), 5], degrees=True).as_matrix()
        F = Rotation.from_euler("x", 150, degrees=True).as_matrix() @ R
        e_true, e_flip = rng.uniform(1, 4), rng.uniform(1, 4)
        sols = [PnPSolution(R, np.array([0, 0, 10.0]), e_true), PnPSolution(F, np.array([0, 0, 10.0]), e_flip)]
        if i == 12:
            sols = []                                        # lost frame
        out.append(sorted(sols, key=lambda s: s.reproj_px))
    return out


def _key(c):
    return None if c.rejected else c.index


@pytest.mark.parametrize("W", [1, 15, 120])
def test_window_equals_global_on_prefix(W):
    cands = _seq()
    ch = WindowChooser(W)
    for t in range(len(cands)):
        emitted, _ = ch.step(cands[t], t + 1)
        lo = max(0, t - W + 1)
        ref = disambiguate_global(cands[lo:t + 1], GlobalParams(), list(range(lo + 1, t + 2)))
        assert _key(emitted) == _key(ref[-1]), t


def test_large_window_last_frame_equals_offline():
    cands = _seq()
    ch = WindowChooser(1000)
    for t in range(len(cands)):
        emitted, _ = ch.step(cands[t], t + 1)
    assert _key(emitted) == _key(choose_offline(cands, list(range(1, len(cands) + 1)))[-1])


def test_w1_picks_lowest_reproj_under_20px():
    R = np.eye(3)
    a, b = PnPSolution(R, np.zeros(3), 3.0), PnPSolution(R, np.zeros(3), 5.0)
    emitted, _ = WindowChooser(1).step([a, b], 1)
    assert emitted.solution is a
    emitted, _ = WindowChooser(1).step([PnPSolution(R, np.zeros(3), 25.0), PnPSolution(R, np.zeros(3), 30.0)], 1)
    assert emitted.rejected


def test_redecided_length():
    cands = _seq()
    ch = WindowChooser(60)
    for t in range(5):
        _, red = ch.step(cands[t], t + 1)
    assert len(red) == 5
    for t in range(5, 30):
        _, red = ch.step(cands[t], t + 1)
    assert len(red) == 15


def test_candidates_none_model():
    assert candidates(np.zeros((4, 3)), np.eye(3), None) == []
    k = np.full((4, 3), np.nan)
    from dronepose.data.drone_models import load_drone_models
    from pathlib import Path
    m = load_drone_models(Path("configs/drone_models.json"))["01"]
    assert candidates(k, np.eye(3), m) == []


def _tilted_pair():
    from dronepose.eval.units import UP_CAM
    up_R = Rotation.align_vectors([UP_CAM], [[0, 0, 1.0]])[0].as_matrix()          # body z = camera up
    tilt_R = Rotation.align_vectors([[0, 0, 1.0]], [[0, 0, 1.0]])[0].as_matrix()    # body z = camera z: 90 deg from up
    assert np.allclose(up_R[:, 2], UP_CAM)
    t = np.array([0, 0, 10.0])
    return PnPSolution(up_R, t, 4.0), PnPSolution(tilt_R, t, 1.0), UP_CAM


def test_gravity_excludes_tilted_candidate():
    ok, tilted, up = _tilted_pair()
    cands = [sorted([ok, tilted], key=lambda s: s.reproj_px)]
    assert choose_offline(cands, [1])[0].solution is tilted
    assert choose_offline(cands, [1], up=up)[0].solution is ok
    assert WindowChooser(5).step(cands[0], 1)[0].solution is tilted
    assert WindowChooser(5, up=up).step(cands[0], 1)[0].solution is ok


@pytest.mark.parametrize("W", [1, 15, 120])
def test_window_with_gravity_equals_global_on_prefix(W):
    from dronepose.eval.units import UP_CAM
    cands = _seq()
    ch = WindowChooser(W, up=UP_CAM)
    for t in range(len(cands)):
        emitted, _ = ch.step(cands[t], t + 1)
        lo = max(0, t - W + 1)
        ref = disambiguate_global(cands[lo:t + 1], GlobalParams(), list(range(lo + 1, t + 2)), up=UP_CAM)
        assert _key(emitted) == _key(ref[-1]), t
