from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.common.geometry import geodesic_deg, project
from dronepose.data.drone_models import load_drone_models
from dronepose.pose.pnp import solve_ippe
from dronepose.pose.solvers import SOLVERS, solve
from tests.fixtures.fake_dataset import FAKE_K

MODEL = load_drone_models(Path(__file__).resolve().parents[1] / "configs" / "drone_models.json")["01"].kpts_flu
# The 4 propellers are near-coplanar (z within +-0.02 of the plane); exact recovery to 1e-4 deg needs an exactly
# planar model for IPPE (planar approximation) and EPnP (inaccurate/ambiguous on 4 near-planar points).
PLANAR = MODEL.copy()
PLANAR[:, 2] = 0.0
R_TRUE = Rotation.from_euler("x", 20, degrees=True).as_matrix()
T_TRUE = np.array([0.1, -0.05, 3.0])
IMG = project(FAKE_K, (R_TRUE @ MODEL.T).T + T_TRUE)
IMG_PLANAR = project(FAKE_K, (R_TRUE @ PLANAR.T).T + T_TRUE)


def test_ippe_matches_solve_ippe():
    ref, got = solve_ippe(MODEL, IMG, FAKE_K), solve("ippe", MODEL, IMG, FAKE_K)
    assert len(ref) == len(got) == 2
    for a, b in zip(ref, got):
        assert np.allclose(a.R, b.R, atol=1e-12)
        assert np.allclose(a.t, b.t, atol=1e-12)
        assert np.allclose(a.reproj_px, b.reproj_px, atol=1e-12)


@pytest.mark.parametrize("refine", [False, True])
@pytest.mark.parametrize("name", SOLVERS)
def test_exact_keypoints_every_solver(name, refine):
    sols = solve(name, PLANAR, IMG_PLANAR, FAKE_K, refine=refine)
    assert sols
    assert any(geodesic_deg(s.R, R_TRUE) < 1e-4 and np.linalg.norm(s.t - T_TRUE) < 1e-6 for s in sols)


@pytest.mark.parametrize("name", ["sqpnp", "iterative"])
def test_non_planar_model_recovered_by_general_solvers(name):
    sols = solve(name, MODEL, IMG, FAKE_K)
    assert geodesic_deg(sols[0].R, R_TRUE) < 1e-4 and np.linalg.norm(sols[0].t - T_TRUE) < 1e-6


def test_single_solution_solvers():
    for name in ("epnp", "sqpnp", "iterative"):
        assert len(solve(name, MODEL, IMG, FAKE_K)) == 1
    assert len(solve("ippe", MODEL, IMG, FAKE_K)) == 2


@pytest.mark.parametrize("name", SOLVERS)
def test_refine_does_not_worsen_reprojection(name):
    noisy = IMG + np.random.default_rng(0).normal(0, 0.5, IMG.shape)
    before = solve(name, MODEL, noisy, FAKE_K, refine=False)
    after = solve(name, MODEL, noisy, FAKE_K, refine=True)
    assert before and after
    assert after[0].reproj_px <= before[0].reproj_px + 1e-9


@pytest.mark.parametrize("name", SOLVERS)
def test_non_finite_raises(name):
    bad = IMG.copy()
    bad[1] = np.nan
    with pytest.raises(ValueError):
        solve(name, MODEL, bad, FAKE_K)


@pytest.mark.parametrize("refine", [False, True])
@pytest.mark.parametrize("name", SOLVERS)
def test_degenerate_returns_empty_or_finite(name, refine):
    line = np.stack([np.linspace(900, 1000, 4), np.linspace(500, 560, 4)], axis=1)
    sols = solve(name, MODEL, line, FAKE_K, refine=refine)
    assert isinstance(sols, list)
    for s in sols:
        assert np.isfinite(s.R).all() and np.isfinite(s.t).all() and np.isfinite(s.reproj_px)


def test_unknown_solver():
    with pytest.raises(ValueError):
        solve("p3p", MODEL, IMG, FAKE_K)
