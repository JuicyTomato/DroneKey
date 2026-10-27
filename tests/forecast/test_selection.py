import numpy as np

from dronepose.forecast.selection import SIMPLICITY, select_method


def _base(n=30, seed=0):
    rng = np.random.default_rng(seed)
    traj = np.repeat(np.arange(n), 4)
    strata = np.array(["03", "09", "14"])[traj % 3]
    return rng.uniform(0.05, 0.2, len(traj)), traj, strata


def test_simplest_wins_tie():
    e, traj, strata = _base()
    jitter = np.random.default_rng(2).normal(scale=0.01, size=len(e))
    r = select_method({"cv": e, "kf_ca": e + jitter - jitter.mean() - 1e-6}, traj, strata, b=2000)
    assert r["best"] == "kf_ca" and r["chosen"] == "cv" and "cv" in r["ties"]


def test_clear_winner():
    e, traj, strata = _base()
    r = select_method({"cv": e, "kf_ca": e * 0.5}, traj, strata, b=2000)
    assert r["chosen"] == "kf_ca" and r["ties"] == []


def test_tie_with_non_simplest():
    e, traj, strata = _base()
    noise = np.random.default_rng(1).normal(scale=1e-4, size=len(e))
    r = select_method({"cv": e * 2, "R-CV-6D": e * 0.6 + noise, "G-6D": e * 0.6}, traj, strata, b=2000)
    assert r["best"] in ("G-6D", "R-CV-6D") and r["chosen"] == "R-CV-6D"


def test_unknown_method_rejected():
    e, traj, strata = _base()
    try:
        select_method({"cv": e, "mystery": e}, traj, strata, b=100)
    except ValueError:
        return
    raise AssertionError("unknown method accepted")


def test_order():
    assert SIMPLICITY == ("cv", "ca", "kf_cv", "kf_ca", "kf_ca_los", "physics", "R-CV-6D", "R-CV-FT", "G-6D")
