import numpy as np

from dronepose.forecast.archive_noise import ARCHIVE_SCENES, fold_train
from dronepose.forecast.nets.evaluate import criterion_d2b


def _data(gain_by_fold, n=60, seed=0):
    rng = np.random.default_rng(seed)
    traj, strata, weight = np.arange(n), np.array(["acro", "angle", "pmode"] * (n // 3)), rng.uniform(0.5, 2, n)
    f, rf = {}, {}
    for fold, g in gain_by_fold.items():
        f[fold] = rng.uniform(0.1, 0.3, n)
        rf[fold] = [f[fold] * (1 - g) + rng.normal(scale=0.002, size=n) for _ in range(3)]
    return rf, f, weight, traj, strata


def test_passes_when_every_fold_gains():
    c = criterion_d2b(*_data({"03": 0.3, "09": 0.2, "14": 0.25}), b=2000)
    assert c["pass"] and c["ci"][0] > 0 and set(c["folds"]) == {"03", "09", "14"}
    assert all(c["folds"][s]["value"] > 0 for s in c["folds"]) and len(c["per_seed"]) == 3


def test_fails_when_one_fold_loses():
    c = criterion_d2b(*_data({"03": 0.4, "09": 0.4, "14": -0.05}), b=2000)
    assert c["ci"][0] > 0 and not c["pass"]


def test_fails_when_total_ci_touches_zero():
    rf, f, w, traj, strata = _data({"03": 0.002, "09": 0.002, "14": 0.002})
    rf = {s: [e + np.random.default_rng(i).normal(scale=0.05, size=len(e)) for i, e in enumerate(v)]
          for s, v in rf.items()}
    c = criterion_d2b(rf, f, w, traj, strata, b=2000)
    assert c["ci"][0] <= 0 and not c["pass"]


def test_total_is_mean_over_folds():
    rf, f, w, traj, strata = _data({"03": 0.3, "09": 0.1, "14": 0.2})
    c = criterion_d2b(rf, f, w, traj, strata, b=500)
    tot_rf = np.mean([np.mean(rf[s], 0) for s in rf], 0)
    tot_f = np.mean([f[s] for s in f], 0)
    assert np.isclose(c["value"], 1 - np.sum(w * tot_rf) / np.sum(w * tot_f))


def test_folds():
    assert ARCHIVE_SCENES == ("03", "09", "14")
    assert fold_train("03") == ("09", "14") and fold_train("14") == ("03", "09")
    assert fold_train(None) == ARCHIVE_SCENES
