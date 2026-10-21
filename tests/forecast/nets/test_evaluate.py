import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dronepose.forecast.nets.evaluate import (criterion_d1, criterion_d2, criterion_d3, criterion_d4, predict_run,
                                              refuse_test_split, seed_mean_errors)
from dronepose.forecast.nets.train import train_run
from tests.forecast.nets.test_train import CFG, CPU, _pool

B = 200
W = 120


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    traj = np.arange(W)
    strata = np.array(["acro", "angle", "pmode"])[traj % 3]
    weight = rng.uniform(0.5, 2, W)
    ref = rng.uniform(0.5, 1.5, W)
    return rng, traj, strata, weight, ref


def test_seed_mean_errors():
    target = np.zeros((2, 30, 3))
    preds = [np.zeros((2, 30, 3)), np.zeros((2, 30, 3)), np.zeros((2, 30, 3))]
    preds[0][0, :, 0] = 1.0
    preds[1][0, :, 0] = 2.0
    preds[2][1, :, 1] = 3.0
    np.testing.assert_allclose(seed_mean_errors(preds, target), [1.0, 1.0])


def test_predict_run_untrained_residual_is_base(tmp_path):
    pool = _pool(W=20)
    train_run("R-F", "clean", 0, dict(CFG, max_epochs=1, lr=0.0), pool, pool, tmp_path, device=CPU)
    pred = predict_run(tmp_path, pool[0], CPU)
    np.testing.assert_allclose(pred, pool[0].bases["phys"], atol=1e-5)


def test_d1_definition_and_pass_logic():
    rng, traj, strata, weight, f = _setup()
    good = [0.8 * f * rng.uniform(0.95, 1.05, W) for _ in range(3)]
    d = criterion_d1(good, f, weight, traj, strata, b=B, seed=0)
    assert set(d) >= {"definition", "value", "ci", "per_seed", "pass"}
    assert "R-F" in d["definition"] and d["pass"] and d["ci"][0] > 0 and len(d["per_seed"]) == 3
    assert d["value"] == pytest.approx(1 - np.sum(weight * np.mean(good, 0)) / np.sum(weight * f))
    bad_seed = [0.7 * f, 0.7 * f, 1.2 * f]                               # mean still better, one seed worse
    d = criterion_d1(bad_seed, f, weight, traj, strata, b=B, seed=0)
    assert d["value"] > 0 and d["ci"][0] > 0 and not d["pass"]


def test_d2_three_of_four_levels():
    rng, traj, strata, weight, f = _setup()
    levels = {f"l{i}": ([0.8 * f, 0.8 * f, 0.8 * f], f) for i in range(3)}
    levels["l3"] = ([1.1 * f, 1.1 * f, 1.1 * f], f)
    d = criterion_d2(levels, weight, traj, strata, b=B, seed=0)
    assert d["pass"] and d["n_levels_pass"] == 3 and not d["levels"]["l3"]["pass"]
    levels["l2"] = ([1.1 * f, 1.1 * f, 1.1 * f], f)
    d = criterion_d2(levels, weight, traj, strata, b=B, seed=0)
    assert not d["pass"] and d["n_levels_pass"] == 2


def test_d3_uses_metric_windows_only():
    rng = np.random.default_rng(1)
    n = 60
    traj = np.repeat(np.arange(12), 5)
    drone = np.where(traj < 10, "01", "08")
    scene = np.array(["03", "09"])[traj % 2]
    metric = drone != "08"
    cv = rng.uniform(0.5, 1, n)
    rf = cv * np.where(metric, 0.9, 5.0)                                 # 08 much worse: must not count
    d = criterion_d3([rf, rf, rf], cv, metric, traj, scene, drone, b=B, seed=0)
    assert d["pass"] and d["value"] == pytest.approx(0.1)
    assert d["n_windows"] == int(metric.sum())
    assert d["non_metric"]["drone 08"]["skill"] == pytest.approx(-4.0)
    assert "CV-2" in d["definition"]


def test_d4_requires_both_contrasts():
    rng, traj, strata, weight, ref = _setup()
    rf = [0.8 * ref] * 3
    d = criterion_d4(rf, [ref] * 3, [ref] * 3, weight, traj, strata, b=B, seed=0)
    assert d["pass"] and set(d["contrasts"]) == {"G", "R-CV"}
    d = criterion_d4(rf, [ref] * 3, [0.7 * ref] * 3, weight, traj, strata, b=B, seed=0)
    assert not d["pass"] and d["contrasts"]["G"]["pass"] and not d["contrasts"]["R-CV"]["pass"]


def test_refuses_test_splits():
    for name in ("test", "test_L0", "dronesyn_test"):
        with pytest.raises(ValueError):
            refuse_test_split(name)
    for name in ("val", "val_L2", "active", "select"):
        refuse_test_split(name)
