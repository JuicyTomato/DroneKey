import copy
import dataclasses

import numpy as np
import pytest

from dronepose.forecast.baselines import ca, constant_position, cv
from dronepose.forecast.config import load_config, load_generator_config
from dronepose.forecast.generator import generate
from dronepose.forecast.kalman import kalman_predict
from dronepose.forecast.metrics import paired_bootstrap_skill
from dronepose.forecast.orientation import constant_angular_velocity, hold_orientation
from dronepose.forecast.physics import physics_predict
from dronepose.forecast.report import summarize_generated
from dronepose.forecast.tuning import KINEMATIC_METHODS, POSITION_METHODS, best_kinematic, predict_all, tune
from dronepose.forecast.windows import freeze_windows, make_windows
from tests.forecast.test_tuning import _traj

CFG = dataclasses.replace(load_config(), kf_cv_q_grid=(1e-3, 1.0), kf_ca_q_grid=(1.0, 1e3), kf_r_grid=(1e-6,),
                          ctr_k_grid=(3, 5), acro_k_grid=(3,), acro_lam_grid=(0.0, 8.0),
                          acro_thrust=("constant", "hold_vz"), acro_kd=("estimate",))
GCFG = copy.deepcopy(load_generator_config())
GCFG["generator"]["warmup_s"] = [1.0, 1.4]
PARAMS = {"cv_k": 2, "ca_k": 3, "kf_cv": {"q": 100.0, "r": 1e-6}, "kf_ca": {"q": 1000.0, "r": 1e-6}, "ctr_k": 3,
          "acro_k": 3, "acro_kd": "estimate", "acro_gen": {"lam": 5.0, "thrust": "hold_vz", "k": 3, "kd": "estimate"},
          "acro_mode": {m: {"lam": lam, "thrust": th, "k": 3, "kd": "estimate"}
                        for m, lam, th in (("acro", 0.0, "constant"), ("angle", 8.0, "hold_vz"), ("pmode", 5.0, "hold_vz"))},
          "scores": {"cv": 0.3, "ca": 0.2, "kf_cv": 0.29, "kf_ca": 0.21, "ctrv": 0.25, "ctra": 0.19,
                     "physics_acro_gen": 0.05}}
B = 200


def _with_strat_meta(trajs, seed=0):
    rng = np.random.default_rng(seed)
    for t in trajs:
        t.meta["weight"] = float(rng.uniform(0.2, 3.0))
        t.meta["cell"] = "".join(str(int(x)) for x in rng.integers(0, 3, 5))
    return trajs


@pytest.fixture(scope="module")
def gen1():
    trajs = _with_strat_meta(generate(24, seed=41, level=1, cfg=GCFG).trajs)
    return freeze_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)


@pytest.fixture(scope="module")
def gen0():
    trajs = _with_strat_meta(generate(18, seed=42, level=0, cfg=GCFG, cand_ids=range(18), keep_rejected=True,
                                      setup_overrides={"mode": ["acro"] * 6 + ["angle"] * 6 + ["pmode"] * 6}).trajs)
    return freeze_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)


def test_existing_methods_unchanged():
    w = make_windows([_traj(), _traj(chaos=True, seed=3)], CFG.obs_len, CFG.pred_len, CFG.fps)
    p = predict_all(w, PARAMS, CFG)
    args = (w.obs_t, w.obs_p, w.fut_t)
    ref = {"const": constant_position(*args), "cv15": cv(*args, len(w.obs_t)), "cv": cv(*args, 2), "ca": ca(*args, 3),
           "kf_cv": kalman_predict(*args, "cv", 100.0, 1e-6), "kf_ca": kalman_predict(*args, "ca", 1000.0, 1e-6),
           "physics": physics_predict(w.obs_t, w.obs_p, w.obs_u, w.fut_t, 2, CFG.g, CFG.max_tilt_deg),
           "rot_hold": hold_orientation(w.obs_R, len(w.fut_t)),
           "rot_const_omega": constant_angular_velocity(w.obs_t, w.obs_R, w.fut_t)}
    for k, v in ref.items():
        np.testing.assert_array_equal(p[k], v)
    assert set(POSITION_METHODS) >= {"ctrv", "ctra", "physics_acro", "physics_acro_gen"}


def test_mode_variant_only_with_meta(gen1):
    w = make_windows([_traj()], CFG.obs_len, CFG.pred_len, CFG.fps)
    assert "physics_acro_mode" not in predict_all(w, PARAMS, CFG)
    p = predict_all(gen1, PARAMS, CFG)
    assert p["physics_acro_mode"].shape == p["cv"].shape
    for k in POSITION_METHODS:
        assert np.isfinite(p[k]).all(), k


def test_best_kinematic_chosen_on_train_scores():
    assert best_kinematic(PARAMS) == "ctra"                       # physics_acro_gen is lower but not kinematic
    assert set(KINEMATIC_METHODS) == {"cv", "ca", "kf_cv", "kf_ca", "ctrv", "ctra"}


def test_tune_generated_keys(gen1):
    params = tune(gen1, CFG)
    assert {"ctr_k", "acro_k", "acro_kd", "acro_gen", "acro_mode"} <= set(params)
    assert set(params["acro_mode"]) == set(gen1.meta["mode"])
    for v in [params["acro_gen"], *params["acro_mode"].values()]:
        assert v["lam"] in CFG.acro_lam_grid and v["thrust"] in CFG.acro_thrust
    assert {"ctrv", "ctra", "physics_acro", "physics_acro_gen"} <= set(params["scores"])


def test_tune_uses_importance_weights(gen1):
    w = copy.deepcopy(gen1)
    a = tune(w, CFG)["scores"]["cv"]
    w.meta["weight"] = np.where(np.arange(len(w)) < 4, 50.0, 0.01)
    b = tune(w, CFG)["scores"]["cv"]
    assert a != b


def test_weighted_vs_balanced(gen1):
    preds = predict_all(gen1, PARAMS, CFG)
    one = summarize_generated(gen1, preds, CFG, np.ones(len(gen1)), "ctra", bootstrap_b=B)
    rng = np.random.default_rng(1)
    other = summarize_generated(gen1, preds, CFG, rng.uniform(0.1, 5, len(gen1)), "ctra", bootstrap_b=B)
    g = lambda s, v: next(x for x in s["versions"][v]["groups"] if x["group"] == "tutti")
    assert g(one, "balanced")["position"] == g(other, "balanced")["position"]
    assert g(one, "weighted")["position"] != g(other, "weighted")["position"]
    assert g(one, "weighted")["position"] == g(one, "balanced")["position"]


def test_weighted_skill_ci_matches_paired_bootstrap(gen1):
    preds = predict_all(gen1, PARAMS, CFG)
    wts = np.random.default_rng(2).uniform(0.1, 5, len(gen1))
    s = summarize_generated(gen1, preds, CFG, wts, "ctra", bootstrap_b=B)
    g = next(x for x in s["versions"]["weighted"]["groups"] if x["group"] == "tutti")
    e = {k: np.linalg.norm(preds[k] - gen1.fut_p, axis=-1).mean(1) for k in ("physics_acro_gen", "ctra")}
    ref = paired_bootstrap_skill(wts * e["physics_acro_gen"], wts * e["ctra"], gen1.traj, gen1.meta["mode"], B,
                                 CFG.bootstrap_seed)
    np.testing.assert_allclose(g["skill_best"]["physics_acro_gen"]["ade_1s"], ref, rtol=1e-12)


def test_generated_report_rejects_windows_without_meta():
    w = make_windows([_traj()], CFG.obs_len, CFG.pred_len, CFG.fps)
    with pytest.raises(ValueError):
        summarize_generated(w, predict_all(w, PARAMS, CFG), CFG, np.ones(len(w)), "ca", bootstrap_b=B)


def test_criteria_block_present(gen1):
    s = summarize_generated(gen1, predict_all(gen1, PARAMS, CFG), CFG, gen1.meta["weight"], "ctra", bootstrap_b=B)
    for c in ("C1", "C2", "C3", "C4"):
        assert {"value", "ci", "pass", "definition"} <= set(s["criteria"][c])
    assert isinstance(s["criteria"]["C1"]["pass"], bool) and len(s["criteria"]["C1"]["ci"]) == 2
    assert s["criteria"]["C4"]["pass"] is None                        # level 1: the oracle check needs level 0


def test_c4_oracle_on_level0(gen0):
    s = summarize_generated(gen0, predict_all(gen0, PARAMS, CFG), CFG, gen0.meta["weight"], "ctra", bootstrap_b=B)
    c4 = s["criteria"]["C4"]
    assert c4["pass"] is True and c4["value"] < 1e-4 and c4["n"] == 6


def test_regime_group_definition(gen1):
    s = summarize_generated(gen1, predict_all(gen1, PARAMS, CFG), CFG, np.ones(len(gen1)), "ctra", bootstrap_b=B)
    cells = gen1.meta["cell"]
    regime = np.array([c[0] == "0" and c[1] == "0" for c in cells])
    high = np.array([c[1] == "2" for c in cells])
    groups = {g["group"]: g for g in s["versions"]["balanced"]["groups"]}
    if regime.any():
        assert groups["regime"]["n_windows"] == regime.sum()
    assert groups["|ω_oriz| alto"]["n_windows"] == high.sum()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")    # C3 against a perfect reference divides by 0
def test_c2_is_one_sided(gen1):
    # C2 (confirmed 2026-10-19): at regime physics may win by any margin, it must not lose more than 10 points
    preds = predict_all(gen1, PARAMS, CFG)
    preds["physics_acro_gen"] = gen1.fut_p.copy()                    # perfect: skill +100% everywhere
    s = summarize_generated(gen1, preds, CFG, np.ones(len(gen1)), "ctra", bootstrap_b=B)
    if s["criteria"]["C2"]["value"] == s["criteria"]["C2"]["value"]:  # regime group present
        assert s["criteria"]["C2"]["pass"] is True
