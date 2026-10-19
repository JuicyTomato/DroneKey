import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import Trajectory
from dronepose.forecast.metrics import ade_at, position_errors
from dronepose.forecast.tuning import check_split, predict_all, tune
from dronepose.forecast.windows import make_windows

CFG = dataclasses.replace(load_config(), kf_cv_q_grid=(1e-6, 1e-3, 1.0), kf_ca_q_grid=(1e-4, 1.0, 1e3),
                          kf_r_grid=(1e-6, 1e-3), ctr_k_grid=(3, 5), acro_k_grid=(3,), acro_lam_grid=(0.0, 8.0),
                          acro_thrust=("constant", "hold_vz"), acro_kd=("estimate",))


def _traj(T=80, drone="01", s_c=1.0, chaos=False, seed=0, scene="03", circle=False):
    t = np.arange(T) / 30
    p = np.stack([0.3 + 0.8 * t, -0.5 * t, 1 + 0.2 * t], axis=1)
    if circle:
        p = np.stack([2 * np.cos(3 * t + seed), 2 * np.sin(3 * t + seed), 0 * t], axis=1)
    if chaos:
        p = p + np.cumsum(np.random.default_rng(seed).normal(scale=0.5, size=(T, 3)), axis=0)
    R = Rotation.from_euler("xz", np.stack([5 * np.sin(t), 20 * t], axis=1), degrees=True).as_matrix()
    return Trajectory(scene=scene, drone=drone, split="train", motion_type="linear", copies=[("01", 0)], s_c=s_c,
                      times=t, p_G=p, R_G=R)


def _windows(trajs):
    return make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)


def test_check_split_refuses_test():
    with pytest.raises(ValueError):
        check_split("test", False)
    check_split("test", True)
    check_split("val", False)


def test_tune_picks_cv_on_uniform_motion():
    w = _windows([_traj(), _traj(seed=1)])
    params = tune(w, CFG)
    assert {"cv_k", "ca_k", "kf_cv", "kf_ca", "grid", "edge_warnings"} <= set(params)
    pred = predict_all(w, params, CFG)["cv"]
    assert ade_at(position_errors(pred, w.fut_p), 30).max() < 1e-9


def test_tuning_flags_grid_edge():
    params = tune(_windows([_traj(chaos=True), _traj(chaos=True, seed=3)]), CFG)
    assert params["edge_warnings"]
    assert all(isinstance(m, str) for m in params["edge_warnings"])


def test_tuning_ignores_non_metric_windows():
    base = [_traj(chaos=True, seed=5), _traj(chaos=True, seed=6, scene="14")]
    a = tune(_windows(base), CFG)
    b = tune(_windows(base + [_traj(drone="08", s_c=None, chaos=True, seed=9)]), CFG)
    assert {k: a[k] for k in ("cv_k", "ca_k", "kf_cv", "kf_ca")} == {k: b[k] for k in ("cv_k", "ca_k", "kf_cv", "kf_ca")}


def test_predict_all_keys_and_shapes():
    w = _windows([_traj()])
    preds = predict_all(w, tune(w, CFG), CFG)
    pos = ("const", "cv15", "cv", "ca", "kf_cv", "kf_ca", "physics", "ctrv", "ctra", "physics_acro", "physics_acro_gen")
    assert set(preds) == set(pos) | {"rot_hold", "rot_const_omega"}
    for k in pos:
        assert preds[k].shape == (len(w), CFG.pred_len, 3)
    assert preds["rot_hold"].shape == preds["rot_const_omega"].shape == (len(w), CFG.pred_len, 3, 3)


def test_minimum_k_is_not_an_edge():
    # k = degree + 1 is the smallest legal fit length: the grid cannot extend below it, so it is not a warning
    w = _windows([_traj(circle=True, seed=1), _traj(circle=True, seed=2)])
    cfg = dataclasses.replace(CFG, cv_k_grid=(2, 3), ca_k_grid=(3, 5))
    params = tune(w, cfg)
    assert not any(m.startswith(("cv: k = 2", "ca: k = 3")) for m in params["edge_warnings"])
    assert (params["cv_k"], params["ca_k"]) == (2, 3)


def test_r_at_grid_edge_is_flagged():
    # noise-free motion: the smallest r always wins, i.e. the lower edge of the r grid -> warning
    params = tune(_windows([_traj(circle=True, seed=1)]), CFG)
    assert params["kf_ca"]["r"] == CFG.kf_r_grid[0]
    assert any(m.startswith("kf_ca: r = ") for m in params["edge_warnings"])
