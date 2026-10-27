import dataclasses

import numpy as np

from dronepose.forecast.dataset import Trajectory
from dronepose.forecast.kalman import kalman_predict
from dronepose.forecast.kalman_los import kalman_los_predict
from dronepose.forecast.metrics import ade_at, position_errors
from dronepose.forecast.tuning import predict_all, tune
from tests.forecast.test_gaps import PARAMS
from tests.forecast.test_tuning import CFG, _traj, _windows


def _los_noisy(w, sigma, seed=0):
    """Noise only along the line of sight camera -> drone (camera at the origin of the absolute positions)."""
    absolute = w.last_p[:, None] + w.obs_p
    los = absolute / np.linalg.norm(absolute, axis=-1, keepdims=True)
    n = np.random.default_rng(seed).normal(scale=sigma, size=w.obs_p.shape[:2])
    return dataclasses.replace(w, obs_p=w.obs_p + n[..., None] * los)


def _jerk_traj(seed, T=150):
    """Random-walk acceleration 4 m from the camera: the motion needs a responsive filter across the line of sight."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 30
    a = np.cumsum(rng.normal(scale=3, size=(T, 3)), 0) / 30
    p = np.array([0, 4.0, 0]) + np.cumsum(np.cumsum(a, 0) / 30, 0) / 30
    return Trajectory(scene="03", drone="01", split="train", motion_type="linear", copies=[("01", 0)], s_c=1.0,
                      times=t, p_G=p, R_G=np.tile(np.eye(3), (T, 1, 1)))


def _ade(pred, w):
    return ade_at(position_errors(pred, w.fut_p), 30).mean()


def test_equals_kf_ca_when_isotropic():
    w = _windows([_traj(chaos=True), _traj(circle=True, seed=2)])
    a = kalman_los_predict(w.obs_t, w.obs_p, w.last_p, w.fut_t, 1e3, 1e-2, 1e-2)
    b = kalman_predict(w.obs_t, w.obs_p, w.fut_t, "ca", 1e3, 1e-2)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_rotation_invariance():
    # the result does not depend on the frame: rotating everything about the camera rotates the prediction
    w = _windows([_traj(chaos=True)])
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    a = kalman_los_predict(w.obs_t, w.obs_p, w.last_p, w.fut_t, 1e3, 1e-1, 1e-4)
    b = kalman_los_predict(w.obs_t, w.obs_p @ Rz.T, w.last_p @ Rz.T, w.fut_t, 1e3, 1e-1, 1e-4)
    np.testing.assert_allclose(a @ Rz.T, b, atol=1e-9)


def test_downweights_los_noise():
    w = _windows([_jerk_traj(s) for s in range(4)])
    noisy = _los_noisy(w, 0.05)
    best_iso = min(_ade(kalman_predict(noisy.obs_t, noisy.obs_p, noisy.fut_t, "ca", q, r), w)
                   for q in (1, 10, 100, 1e3, 1e4) for r in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0))
    los = _ade(kalman_los_predict(noisy.obs_t, noisy.obs_p, noisy.last_p, noisy.fut_t, 100, 1.0, 1e-4), w)
    assert los < 0.75 * best_iso


def test_predict_all_los_key_only_on_request():
    w = _windows([_traj()])
    assert "kf_ca_los" not in predict_all(w, PARAMS, CFG)
    p = {**PARAMS, "kf_ca_los": {"q": 1e3, "r_los": 1e-2, "r_lat": 1e-4}}
    out = predict_all(w, p, CFG, los=True)
    np.testing.assert_allclose(out["kf_ca_los"], kalman_los_predict(w.obs_t, w.obs_p, w.last_p, w.fut_t, 1e3, 1e-2,
                                                                    1e-4))


def test_tune_los_picks_anisotropy():
    cfg = dataclasses.replace(CFG, kf_ca_q_grid=(100.0,), kf_los_r_los_grid=(1e-4, 1.0), kf_los_r_lat_grid=(1e-4, 1.0))
    w = _los_noisy(_windows([_jerk_traj(s) for s in range(4)]), 0.05)
    p = tune(w, cfg, los=True)
    assert p["kf_ca_los"] == {"q": 100.0, "r_los": 1.0, "r_lat": 1e-4}
    assert "kf_ca_los" not in tune(w, cfg)
