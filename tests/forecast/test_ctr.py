import numpy as np

from dronepose.forecast.baselines import cv
from dronepose.forecast.ctr import ctra, ctrv

N, H, FPS = 15, 30, 30
OBS_T = (np.arange(N) - (N - 1)) / FPS
FUT_T = np.arange(1, H + 1) / FPS
T_ALL = np.concatenate([OBS_T, FUT_T])


def split(p):
    last = p[:, N - 1:N]
    return p[:, :N] - last, p[:, N:] - last


def circle(r=3.0, speed=2.0, phase=0.3, vz=0.0, sign=1.0):
    w = sign * speed / r
    a = phase + w * T_ALL
    return np.stack([r * np.cos(a), r * np.sin(a), vz * T_ALL], -1)[None]


def test_ctrv_exact_on_circle():
    p = np.concatenate([circle(), circle(sign=-1.0, phase=2.0), circle(r=5.0, speed=4.0)])
    obs, fut = split(p)
    err = np.linalg.norm(ctrv(OBS_T, obs, FUT_T, 5) - fut, axis=-1)
    assert err.max() < 1e-3


def test_ctra_exact_on_accelerating_turn():
    w, v0, acc = 0.6, 1.5, 0.8
    tf = np.linspace(T_ALL[0], T_ALL[-1], 20001)
    psi = 0.4 + w * tf
    speed = v0 + acc * (tf - tf[0])
    vel = np.stack([speed * np.cos(psi), speed * np.sin(psi)], -1)
    pos = np.concatenate([np.zeros((1, 2)), np.cumsum((vel[1:] + vel[:-1]) / 2 * np.diff(tf)[:, None], 0)])
    idx = np.searchsorted(tf, T_ALL - 1e-12)
    p = np.concatenate([pos[idx], np.zeros((len(T_ALL), 1))], -1)[None]
    obs, fut = split(p)
    err = np.linalg.norm(ctra(OBS_T, obs, FUT_T, 5) - fut, axis=-1)
    assert err.max() < 1e-3
    assert np.linalg.norm(ctrv(OBS_T, obs, FUT_T, 5) - fut, axis=-1).max() > 0.1   # CTRV misses the acceleration


def test_straight_line_limit():
    v = np.array([[0.8, -0.5, 0.2], [0.0, 0.0, 0.0], [3.0, 1.0, -1.0]])
    p = v[:, None] * T_ALL[None, :, None] + np.array([1.0, 2.0, 3.0])
    obs, _ = split(p)
    for fn in (ctrv, ctra):
        pred = fn(OBS_T, obs, FUT_T, 5)
        assert np.isfinite(pred).all()
        np.testing.assert_allclose(pred, cv(OBS_T, obs, FUT_T, 5), atol=1e-9)


def test_vertical_is_cv():
    p = circle(vz=1.2)
    obs, fut = split(p)
    for fn in (ctrv, ctra):
        np.testing.assert_allclose(fn(OBS_T, obs, FUT_T, 5)[..., 2], fut[..., 2], atol=1e-9)


def test_ctra_stops_instead_of_reversing():
    # braking along x: the fitted deceleration must stop the drone, not drive it backwards
    v0, a = 2.0, -4.0
    t = T_ALL - T_ALL[0]
    t_stop = -v0 / a
    x = np.where(t < t_stop, v0 * t + a * t ** 2 / 2, v0 * t_stop + a * t_stop ** 2 / 2)
    p = np.stack([x, np.zeros_like(x), np.zeros_like(x)], -1)[None]
    obs, _ = split(p)
    pred = ctra(OBS_T, obs, FUT_T, 5)
    assert (np.diff(pred[0, :, 0]) >= -1e-12).all()
    v_last = v0 + a * t[N - 1]                     # speed at the last observation (it stops 0.1 s later)
    assert pred[0, -1, 0] <= v_last ** 2 / (2 * -a) + 0.01
