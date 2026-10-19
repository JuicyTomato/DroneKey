import copy

import numpy as np
import pytest

from dronepose.forecast.config import load_generator_config
from dronepose.forecast.controller import heading_of
from dronepose.forecast.dynamics import E3, G
from dronepose.forecast.generator import META_KEYS, generate, omega_split

CFG = load_generator_config()
FI = CFG["generator"]["freeze_idx"]
FAST = copy.deepcopy(CFG)
FAST["generator"]["warmup_s"] = [1.0, 1.4]     # tests of properties that do not need the full 3-4 s warm-up


@pytest.fixture(scope="module")
def lv0():
    return generate(32, seed=11, level=0, cfg=CFG, record_hf=True)


def test_shapes_and_meta(lv0):
    assert len(lv0.trajs) == 32
    for tr in lv0.trajs:
        assert len(tr.times) == 60 and tr.freeze_idx == FI
        assert tr.p_G.dtype == np.float32 and tr.R_G.dtype == np.float32
        assert set(META_KEYS) <= set(tr.meta)
        assert tr.scene == "gen" and tr.drone == tr.meta["drone_class"] and tr.motion_type == tr.meta["style"]
        np.testing.assert_allclose(tr.p_G[0], 0, atol=1e-6)
    assert len({tr.key for tr in lv0.trajs}) == 32


def test_reproducible_with_seed():
    one = generate(8, seed=11, level=0, cfg=FAST)
    again = generate(8, seed=11, level=0, cfg=FAST)
    other = generate(8, seed=12, level=0, cfg=FAST, cand_ids=range(8), keep_rejected=True)
    for a, b in zip(one.trajs, again.trajs):
        np.testing.assert_array_equal(a.p_G, b.p_G)
        assert a.meta == b.meta
    assert not np.array_equal(one.trajs[0].p_G, other.trajs[0].p_G)


def test_levels_share_random_stream():
    ids = list(range(8))
    out = {lv: generate(8, seed=5, level=lv, cfg=FAST, cand_ids=ids, keep_rejected=True) for lv in (0, 1, 2)}
    for key in ("mode", "s", "drone_class", "yaw_policy", "k_d", "tau_w"):
        assert [t.meta[key] for t in out[0].trajs] == [t.meta[key] for t in out[1].trajs] == [t.meta[key] for t in out[2].trajs]
    hf = {lv: out[lv].internals["v"] for lv in (0, 1)}
    assert np.abs(hf[0][:, 0] - hf[1][:, 0]).max() < 0.5          # same intentions: first frame of the window close


def test_frozen_sticks_constant_in_future(lv0):
    keys = [k for k in lv0.internals if k.startswith("sticks_")]
    for k in keys:
        x = lv0.internals[k]
        np.testing.assert_array_equal(x[:, FI:], np.repeat(x[:, FI:FI + 1], 60 - FI, axis=1))
    # no step at the freeze: at the integration rate the last live increment matches the ones just before
    oe = CFG["dynamics"]["out_every"]
    f = FI * oe                                                   # freeze step in the hf window
    for k in ("v_xy", "tilt_xy", "vz", "yaw_rate"):
        x = lv0.internals[f"hf_sticks_{k}"].reshape(32, -1, np.prod(lv0.internals[f"hf_sticks_{k}"].shape[2:], dtype=int))
        d = np.abs(np.diff(x, axis=1)).max(-1)                   # (n, steps)
        assert (d[:, f - 1] <= 1.5 * np.maximum(d[:, f - 2], d[:, f - 3]) + 1e-4).all(), k   # a step is >> 1e-4
        assert (d[:, f:] == 0).all()


def test_flatness_level0(lv0):
    hf = lv0.internals
    dt = CFG["dynamics"]["dt"]
    v, c, b3 = hf["hf_v"], hf["hf_c"], hf["hf_b3"]
    a = (v[:, 2:] - v[:, :-2]) / (2 * dt)
    k_d = np.array([t.meta["k_d"] for t in lv0.trajs])[:, None, None]
    res = c[:, 1:-1, None] * b3[:, 1:-1] - (a + G * E3 + k_d * v[:, 1:-1])
    assert np.abs(res).max() < 0.05


def test_coordinated_turn_continues():
    # a settled orbit at the freeze (long warm-up, one long intention): the frozen sticks keep turning at the same rate
    cfg = copy.deepcopy(CFG)
    cfg["generator"]["warmup_s"] = [6.0, 6.0]
    over = {"mode": ["pmode"] * 4 + ["angle"] * 4, "seg_maneuver": "orbit", "seg_abrupt": "", "seg_dur": 12.0,
            "stick_noise": 0.0, "yaw_policy": "fixed", "s": 0.5}
    out = generate(8, seed=3, level=0, cfg=cfg, setup_overrides=over, cand_ids=range(8))
    assert len(out.trajs) == 8 and {t.meta["mode"] for t in out.trajs} == {"pmode", "angle"}
    v = out.internals["v"][..., :2]
    psi = np.unwrap(np.arctan2(v[..., 1], v[..., 0]), axis=1)
    rate = np.gradient(psi, 1 / 30, axis=1)
    at_freeze = rate[:, FI]
    future = rate[:, FI + 5:].mean(1)
    np.testing.assert_allclose(future, at_freeze, rtol=0.1)


def test_acro_flip_is_kept_and_marked():
    over = {"mode": "acro", "s": 1.0, "seg_abrupt": "step_max", "seg_dur": 0.4}   # violent steps: high rates at the freeze
    out = generate(16, seed=7, level=1, cfg=FAST, setup_overrides=over)
    flips = [t for t in out.trajs if t.meta["flip"]]
    assert flips
    for t in flips:
        R = t.R_G.astype(float)
        assert np.abs(np.einsum("tji,tjk->tik", R, R) - np.eye(3)).max() < 1e-5
        assert np.degrees(np.arccos(np.clip(R[FI + 1:, 2, 2], -1, 1))).max() > 90


def test_rejection_reasons_counted():
    cfg = copy.deepcopy(FAST)
    cfg["generator"]["tilt_margin_deg"] = -15.0
    out = generate(8, seed=2, level=1, cfg=cfg)
    assert out.n_rejected["tilt"] > 0
    assert set(out.n_rejected) == {"tilt", "thrust", "fall"}
    assert sum(r["count"] for r in out.rejected_by.values()) == sum(out.n_rejected.values())


def test_active_pilot_changes_future():
    out = generate(16, seed=4, level=1, cfg=FAST, freeze=False)
    x = np.concatenate([out.internals[k].reshape(16, 60, -1) for k in out.internals if k.startswith("sticks_")], -1)
    changing = np.abs(x[:, FI + 1:] - x[:, FI:FI + 1]).max(axis=(1, 2)) > 1e-9
    assert changing.mean() > 0


def test_perturbation_stability():
    over = {"mode": ["angle"] * 6 + ["pmode"] * 6}
    a = generate(12, seed=9, level=1, cfg=FAST, setup_overrides=over, cand_ids=range(12), keep_rejected=True)
    b = generate(12, seed=9, level=1, cfg=FAST, setup_overrides=over, cand_ids=range(12), keep_rejected=True,
                 init_perturbation=1e-6)
    pa, pb = a.internals["p"], b.internals["p"]
    d = np.linalg.norm((pa[:, -1] - pa[:, FI]) - (pb[:, -1] - pb[:, FI]), axis=-1)
    assert np.abs(a.internals["v"] - b.internals["v"]).max() > 0          # the perturbation did propagate
    assert d.max() <= 1e-3


def test_heading_velocity_angle_covers_range():
    out = generate(64, seed=21, level=1, cfg=FAST)
    v = out.internals["v"][:, FI]
    psi = heading_of(out.internals["R"][:, FI])
    moving = np.linalg.norm(v[:, :2], axis=1) > 0.3
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(v[:, 1], v[:, 0]) - psi)))))[moving]
    assert ang.min() < 15 and ang.max() > 150


def test_jerk_continuous_outside_abrupt_events(lv0):
    hf = lv0.internals
    dt = CFG["dynamics"]["dt"]
    a = np.diff(hf["hf_v"], axis=1) / dt
    j = np.diff(a, axis=1) / dt
    dj = np.linalg.norm(np.diff(j, axis=1), axis=-1)            # jump of the jerk per 600 Hz step
    calm = ~hf["hf_abrupt"][:, 3:]
    assert np.isfinite(dj).all()
    # threshold from 64 level-0 samples: calm max 7.5, abrupt events up to 14 m/s^3 per step (the first-order
    # actuators keep the jerk continuous even through command steps; level 1 adds kinks at thrust saturation)
    assert dj[calm].max() < 20.0


def test_omega_split():
    R = np.repeat(np.eye(3)[None], 2, 0)
    w = np.array([[0.0, 0.0, 1.0], [3.0, 4.0, -2.0]])
    vert, horiz = omega_split(R, w)
    np.testing.assert_allclose(vert, [1.0, -2.0])
    np.testing.assert_allclose(horiz, [0.0, 5.0])


def test_generate_parallel_matches_serial():
    cfg = copy.deepcopy(FAST)
    cfg["generator"]["chunk"] = 4
    cfg["generator"]["warmup_s"] = [1.0, 1.0]
    a = generate(6, seed=13, level=1, cfg=cfg, cand_ids=range(8))        # the first candidates, serially
    b = generate(6, seed=13, level=1, cfg=cfg, workers=2)                 # two processes, chunks of 4
    assert [t.meta for t in a.trajs[:6]] == [t.meta for t in b.trajs]
    np.testing.assert_array_equal(a.trajs[5].p_G, b.trajs[-1].p_G)


def test_falling_acro_flip_is_kept():
    # spec §5.2: a flip in the future is never rejected; a flipping acro drone also falls, so the fall check must not
    # reject it either (here even a 0 m fall would count)
    from dronepose.forecast.generator import simulate_candidates
    cfg = copy.deepcopy(FAST)
    cfg["generator"]["max_fall_m"] = 0.0
    over = {"mode": "acro", "s": 1.0, "seg_abrupt": "step_max", "seg_dur": 0.4}
    r = simulate_candidates(np.arange(16), 7, 1, cfg, setup_overrides=over)
    flip = np.array([t.meta["flip"] for t in r["trajs"]])
    assert flip.any() and (r["reason"] == "fall").any()                       # the check is active on non-flips
    assert not (r["reason"][flip] == "fall").any()
