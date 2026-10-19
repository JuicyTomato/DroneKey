import numpy as np

from dronepose.forecast.config import load_generator_config
from dronepose.forecast.controller import heading_of
from dronepose.forecast.dynamics import G, QuadParams, QuadState, simulate
from dronepose.forecast.pilot import Pilot, limits, sample_setup

CFG = load_generator_config()
DT = 1 / 300


def hover_state(B):
    return QuadState(p=np.zeros((B, 3)), v=np.zeros((B, 3)), R=np.repeat(np.eye(3)[None], B, 0),
                     omega=np.zeros((B, 3)), c=np.full(B, G))


def quad(B, k_d=0.6):
    f = lambda x: np.full(B, float(x))
    return QuadParams(k_d=f(k_d), tau_w=f(0.04), tau_c=f(0.03), k_q=f(0), d_xy=np.zeros((B, 2)), c_max=f(2 * G),
                      delta=f(0), level=np.zeros(B, int))


def quiet_setup(B, **over):
    base = {"stick_noise": 0.0, "deadzone": 0.0, "seg_abrupt": "", "reaction_delay": 0.2, "overshoot": 0.05,
            "hand_band_hz": 3.0, "s": 0.5, "drone_class": "03"}
    base.update(over)
    return sample_setup(B, np.random.default_rng(0), CFG, overrides=base)


def sticks_series(pilot, B, seconds, dt=1 / 600, key="v_xy"):
    s = hover_state(B)
    out = []
    for i in range(int(round(seconds / dt))):
        out.append(pilot.sticks(i * dt, s)[key].copy())
    return np.array(out)


def fly(setup, seconds, k_d=0.6):
    B = len(setup.mode)
    prm = quad(B, k_d)
    pilot = Pilot(setup, CFG, prm.k_d)
    states, _ = simulate(hover_state(B), lambda s, k: pilot.command(k * DT, s), prm, int(round(seconds / DT)), DT, 3)
    return states, pilot


def test_setup_proportions():
    st = sample_setup(20000, np.random.default_rng(1), CFG)
    for m, p in (("acro", 0.2), ("angle", 0.4), ("pmode", 0.4)):
        assert abs((st.mode == m).mean() - p) < 0.015
    for y, p in (("fixed", 0.5), ("slow", 0.3), ("toward_motion", 0.2)):
        assert abs((st.yaw_policy == y).mean() - p) < 0.015
    terc = np.digitize(st.s, [1 / 3, 2 / 3])
    for i, p in enumerate((0.4, 0.4, 0.2)):
        assert abs((terc == i).mean() - p) < 0.015
    assert set(st.drone_class) == set(CFG["classes"])
    assert set(st.style) == {"cine", "normale", "sport"}


def test_limits_follow_s_and_class():
    lo = limits(np.array([0.0]), np.array(["03"]), CFG)
    hi = limits(np.array([1.0, 1.0]), np.array(["03", "08"]), CFG)
    np.testing.assert_allclose(np.degrees(lo["max_tilt"]), 20)
    np.testing.assert_allclose(np.degrees(lo["max_rate"]), 60)
    np.testing.assert_allclose(np.degrees(hi["max_tilt"]), 45)
    np.testing.assert_allclose(np.degrees(hi["max_rate"]), 300)
    np.testing.assert_allclose(lo["max_speed"], 2.0)
    np.testing.assert_allclose(hi["max_speed"], [15.0, 8.0])        # capped by the class (Tello 8 m/s)
    assert (hi["max_climb"] <= [8, 2]).all() and (hi["max_descent"] <= [6, 2]).all()


def test_hand_filter_continuous():
    B = 4
    st = quiet_setup(B, mode="pmode", yaw_policy="fixed", seg_maneuver="cruise", seg_dur=0.5)
    x = sticks_series(Pilot(st, CFG, np.full(B, 0.6)), B, 3.0)                 # (T,B,2)
    d1 = np.abs(np.diff(x, axis=0)).max()
    d2 = np.abs(np.diff(x, 2, axis=0)).max() * 600 ** 2
    assert np.abs(x).max() > 0.1                                               # the sticks do move
    wn = 2 * np.pi * 3.0
    assert d1 < 2 * st.max_speed.max() * wn / 600                              # no jump at intention changes
    assert d2 < 2 * wn ** 2 * 2 * st.max_speed.max()                          # finite second derivative


def test_abrupt_event_is_a_step():
    B = 2
    st = quiet_setup(B, mode="pmode", yaw_policy="fixed", seg_maneuver="hover", seg_dur=0.5)
    st.seg_abrupt[:, 1] = "step_max"
    x = sticks_series(Pilot(st, CFG, np.full(B, 0.6)), B, 1.0)
    assert np.abs(np.diff(x, axis=0)).max() > 1.0                              # a true step, marked abrupt


def test_reaction_delay():
    B = 3
    st = quiet_setup(B, mode="pmode", yaw_policy="fixed", seg_maneuver="cruise", seg_dur=1.0, reaction_delay=0.2)
    st.seg_maneuver[:, 0] = "hover"
    st.seg_u[:, 1, 0] = 0.8
    x = np.linalg.norm(sticks_series(Pilot(st, CFG, np.full(B, 0.6)), B, 1.6), axis=-1)    # (T,B)
    t = np.arange(len(x)) / 600
    assert np.abs(x[t < 1.2 - 1e-9]).max() < 1e-12
    assert (x[t > 1.25] > 1e-3).all()


def test_freeze_holds_sticks():
    B = 3
    st = quiet_setup(B, mode="pmode", yaw_policy="slow", seg_maneuver="slalom", seg_dur=0.6, stick_noise=0.02)
    pilot = Pilot(st, CFG, np.full(B, 0.6))
    s = hover_state(B)
    for i in range(600):
        last = {k: v.copy() for k, v in pilot.sticks(i / 600, s).items()}
    pilot.freeze()
    for i in range(600, 1200, 7):
        now = pilot.sticks(i / 600, s)
        for k in last:
            np.testing.assert_array_equal(now[k], last[k])


def test_partial_freeze_only_masked():
    B = 2
    st = quiet_setup(B, mode="pmode", yaw_policy="fixed", seg_maneuver="slalom", seg_dur=2.0)
    st.seg_u[:, :, 0] = 0.8
    pilot = Pilot(st, CFG, np.full(B, 0.6))
    s = hover_state(B)
    a = pilot.sticks(0.5, s)["v_xy"].copy()
    pilot.freeze(np.array([True, False]))
    b = pilot.sticks(1.0, s)["v_xy"]
    np.testing.assert_array_equal(b[0], a[0])
    assert np.abs(b[1] - a[1]).max() > 1e-3


def test_toward_motion_policy_aligns_heading():
    B = 3
    st = quiet_setup(B, mode="pmode", yaw_policy="toward_motion", seg_maneuver="cruise", seg_dur=5.0)
    st.seg_u[:, :, 0] = 0.8
    states, _ = fly(st, 4.0)
    v, R = states[-1].v, states[-1].R
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(v[:, 1], v[:, 0]) - heading_of(R))))))
    assert (ang < 15).all()


def test_fixed_policy_keeps_heading_while_moving_sideways():
    B = 3
    st = quiet_setup(B, mode="pmode", yaw_policy="fixed", seg_maneuver="lateral", seg_dur=5.0)
    st.seg_u[:, :, 0] = 0.8
    states, _ = fly(st, 3.0)
    heads = np.degrees(np.stack([heading_of(s.R) for s in states]))
    assert np.abs(heads).max() < 1.0
    v = states[-1].v
    assert (np.linalg.norm(v[:, :2], axis=1) > 1.0).all()
    np.testing.assert_allclose(np.abs(v[:, 0]) / np.linalg.norm(v[:, :2], axis=1), 0, atol=0.05)   # along body y


def test_all_modes_track_cruise():
    B = 3
    st = quiet_setup(B, yaw_policy="fixed", seg_maneuver="cruise", seg_dur=5.0, s=0.6)
    st.mode[:] = ["acro", "angle", "pmode"]
    st.seg_u[:, :, 0] = 0.5
    states, pilot = fly(st, 4.0)
    target = pilot.target_velocity
    err = np.linalg.norm(states[-1].v - target, axis=1) / np.linalg.norm(target, axis=1)
    assert (err < 0.15).all()
    assert all(np.isfinite(s.R).all() for s in states)


def test_repeated_call_at_same_time_keeps_filter_state():
    B = 2
    st = quiet_setup(B, mode="pmode", yaw_policy="fixed", seg_maneuver="cruise", seg_dur=0.4)
    a, b = Pilot(st, CFG, np.full(B, 0.6)), Pilot(st, CFG, np.full(B, 0.6))
    s = hover_state(B)
    for i in range(400):
        xa = a.sticks(i / 600, s)["v_xy"].copy()
        b.sticks(i / 600, s)
        xb = b.sticks(i / 600, s)["v_xy"].copy()      # second call at the same instant: no time passes
    np.testing.assert_array_equal(xa, xb)


def test_hand_filter_does_not_ring_at_high_gain():
    # high_gain hand (5 Hz x 3) with no overshoot (critical damping): the pilot updates at 120 Hz must approach the
    # target monotonically, without the 60 Hz alternation of an explicit integrator at large wn * dt
    B = 2
    st = quiet_setup(B, mode="pmode", yaw_policy="fixed", seg_maneuver="hover", seg_dur=0.5, hand_band_hz=5.0,
                     overshoot=0.0)
    st.seg_abrupt[:, 1] = "high_gain"
    st.seg_dur[:, 1] = 2.0                                                     # the event lasts past the series
    st.seg_u[:, 1, 0] = 0.8
    pilot = Pilot(st, CFG, np.full(B, 0.6))
    x = sticks_series(pilot, B, 1.5, dt=1 / 120)                               # (T,B,2) at the update instants
    d = np.diff(x[int(0.75 * 120):], axis=0)                                   # after the event starts (0.7 s)
    d = np.where(np.abs(d) < 1e-9, 0.0, d)
    flips = (np.sign(d[1:]) * np.sign(d[:-1]) < 0).sum(0)
    assert np.abs(x[-1]).max() > 0.5                                           # the sticks did move
    assert flips.max() == 0


def test_toward_motion_policy_share():
    # spec §5.6: the "nose in the direction of motion" share is about 20% (the yaw policy drawn per sample)
    st = sample_setup(20000, np.random.default_rng(5), CFG)
    assert abs((st.yaw_policy == "toward_motion").mean() - 0.20) < 0.015


def test_hand_overshoot_matches_the_drawn_one():
    B = 1
    st = quiet_setup(B, mode="pmode", yaw_policy="fixed", seg_maneuver="hover", seg_dur=0.5, overshoot=0.2)
    st.seg_maneuver[:, 1] = "cruise"
    st.seg_dur[:, 1] = 3.0
    st.seg_u[:, 1, 0] = 0.8
    n = np.linalg.norm(sticks_series(Pilot(st, CFG, np.full(B, 0.6)), B, 2.5, dt=1 / 120)[:, 0], axis=-1)
    assert abs(n.max() / n[-1] - 1 - 0.2) < 0.005
