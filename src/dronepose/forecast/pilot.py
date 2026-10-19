"""Simulated pilot (spec §5.2): piecewise-constant intentions through the pilot's "hand", per command mode.

An intention is a target velocity in G plus a yaw behaviour, held for U(0.3, 2.0) s and drawn from a declared
repertoire of maneuvers, with rare abrupt events. The pilot reacts after a delay; the intention goes through a
second-order filter (the hand: band 2-5 Hz, damping from the overshoot) so the sticks are continuous, except at
abrupt events (true steps). Small coloured corrections (sum of sinusoids) and a dead zone act on the sticks.

Sticks per mode: pmode {v_xy, yaw_rate, vz} (heading frame), angle {tilt_xy, yaw_rate, vz}, acro {omega, c}
(body rate and thrust from the pilot's own angle-like loop). `freeze()` holds the current sticks forever.
The pilot updates at `rate_hz` (120 Hz); between updates the sticks follow a uniform quadratic B-spline through
the last three updates (C1: continuous, finite second derivative; about 1.5 periods of delay).
All randomness is drawn by `sample_setup`; the pilot itself is deterministic given its setup.
"""
from dataclasses import dataclass, fields
from types import SimpleNamespace

import numpy as np

from dronepose.forecast.controller import CtrlGains, _heading_axes, _hold_vz_thrust, angle_mode, attitude_rate_cmd, heading_of, p_mode
from dronepose.forecast.dynamics import E3, G, QuadState, cross3

MODES = ("acro", "angle", "pmode")
YAW_POLICIES = ("fixed", "slow", "toward_motion")
N_NOISE = 4   # sinusoids per noise channel


@dataclass
class PilotSetup:
    mode: np.ndarray            # (B,) str
    s: np.ndarray               # (B,) aggressiveness in [0, 1]
    style: np.ndarray           # (B,) str: cine / normale / sport (terciles of s)
    drone_class: np.ndarray     # (B,) str
    yaw_policy: np.ndarray      # (B,) str
    yaw_rate_slow: np.ndarray   # (B,) rad/s
    max_tilt: np.ndarray        # (B,) rad
    max_rate: np.ndarray        # (B,) rad/s
    max_speed: np.ndarray       # (B,) m/s
    max_climb: np.ndarray       # (B,) m/s
    max_descent: np.ndarray     # (B,) m/s
    c_max: np.ndarray           # (B,) m/s^2 of the class
    reaction_delay: np.ndarray  # (B,) s
    overshoot: np.ndarray       # (B,) fraction
    hand_band_hz: np.ndarray    # (B,)
    stick_noise: np.ndarray     # (B,) fraction of the stick range
    deadzone: np.ndarray        # (B,) fraction of the stick range
    tau_att: np.ndarray         # (B,) s, attitude loop (angle mode and the acro pilot)
    k_v: np.ndarray             # (B,) 1/s, velocity loop (P-mode and the pilot's outer loop)
    seg_dur: np.ndarray         # (B,K) s
    seg_maneuver: np.ndarray    # (B,K) str
    seg_abrupt: np.ndarray      # (B,K) str, "" = none
    seg_u: np.ndarray           # (B,K,6) uniforms for the maneuver parameters
    noise_f: np.ndarray         # (B,4,N_NOISE) Hz
    noise_phase: np.ndarray     # (B,4,N_NOISE) rad

    def take(self, idx) -> "PilotSetup":
        return PilotSetup(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    @staticmethod
    def concat(parts: list["PilotSetup"]) -> "PilotSetup":
        return PilotSetup(**{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(PilotSetup)})


def _lerp(r, s):
    return r[0] + (r[1] - r[0]) * s


def limits(s: np.ndarray, drone_class: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    pc, cl = cfg["pilot"], cfg["classes"]
    cls = lambda key: np.array([cl[c][key] for c in drone_class], float)
    climb, descent = cls("climb"), cls("descent")
    return {"max_tilt": np.radians(_lerp(pc["max_tilt_deg"], s)), "max_rate": np.radians(_lerp(pc["max_rate_deg"], s)),
            "max_speed": np.minimum(_lerp(pc["max_speed"], s), cls("max_speed")),
            "max_climb": 1 + (climb - 1) * s, "max_descent": 1 + (descent - 1) * s, "c_max": cls("c_max_g") * G}


def style_of(s: np.ndarray, cfg: dict) -> np.ndarray:
    return np.array(cfg["pilot"]["styles"])[np.digitize(s, [1 / 3, 2 / 3])]


def sample_setup(n: int, rng: np.random.Generator, cfg: dict, overrides: dict | None = None) -> PilotSetup:
    pc, mc = cfg["pilot"], cfg["maneuvers"]
    K = pc["n_segments"]
    U = lambda r, size=n: rng.uniform(r[0], r[1], size)
    pick = lambda probs, size=n: np.array(list(probs))[rng.choice(len(probs), size=size, p=list(probs.values()))]
    terc = rng.choice(3, size=n, p=pc["s_tercile_probs"])
    d = {"mode": pick(pc["mode_probs"]), "s": (terc + rng.uniform(0, 1, n)) / 3,
         "drone_class": np.array(sorted(cfg["classes"]))[rng.integers(0, len(cfg["classes"]), n)],
         "yaw_policy": pick(pc["yaw_policy_probs"]),
         "yaw_rate_slow": np.radians(U([-pc["yaw_rate_slow_deg"], pc["yaw_rate_slow_deg"]])),
         "reaction_delay": U(pc["reaction_delay"]), "overshoot": U(pc["overshoot"]),
         "hand_band_hz": U(pc["hand_band_hz"]), "stick_noise": U(pc["stick_noise"]), "deadzone": U(pc["deadzone"]),
         "tau_att": U(cfg["controller"]["tau_att"]), "k_v": U(cfg["controller"]["k_v"]),
         "seg_dur": U(pc["seg_duration"], (n, K)), "seg_maneuver": pick(mc["probs"], (n, K)),
         "seg_abrupt": np.where(rng.uniform(0, 1, (n, K)) < mc["abrupt_prob"],
                                np.array(mc["abrupt_types"])[rng.integers(0, len(mc["abrupt_types"]), (n, K))], ""),
         "seg_u": rng.uniform(0, 1, (n, K, 6)),
         "noise_f": U(pc["noise_freq_hz"], (n, 4, N_NOISE)), "noise_phase": U([0, 2 * np.pi], (n, 4, N_NOISE))}
    d["seg_abrupt"][:, 0] = ""     # the first intention is the initial state, never a step
    for k, v in (overrides or {}).items():
        d[k] = np.broadcast_to(np.asarray(v, dtype=object if d[k].dtype.kind == "U" else d[k].dtype), d[k].shape).copy()
    for k, v in d.items():
        if v.dtype.kind in "UO":
            d[k] = v.astype("U16")       # room for any label: a short override must not truncate later writes
    d.update(limits(d["s"], d["drone_class"], cfg))
    d["style"] = style_of(d["s"], cfg)
    return PilotSetup(**d)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Pilot:
    def __init__(self, setup: PilotSetup, cfg: dict, k_d: np.ndarray):
        self.st, self.cfg, self.k_d = setup, cfg, np.asarray(k_d, float)
        self._drag = SimpleNamespace(k_d=self.k_d)      # the controllers only use the linear drag the pilot knows
        st = setup
        B, K = st.seg_dur.shape
        self.B = B
        start = np.concatenate([np.zeros((B, 1)), np.cumsum(st.seg_dur, axis=1)[:, :-1]], axis=1)
        self.eff_start = start + np.where(np.arange(K) > 0, st.reaction_delay[:, None], 0.0)
        self.seg = np.full(B, -1)
        self.p0, self.h0, self.l0, self.v0 = np.zeros((B, 3)), np.zeros((B, 3)), np.zeros((B, 3)), np.zeros((B, 3))
        self.psi0, self.center = np.zeros(B), np.zeros((B, 3))
        self.x, self.xd = np.zeros((B, 4)), np.zeros((B, 4))       # hand filter: target v (3) and yaw rate (1)
        self.t_prev = None
        self._started = False
        self.frozen = np.zeros(B, bool)
        self.last_sticks: dict[str, np.ndarray] | None = None
        self.frozen_sticks: dict[str, np.ndarray] | None = None
        self.period = 1.0 / cfg["pilot"]["rate_hz"]
        self._t_upd = None
        self._L = None          # last three pilot updates, oldest first
        self.last_abrupt_t = np.full(B, -np.inf)
        self.target_velocity = np.zeros((B, 3))
        zeta = np.where(st.overshoot > 0, -np.log(np.maximum(st.overshoot, 1e-12))
                        / np.sqrt(np.pi ** 2 + np.log(np.maximum(st.overshoot, 1e-12)) ** 2), 1.0)
        self.zeta = np.minimum(zeta, 1.0)
        self.max_yaw = np.minimum(st.max_rate, np.radians(cfg["pilot"]["max_yaw_rate_deg"]))
        self.ctrl = CtrlGains(k_att=1 / st.tau_att, k_v=st.k_v, k_vz=np.full(B, cfg["controller"]["k_vz"]),
                              max_tilt=st.max_tilt, max_rate=st.max_rate)

    # ----- intentions -------------------------------------------------------------------------------------------
    def _segment(self, t):
        return np.clip((self.eff_start <= t + 1e-12).sum(axis=1) - 1, 0, None)

    @property
    def maneuver(self) -> np.ndarray:
        rows = np.arange(self.B)
        ab = self.st.seg_abrupt[rows, self.seg]
        return np.where(ab != "", ab, self.st.seg_maneuver[rows, self.seg])

    def t_since_change(self, t: float) -> np.ndarray:
        return t - self.eff_start[np.arange(self.B), self.seg]

    def abrupt_last_second(self, t: float) -> np.ndarray:
        return t - self.last_abrupt_t <= 1.0

    def _on_new_segment(self, new, t, state):
        rows = np.flatnonzero(new)
        k = self.seg[rows]
        psi = heading_of(state.R[rows])
        h, l = _heading_axes(psi)
        self.psi0[rows], self.h0[rows], self.l0[rows] = psi, h, l
        self.p0[rows], self.v0[rows] = state.p[rows], state.v[rows]
        u = self.st.seg_u[rows, k]
        r = _lerp(self.cfg["maneuvers"]["orbit_radius"], u[:, 3])
        side = np.where(u[:, 4] < 0.5, 1.0, -1.0)
        self.center[rows] = state.p[rows] + (r * side)[:, None] * l
        ab = self.st.seg_abrupt[rows, k] != ""
        self.last_abrupt_t[rows[ab]] = t

    def _targets(self, t, state):
        """Raw target velocity (B,3) in G and target yaw rate (B,) of the current intention."""
        st, mc = self.st, self.cfg["maneuvers"]
        rows = np.arange(self.B)
        k = self.seg
        man, ab = st.seg_maneuver[rows, k], st.seg_abrupt[rows, k]
        u = st.seg_u[rows, k]
        tau = t - self.eff_start[rows, k]
        sp = st.max_speed * u[:, 0] ** 2
        ang = 2 * np.pi * u[:, 1]
        side = np.where(u[:, 4] < 0.5, 1.0, -1.0)
        climb = (0.3 + 0.7 * u[:, 2]) * st.max_climb
        h, l = self.h0, self.l0
        horiz = lambda a: np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1)
        v = np.zeros((self.B, 3))
        yaw = np.zeros(self.B)
        yaw_free = np.ones(self.B, bool)       # yaw follows the sample's policy

        def put(mask, value):
            v[mask] = np.broadcast_to(value, v.shape)[mask]

        put(man == "cruise", sp[:, None] * horiz(ang) + ((2 * u[:, 2] - 1) * 0.3)[:, None] * E3)
        put((man == "lateral") | (man == "dolly"), (sp * side)[:, None] * l)
        put(man == "backward", -sp[:, None] * h)
        put(man == "climb", 0.3 * sp[:, None] * horiz(ang) + climb[:, None] * E3)
        put(man == "descend", 0.3 * sp[:, None] * horiz(ang) - ((0.3 + 0.7 * u[:, 2]) * st.max_descent)[:, None] * E3)
        put(man == "reveal", -0.7 * sp[:, None] * h + 0.5 * climb[:, None] * E3)
        f = _lerp(mc["slalom_freq_hz"], u[:, 5])
        put(man == "slalom", sp[:, None] * h + (0.6 * sp * np.sin(2 * np.pi * f * tau))[:, None] * l)
        w8 = 2 * np.pi / _lerp(mc["figure8_period"], u[:, 3])
        put(man == "figure8", sp[:, None] * (np.cos(w8 * tau)[:, None] * h + np.cos(2 * w8 * tau)[:, None] * l))
        orb = (man == "orbit") | (man == "helix")
        if orb.any():
            r = _lerp(mc["orbit_radius"], u[:, 3])
            rel = state.p - self.center
            rel[:, 2] = 0
            d = np.maximum(np.linalg.norm(rel, axis=1), 1e-6)
            out = rel / d[:, None]
            a_ok = 0.7 * G * np.tan(st.max_tilt)
            sp_o = np.minimum(np.maximum(sp, 0.3), np.sqrt(a_ok * r))
            tang = side[:, None] * cross3(E3, out)
            vo = sp_o[:, None] * tang + (1.0 * (r - d))[:, None] * out
            vo[:, 2] = np.where(man == "helix", 0.5 * climb, 0.0)
            put(orb, vo)
            bearing = np.arctan2(-out[:, 1], -out[:, 0])
            yaw = np.where(orb, self.cfg["pilot"]["yaw_gain"] * _wrap(bearing - heading_of(state.R)) + side * sp_o / r, yaw)
            yaw_free &= ~orb
        yaw_free &= man != "dolly"
        put(ab == "step_max", st.max_speed[:, None] * horiz(ang))
        put(ab == "brake", np.zeros(3))
        put(ab == "reversal", -self.v0)
        put(ab == "high_gain", sp[:, None] * horiz(ang))
        v[:, 2] = np.clip(v[:, 2], -st.max_descent, st.max_climb)
        vh = np.linalg.norm(v[:, :2], axis=1)
        v[:, :2] *= np.minimum(1.0, st.max_speed / np.maximum(vh, 1e-12))[:, None]

        pol = st.yaw_policy
        speed = np.linalg.norm(state.v[:, :2], axis=1)
        course = np.arctan2(state.v[:, 1], state.v[:, 0])
        toward = self.cfg["pilot"]["yaw_gain"] * _wrap(course - heading_of(state.R)) * np.clip(speed / 1.0, 0, 1) ** 2
        yaw = np.where(yaw_free, np.select([pol == "slow", pol == "toward_motion"], [st.yaw_rate_slow, toward], 0.0), yaw)
        return v, np.clip(yaw, -self.max_yaw, self.max_yaw)

    # ----- sticks -----------------------------------------------------------------------------------------------
    def _noise(self, t):
        st = self.st
        n = np.sin(2 * np.pi * st.noise_f * t + st.noise_phase).sum(-1) / np.sqrt(N_NOISE / 2)   # (B,4), unit rms
        return st.stick_noise[:, None] * n

    def _dead(self, x, full):
        """Dead zone on the stick normalised by its full range (continuous)."""
        d = self.st.deadzone.reshape((-1,) + (1,) * (x.ndim - 1))
        full = np.broadcast_to(full.reshape((-1,) + (1,) * (x.ndim - 1)), x.shape)
        xn = x / full
        return np.sign(xn) * np.maximum(np.abs(xn) - d, 0) / (1 - d) * full

    def _live_sticks(self, t, state):
        st = self.st
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        seg = self._segment(t)
        new = seg != self.seg
        self.seg = seg
        if new.any():
            self._on_new_segment(new, t, state)
        v_t, yaw_t = self._targets(t, state)
        self.target_velocity = v_t
        u = np.concatenate([v_t, yaw_t[:, None]], axis=1)
        rows = np.arange(self.B)
        hg = st.seg_abrupt[rows, seg] == "high_gain"
        wn = 2 * np.pi * st.hand_band_hz * np.where(hg, self.cfg["maneuvers"]["high_gain_factor"], 1.0)
        step_now = new & np.isin(st.seg_abrupt[rows, seg], ("step_max", "brake", "reversal"))
        if not self._started:
            self.x, self.xd = u.copy(), np.zeros_like(u)       # start in equilibrium on the first intention
            self._started = True
        elif dt > 0:
            self.x[step_now], self.xd[step_now] = u[step_now], 0.0
            # exact solution over the pilot period with the target held (zeta <= 1): an explicit integrator at
            # 120 Hz rings (60 Hz alternation) once zeta * wn * dt > 0.5, as in the high_gain events
            sig = (self.zeta * wn)[:, None]
            wd = (wn * np.sqrt(1 - self.zeta ** 2))[:, None]
            cos, sin_wd = np.cos(wd * dt), dt * np.sinc(wd * dt / np.pi)     # sin(wd dt) / wd, = dt when wd = 0
            decay = np.exp(-sig * dt)
            e, ed = self.x - u, self.xd
            self.x = u + decay * (e * (cos + sig * sin_wd) + ed * sin_wd)
            self.xd = decay * (ed * (cos - sig * sin_wd) - wn[:, None] ** 2 * e * sin_wd)
        v_f, yaw_f = self.x[:, :3], self.x[:, 3]
        nz = self._noise(t)
        psi = heading_of(state.R)
        h, l = _heading_axes(psi)
        k_v = st.k_v * np.where(hg, 2.0, 1.0)
        vz = self._dead(np.clip(v_f[:, 2] + nz[:, 3] * st.max_climb, -st.max_descent, st.max_climb), st.max_climb)
        yaw = self._dead(yaw_f + nz[:, 2] * self.max_yaw, self.max_yaw)
        sticks = {}
        # pmode: the target velocity in the heading frame
        vxy = np.stack([np.sum(v_f * h, 1), np.sum(v_f * l, 1)], -1) + nz[:, :2] * st.max_speed[:, None]
        sticks["pmode"] = {"v_xy": self._dead(vxy, st.max_speed), "yaw_rate": yaw, "vz": vz}
        # angle and acro: the pilot's outer loop, target velocity -> tilt (drag compensated by experience)
        a = k_v[:, None] * (v_f - state.v) + self.k_d[:, None] * state.v
        tilt = np.stack([np.arctan2(np.sum(a * h, 1), G), np.arctan2(np.sum(a * l, 1), G)], -1)
        tn = np.linalg.norm(tilt, axis=1)
        tilt *= np.minimum(1.0, st.max_tilt / np.maximum(tn, 1e-12))[:, None]
        sticks["angle"] = {"tilt_xy": self._dead(tilt + nz[:, :2] * st.max_tilt[:, None], st.max_tilt),
                           "yaw_rate": yaw, "vz": vz}
        tl = np.linalg.norm(tilt, axis=1)
        t_dir = (tilt[:, :1] * h + tilt[:, 1:] * l) / np.maximum(tl, 1e-12)[:, None]
        b3 = np.sin(tl)[:, None] * t_dir + np.cos(tl)[:, None] * E3
        w = attitude_rate_cmd(state.R, b3, psi, 1 / st.tau_att, st.max_rate, yaw_f)
        w = w + nz[:, [0, 1, 2]] * st.max_rate[:, None] * np.array([1.0, 1.0, 0.5])
        sticks["acro"] = {"omega": w, "c": _hold_vz_thrust(state, v_f[:, 2], self._drag, self.ctrl) + nz[:, 3] * 0.3 * G}
        return sticks

    def _combined(self, live: dict) -> dict[str, np.ndarray]:
        out = {"v_xy": np.zeros((self.B, 2)), "tilt_xy": np.zeros((self.B, 2)), "yaw_rate": np.zeros(self.B),
               "vz": np.zeros(self.B), "omega": np.zeros((self.B, 3)), "c": np.zeros(self.B)}
        for m in MODES:
            mask = self.st.mode == m
            if mask.any():
                for key, val in live[m].items():
                    out[key][mask] = val[mask]
        return out

    def sticks(self, t: float, state: QuadState) -> dict[str, np.ndarray]:
        """Sticks of every sample in its own mode's units; padded to a common dict (unused keys zero)."""
        T = self.period
        if self._t_upd is None or t >= self._t_upd + 2 * T - 1e-12:      # start, or a pause longer than a period
            L = self._combined(self._live_sticks(t, state))
            self._L, self._t_upd = [L, L, L], t
        elif t >= self._t_upd + T - 1e-12:
            self._L = self._L[1:] + [self._combined(self._live_sticks(t, state))]
            self._t_upd = t
        u = min((t - self._t_upd) / T, 1.0)
        b0, b1, b2 = 0.5 * (1 - u) ** 2, -u * u + u + 0.5, 0.5 * u * u
        P0, P1, P2 = self._L
        out = {k: b0 * P0[k] + b1 * P1[k] + b2 * P2[k] for k in P2}
        if self.frozen_sticks is not None:
            for key in out:
                out[key][self.frozen] = self.frozen_sticks[key][self.frozen]
        self.last_sticks = {k: v.copy() for k, v in out.items()}
        return out

    def freeze(self, mask: np.ndarray | None = None) -> None:
        """Hold the last sticks (hand filter included) of the masked samples for the rest of the flight."""
        if self.last_sticks is None:
            raise RuntimeError("freeze() before any sticks were computed")
        mask = np.ones(self.B, bool) if mask is None else np.asarray(mask, bool)
        new = mask & ~self.frozen
        if self.frozen_sticks is None:
            self.frozen_sticks = {k: v.copy() for k, v in self.last_sticks.items()}
        for k in self.frozen_sticks:
            self.frozen_sticks[k][new] = self.last_sticks[k][new]
        self.frozen |= new

    def command(self, t: float, state: QuadState) -> tuple[np.ndarray, np.ndarray]:
        """Body-rate and thrust commands: acro sticks directly, angle and P-mode sticks through their controller."""
        sk = self.sticks(t, state)
        q = self._drag
        w, c = sk["omega"].copy(), sk["c"].copy()
        for m, fn in (("angle", angle_mode), ("pmode", p_mode)):
            mask = self.st.mode == m
            if mask.any():
                wm, cm = fn(state, sk, q, self.ctrl)
                w[mask], c[mask] = wm[mask], cm[mask]
        return w, c
