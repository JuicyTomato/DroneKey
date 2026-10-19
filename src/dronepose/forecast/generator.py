"""Generator of physically consistent trajectories (plan step 3.1b, spec §5.2-5.3), vectorised over candidates.

Each candidate: warm-up of 3-4 s flown by the simulated pilot, freeze at t_f (output frame 29, the last observed),
then 1 s with the sticks of its command mode held (or, freeze=False, the pilot keeps flying: "active pilot" set).
Output: 60 frames at 30 FPS in metres, frame G, positions relative to the first frame, float32.

Every candidate has an id; its randomness comes from the block of BLOCK candidates it belongs to, drawn from
default_rng([seed, block]) before any simulation and independently of the mismatch level. Hence the same id gives
the same pilot, drone and intentions at every level, and a subset of ids can be regenerated alone.
Implausible candidates (past tilt beyond the style limit + margin, thrust command outside the range, a fall in the
future below the freeze altitude) are rejected and counted; acro flips in the future are kept and marked, even when
they fall.
"""
import itertools
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from dronepose.forecast.dataset import Trajectory
from dronepose.forecast.dynamics import E3, G, QuadParams, QuadState, acceleration, orthonormalize, step
from dronepose.forecast.pilot import Pilot, PilotSetup, sample_setup

META_KEYS = ("mode", "s", "style", "drone_class", "k_d", "tau_w", "tau_c", "tau_att", "level", "yaw_policy",
             "maneuver", "t_since_change", "abrupt_last_second", "omega_vert", "omega_horiz", "flip", "accel_norm",
             "speed", "sample_id", "seed", "k_q", "d_xy", "c_max", "delta",
             "fz_v", "fz_omega", "fz_c", "fz_omega_cmd", "fz_c_cmd")
REASONS = ("tilt", "thrust", "fall")
STICK_KEYS = ("v_xy", "tilt_xy", "yaw_rate", "vz", "omega", "c")


@dataclass
class GeneratedBatch:
    trajs: list[Trajectory]
    internals: dict[str, np.ndarray]                  # (n, 60, ...) at 30 FPS; hf_* at the integration rate
    n_rejected: dict[str, int]
    rejected_by: dict[str, dict] = field(default_factory=dict)   # "reason/mode/style" -> {"count": ...}
    n_candidates: int = 0


def omega_split(R: np.ndarray, omega_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angular velocity in G: component along the vertical (signed) and norm of the horizontal part."""
    w = np.einsum("...ij,...j->...i", R, omega_body)
    return w[..., 2], np.linalg.norm(w[..., :2], axis=-1)


def _draw_block(seed: int, block: int, cfg: dict) -> dict:
    """All random draws of one block of candidates (independent of the mismatch level)."""
    rng = np.random.default_rng([seed, block])
    n = cfg["generator"]["block"]
    dc, gc = cfg["dynamics"], cfg["generator"]
    setup = sample_setup(n, rng, cfg)
    U = lambda r, size=n: rng.uniform(r[0], r[1], size)
    fps = gc["fps"]
    return {"setup": setup, "k_d": U(dc["k_d"]), "tau_w": U(dc["tau_w"]), "tau_c": U(dc["tau_c"]),
            "k_q": U(dc["k_q"]), "d_xy": U(dc["d_xy"], (n, 2)), "delta": U(dc["delta"]),
            "warm": rng.integers(int(round(gc["warmup_s"][0] * fps)), int(round(gc["warmup_s"][1] * fps)) + 1, n),
            "psi0": rng.uniform(-np.pi, np.pi, n)}


def _draw(cand_ids: np.ndarray, seed: int, cfg: dict) -> dict:
    bs = cfg["generator"]["block"]
    blocks = {b: _draw_block(seed, int(b), cfg) for b in np.unique(cand_ids // bs)}
    parts = [(blocks[i // bs], i % bs) for i in cand_ids]
    out = {"setup": PilotSetup.concat([d["setup"].take([j]) for d, j in parts])}
    for k in ("k_d", "tau_w", "tau_c", "k_q", "d_xy", "delta", "warm", "psi0"):
        out[k] = np.stack([d[k][j] for d, j in parts])
    return out


def _apply_overrides(setup: PilotSetup, overrides: dict | None, cfg: dict) -> PilotSetup:
    if not overrides:
        return setup
    from dronepose.forecast.pilot import limits, style_of
    d = {k: getattr(setup, k).copy() for k in setup.__dataclass_fields__}
    for k, v in overrides.items():
        d[k] = np.broadcast_to(np.asarray(v, dtype=object if d[k].dtype.kind == "U" else d[k].dtype), d[k].shape).copy()
        if d[k].dtype.kind == "O":
            d[k] = d[k].astype("U16")
    d.update(limits(d["s"], d["drone_class"], cfg))
    d["style"] = style_of(d["s"], cfg)
    return PilotSetup(**d)


def _tilt(R):
    return np.arccos(np.clip(R[..., 2, 2], -1.0, 1.0))


def _py(x):
    x = np.asarray(x)
    return x.tolist() if x.ndim else x.item()


def _simulate(draw: dict, level: int, cfg: dict, freeze: bool, record_hf: bool, init_perturbation: float) -> dict:
    st: PilotSetup = draw["setup"]
    B = len(st.mode)
    dc, gc = cfg["dynamics"], cfg["generator"]
    dt, oe, ro = dc["dt"], dc["out_every"], dc["reorth_every"]
    FI, NF = gc["freeze_idx"], gc["n_frames"]
    warm = draw["warm"]
    qp = QuadParams(k_d=draw["k_d"], tau_w=draw["tau_w"], tau_c=draw["tau_c"], k_q=draw["k_q"], d_xy=draw["d_xy"],
                    c_max=st.c_max, delta=draw["delta"], level=np.full(B, level, int))
    c, s = np.cos(draw["psi0"]), np.sin(draw["psi0"])
    R0 = np.zeros((B, 3, 3))
    R0[:, 0, 0], R0[:, 0, 1], R0[:, 1, 0], R0[:, 1, 1], R0[:, 2, 2] = c, -s, s, c, 1.0
    state = QuadState(p=np.zeros((B, 3)), v=np.full((B, 3), float(init_perturbation)), R=R0,
                      omega=np.zeros((B, 3)), c=np.full(B, G))
    pilot = Pilot(st, cfg, qp.k_d)
    n_out = int(warm.max()) + NF - FI                       # frames 0 .. warm + 30
    n_steps = (n_out - 1) * oe
    rec = {k: [] for k in ("p", "v", "R", "omega", "c", "omega_cmd", "c_cmd", "abrupt")}
    rec_st = {k: [] for k in STICK_KEYS}
    hf = {k: [] for k in ("hf_v", "hf_c", "hf_b3", "hf_abrupt") + tuple(f"hf_sticks_{s}" for s in STICK_KEYS)} \
        if record_hf else None
    freeze_step = warm * oe
    c_lo, c_hi = np.full(B, np.inf), np.full(B, -np.inf)
    at_f = {}
    for k in range(n_steps + 1):
        t = k * dt
        w_cmd, c_cmd = pilot.command(t, state)
        past = k <= freeze_step
        c_lo = np.where(past, np.minimum(c_lo, c_cmd), c_lo)
        c_hi = np.where(past, np.maximum(c_hi, c_cmd), c_hi)
        if hf is not None:
            hf["hf_v"].append(state.v.copy())
            hf["hf_c"].append(state.c.copy())
            hf["hf_b3"].append(state.R[:, :, 2].copy())
            hf["hf_abrupt"].append(pilot.abrupt_last_second(t))
            for name in STICK_KEYS:
                hf[f"hf_sticks_{name}"].append(pilot.last_sticks[name].copy())
        if k % oe == 0:
            for name, val in (("p", state.p), ("v", state.v), ("R", state.R), ("omega", state.omega), ("c", state.c),
                              ("omega_cmd", w_cmd), ("c_cmd", c_cmd)):
                rec[name].append(np.array(val, float))
            rec["abrupt"].append(pilot.abrupt_last_second(t))
            for name in STICK_KEYS:
                rec_st[name].append(pilot.last_sticks[name].copy())
            now = freeze_step == k
            if now.any():
                for name, val in (("maneuver", pilot.maneuver), ("t_since_change", pilot.t_since_change(t)),
                                  ("abrupt_last_second", pilot.abrupt_last_second(t))):
                    at_f.setdefault(name, np.empty(B, dtype=val.dtype if val.dtype.kind != "U" else "U16"))[now] = val[now]
                if freeze:
                    pilot.freeze(now)
        if k == n_steps:
            break
        state = step(state, w_cmd, c_cmd, qp, dt)
        if (k + 1) % ro == 0:
            state = replace(state, R=orthonormalize(state.R))
    full = {k: np.stack(v, 1) for k, v in rec.items()}                 # (B, n_out, ...)
    full.update({f"sticks_{k}": np.stack(v, 1) for k, v in rec_st.items()})
    idx = (warm - FI)[:, None] + np.arange(NF)[None]                    # output window per candidate
    rows = np.arange(B)[:, None]
    out = {k: v[rows, idx] for k, v in full.items()}
    tilt_past = _tilt(full["R"])
    past_frames = np.arange(n_out)[None] <= warm[:, None]
    tilt_bad = (np.where(past_frames, tilt_past, 0).max(1) > st.max_tilt + np.radians(gc["tilt_margin_deg"]))
    lo, hi = gc["c_cmd_range_g"]
    thrust_bad = (c_lo < lo * G - 1e-9) | (c_hi > hi * G + 1e-9)
    z = out["p"][..., 2]
    flip = _tilt(out["R"][:, FI + 1:]).max(1) > np.pi / 2
    # fall below the altitude at the freeze (the spec's "quota iniziale" read as the start of the future: the
    # warm-up altitude would reject descents flown in the past); a flip also falls and is kept (spec §5.2)
    fall_bad = ((z[:, FI:] - z[:, FI:FI + 1]).min(1) < -gc["max_fall_m"]) & ~flip
    res = {"out": out, "qp": qp, "setup": st, "warm": warm, "at_f": at_f, "flip": flip,
           "bad": {"tilt": tilt_bad, "thrust": thrust_bad, "fall": fall_bad}}
    if hf is not None:
        H = {k: np.stack(v, 1) for k, v in hf.items()}
        hidx = (freeze_step - FI * oe)[:, None] + np.arange((NF - 1) * oe + 1)[None]
        res["hf"] = {k: v[rows, hidx] for k, v in H.items()}
    return res


def _make_trajs(res: dict, sel: np.ndarray, cand_ids: np.ndarray, seed: int, level: int, split: str,
                fps: int, FI: int) -> list[Trajectory]:
    out, qp, st, at_f = res["out"], res["qp"], res["setup"], res["at_f"]
    trajs = []
    for i in np.flatnonzero(sel):
        p = out["p"][i] - out["p"][i, 0]
        R = out["R"][i]
        v, w = out["v"][i, FI], out["omega"][i, FI]
        vert, horiz = omega_split(R[FI], w)
        s_f = QuadState(p=out["p"][i, FI][None], v=v[None], R=R[FI][None], omega=w[None], c=out["c"][i, FI][None])
        q_i = QuadParams(k_d=qp.k_d[[i]], tau_w=qp.tau_w[[i]], tau_c=qp.tau_c[[i]], k_q=qp.k_q[[i]], d_xy=qp.d_xy[[i]],
                         c_max=qp.c_max[[i]], delta=qp.delta[[i]], level=qp.level[[i]])
        meta = {"mode": str(st.mode[i]), "s": float(st.s[i]), "style": str(st.style[i]),
                "drone_class": str(st.drone_class[i]), "k_d": float(qp.k_d[i]), "tau_w": float(qp.tau_w[i]),
                "tau_c": float(qp.tau_c[i]), "tau_att": float(st.tau_att[i]), "level": int(level),
                "yaw_policy": str(st.yaw_policy[i]), "maneuver": str(at_f["maneuver"][i]),
                "t_since_change": float(at_f["t_since_change"][i]),
                "abrupt_last_second": bool(at_f["abrupt_last_second"][i]),
                "omega_vert": float(vert), "omega_horiz": float(horiz), "flip": bool(res["flip"][i]),
                "accel_norm": float(np.linalg.norm(acceleration(s_f, q_i)[0])), "speed": float(np.linalg.norm(v)),
                "sample_id": int(cand_ids[i]), "seed": int(seed), "k_q": float(qp.k_q[i]), "d_xy": _py(qp.d_xy[i]),
                "c_max": float(qp.c_max[i]), "delta": float(qp.delta[i]),
                "fz_v": _py(v), "fz_omega": _py(w), "fz_c": float(out["c"][i, FI]),
                "fz_omega_cmd": _py(out["omega_cmd"][i, FI]), "fz_c_cmd": float(out["c_cmd"][i, FI])}
        trajs.append(Trajectory(scene="gen", drone=meta["drone_class"], split=split, motion_type=meta["style"],
                                copies=[], s_c=1.0, times=np.arange(len(p)) / fps,
                                p_G=p.astype(np.float32), R_G=R.astype(np.float32), freeze_idx=FI, meta=meta))
    return trajs


def simulate_candidates(cand_ids, seed: int, level: int, cfg: dict, freeze: bool = True,
                        setup_overrides: dict | None = None, record_hf: bool = False,
                        init_perturbation: float = 0.0, split: str = "gen") -> dict:
    """Simulate the given candidate ids; returns trajectories of all of them, the rejection masks and internals."""
    cand_ids = np.asarray(cand_ids, int)
    gc = cfg["generator"]
    draw = _draw(cand_ids, seed, cfg)
    draw["setup"] = _apply_overrides(draw["setup"], setup_overrides, cfg)
    res = _simulate(draw, level, cfg, freeze, record_hf, init_perturbation)
    reject = np.zeros(len(cand_ids), bool)
    reason = np.full(len(cand_ids), "", dtype="U8")
    for r in REASONS:
        reason[res["bad"][r] & ~reject] = r
        reject |= res["bad"][r]
    trajs = _make_trajs(res, np.ones(len(cand_ids), bool), cand_ids, seed, level, split, gc["fps"], gc["freeze_idx"])
    internals = dict(res["out"])
    internals.update(res.get("hf", {}))
    return {"trajs": trajs, "reject": reject, "reason": reason, "internals": internals}


def iter_chunks(sim_fn, seed: int, level: int, cfg: dict, freeze: bool, start_id: int = 0, workers: int = 1, **kw):
    """Endless (ids, result) for consecutive chunks of candidate ids, in id order; workers > 1 = processes.

    The results do not depend on `workers`: each candidate's randomness is its own and the caller consumes in order.
    """
    chunk = cfg["generator"]["chunk"]
    starts = (start_id + k * chunk for k in itertools.count())
    if workers <= 1:
        for a in starts:
            ids = np.arange(a, a + chunk)
            yield ids, sim_fn(ids, seed, level, cfg, freeze, **kw)
        return
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        pending = deque()
        for _ in range(2 * workers):
            a = next(starts)
            ids = np.arange(a, a + chunk)
            pending.append((ids, pool.submit(sim_fn, ids, seed, level, cfg, freeze, **kw)))
        while True:
            ids, fut = pending.popleft()
            a = next(starts)
            nxt = np.arange(a, a + chunk)
            pending.append((nxt, pool.submit(sim_fn, nxt, seed, level, cfg, freeze, **kw)))
            yield ids, fut.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _subset(internals: dict, sel: np.ndarray) -> dict:
    return {k: v[sel] for k, v in internals.items()}


def generate(n: int, seed: int, level: int, cfg: dict, freeze: bool = True, cand_ids=None,
             keep_rejected: bool = False, setup_overrides: dict | None = None, record_hf: bool = False,
             init_perturbation: float = 0.0, split: str = "gen", start_id: int = 0, workers: int = 1) -> GeneratedBatch:
    """n accepted samples (candidates in id order from start_id), or exactly the given cand_ids."""
    chunk = cfg["generator"]["chunk"]
    trajs, internals, n_rej = [], [], {r: 0 for r in REASONS}
    rejected_by: dict[str, dict] = {}
    next_id, n_cand = start_id, 0
    fixed = cand_ids is not None
    kw = {"setup_overrides": setup_overrides, "record_hf": record_hf, "init_perturbation": init_perturbation,
          "split": split}
    chunks = iter_chunks(simulate_candidates, seed, level, cfg, freeze, start_id, workers, **kw) \
        if workers > 1 and not fixed else None
    while len(trajs) < n:
        if chunks is not None:
            ids, r = next(chunks)
        else:
            ids = np.asarray(cand_ids, int) if fixed else np.arange(next_id, next_id + min(chunk, max(2 * (n - len(trajs)), 16)))
            next_id = int(ids[-1]) + 1
            r = simulate_candidates(ids, seed, level, cfg, freeze, **kw)
        n_cand += len(ids)
        for i in np.flatnonzero(r["reject"]):
            n_rej[r["reason"][i]] += 1
            t = r["trajs"][i]
            key = f"{r['reason'][i]}/{t.meta['mode']}/{t.meta['style']}"
            rejected_by.setdefault(key, {"count": 0})["count"] += 1
        keep = np.ones(len(ids), bool) if keep_rejected else ~r["reject"]
        trajs += [t for t, k in zip(r["trajs"], keep) if k]
        internals.append(_subset(r["internals"], keep))
        if fixed:
            break
    if chunks is not None:
        chunks.close()
    m = len(trajs) if fixed else n
    internals = {k: np.concatenate([d[k] for d in internals])[:m] for k in internals[0]}
    return GeneratedBatch(trajs=trajs[:m], internals=internals, n_rejected=n_rej, rejected_by=rejected_by,
                          n_candidates=n_cand)
