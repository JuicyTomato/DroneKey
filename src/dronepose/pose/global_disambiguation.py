"""Global (whole-sequence) choice between the two IPPE solutions of every frame, by dynamic programming.

Cost of a path = sum over accepted frames of reproj_px / reproj_scale
               + sum over consecutive accepted frames of min(z, z_cap)^2, z = geodesic_deg / (step_deg * dt)
               + reject_cost for every frame that is not accepted.
The path with the lowest total cost wins, so later frames vote on earlier ambiguous ones. Across a burst of rejected
frames the reference fades with the length of the burst: a jump of 180 degrees over dt frames costs about
(36 / dt)^2, so beyond roughly 20-30 frames the two chains become nearly equivalent (measured: a 14-frame burst is
bridged, 30 frames is not). Beyond max_span frames the chain restarts at the fixed cost z_cap^2.
"""
import math
from dataclasses import dataclass

import numpy as np

from dronepose.pose.disambiguation import Choice, reprojection_ratio


@dataclass(frozen=True)
class GlobalParams:
    step_deg: float = 5.0          # expected max rotation per frame (GT p99 on val: 4.5 deg)
    max_reproj_px: float = 20.0    # candidates above this mean reprojection error cannot be chosen
    reproj_scale: float = 10.0     # unary cost = reproj_px / reproj_scale
    reject_cost: float = 12.0      # cost of leaving a frame without a pose
    z_cap: float = 6.0             # normalised jump beyond which the chain counts as restarted
    max_span: int = 60             # frames: beyond this distance two poses are linked at the fixed cost z_cap^2
    max_tilt_deg: float = 70.0     # only with `up`: candidates tilted more than this from the vertical are excluded


def _tilt_deg(R: np.ndarray, up: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(R[:, 2] @ up)))))   # R[:, 2] = body "up" in camera frame


def disambiguate_global(candidates, params: GlobalParams = GlobalParams(), frame_idx=None, up=None) -> list[Choice]:
    n = len(candidates)
    if n == 0:
        return []
    t = np.arange(n, dtype=float) if frame_idx is None else np.asarray(frame_idx, dtype=float)
    if t.shape != (n,) or np.any(np.diff(t) <= 0):
        raise ValueError("frame_idx must have one strictly increasing entry per frame")
    if up is not None:
        up = np.asarray(up, dtype=float)
        if up.shape != (3,) or not np.linalg.norm(up) > 0:
            raise ValueError("up must be a non-zero 3-vector")
        up = up / np.linalg.norm(up)

    # allowed nodes (frame, candidate), in frame order
    nf, nk, nR, nu = [], [], [], []
    first = [0] * (n + 1)
    for j, sols in enumerate(candidates):
        first[j] = len(nf)
        for k, s in enumerate(sols):
            if s.reproj_px > params.max_reproj_px:
                continue
            if up is not None and _tilt_deg(s.R, up) > params.max_tilt_deg:
                continue
            nf.append(j)
            nk.append(k)
            nR.append(s.R)
            nu.append(s.reproj_px / params.reproj_scale)
    first[n] = len(nf)
    m = len(nf)
    fr = np.array(nf, dtype=int)
    R = np.asarray(nR, dtype=float).reshape(m, 3, 3)
    u = np.array(nu, dtype=float)
    tt = t[fr] if m else np.zeros(0)

    rc, zc = params.reject_cost, params.z_cap
    best = np.full(m, np.inf)
    prev = np.full(m, -1, dtype=int)
    pref_val = np.full(m, np.inf)       # prefix minimum of best - rc * frame (for links beyond max_span)
    pref_arg = np.full(m, -1, dtype=int)
    for j in range(n):
        q0, q1 = first[j], first[j + 1]
        if q0 == q1:
            continue
        w = q1 - q0
        cost_in = np.full(w, rc * j)            # path starts here: every earlier frame skipped, no transition cost
        arg_in = np.full(w, -1, dtype=int)
        lo = int(np.searchsorted(tt[:q0], t[j] - params.max_span, side="left"))
        if lo < q0:                              # predecessors inside the window
            tr = np.einsum("pij,qij->pq", R[lo:q0], R[q0:q1])
            ang = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
            dt = (t[j] - tt[lo:q0])[:, None]
            z = np.minimum(ang / (params.step_deg * dt), zc)
            tot = best[lo:q0, None] + z ** 2 + (j - fr[lo:q0] - 1)[:, None] * rc
            p = np.argmin(tot, axis=0)
            c = tot[p, np.arange(w)]
            better = c < cost_in
            cost_in = np.where(better, c, cost_in)
            arg_in = np.where(better, lo + p, arg_in)
        if lo > 0:                               # predecessors beyond the window: chain restarts at fixed cost
            c = pref_val[lo - 1] + rc * (j - 1) + zc ** 2
            better = c < cost_in
            cost_in = np.where(better, c, cost_in)
            arg_in = np.where(better, pref_arg[lo - 1], arg_in)
        best[q0:q1] = u[q0:q1] + cost_in
        prev[q0:q1] = arg_in
        for q in range(q0, q1):
            v = best[q] - rc * fr[q]
            if q > 0 and pref_val[q - 1] <= v:
                pref_val[q], pref_arg[q] = pref_val[q - 1], pref_arg[q - 1]
            else:
                pref_val[q], pref_arg[q] = v, q

    chosen: dict[int, int] = {}
    if m:
        end_cost = best + (n - 1 - fr) * rc
        q = int(np.argmin(end_cost))
        if end_cost[q] < n * rc:
            while q != -1:
                chosen[int(fr[q])] = q
                q = int(prev[q])

    out = []
    for j, sols in enumerate(candidates):
        ratio = reprojection_ratio(sols)
        if j in chosen:
            k = nk[chosen[j]]
            out.append(Choice(k, sols[k], ratio, False, None))
        else:
            reason = "no_solution" if not sols else ("reproj" if first[j] == first[j + 1] else "cost")
            out.append(Choice(None, None, ratio, True, reason))
    return out
