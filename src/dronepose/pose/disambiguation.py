"""Choosing between the two IPPE solutions of each frame:
(a) 'reproj'          lowest reprojection error, frame by frame;
(b) 'temporal'        closest rotation to the last accepted pose, with re-anchoring on clear reprojection evidence;
(c) 'temporal_reject' (b) plus rejection of frames whose choice jumps away or reprojects badly.
(d) 'global'          dynamic programming over the whole sequence (see global_disambiguation.py).
"""
import math
from dataclasses import dataclass

import numpy as np

from dronepose.common.geometry import geodesic_deg
from dronepose.pose.pnp import PnPSolution

STRATEGIES = ("reproj", "temporal", "temporal_reject", "global")


@dataclass(frozen=True)
class TemporalParams:
    reanchor_ratio: float = 3.0   # reprojection favours index 0 at least this much...
    reanchor_frames: int = 3      # ...for this many consecutive frames -> follow reprojection again
    max_gap: int = 5              # frames without an accepted pose before the reference is forgotten
    max_jump_deg: float = 20.0    # (c) max rotation change per frame w.r.t. the last accepted pose
    max_reproj_px: float = 20.0   # (c) max mean reprojection error of the chosen solution


@dataclass
class Choice:
    index: int | None
    solution: PnPSolution | None
    ratio: float
    rejected: bool
    reason: str | None


def reprojection_ratio(sols: list[PnPSolution]) -> float:
    if len(sols) < 2:
        return math.inf
    return sols[1].reproj_px / max(sols[0].reproj_px, 1e-9)


def disambiguate(candidates: list[list[PnPSolution]], strategy: str = "reproj", params=None,
                 frame_idx=None, up=None) -> list[Choice]:
    """`params`: TemporalParams for the three greedy strategies, GlobalParams for 'global'; None = the default of the
    strategy. `frame_idx` (real frame numbers) and `up` (camera-frame vertical) are used only by 'global'."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    if strategy == "global":
        from dronepose.pose.global_disambiguation import GlobalParams, disambiguate_global  # lazy: avoids a circular import
        params = GlobalParams() if params is None else params
        if not isinstance(params, GlobalParams):
            raise TypeError(f"strategy 'global' needs GlobalParams, got {type(params).__name__}")
        return disambiguate_global(candidates, params, frame_idx, up)
    params = TemporalParams() if params is None else params
    if not isinstance(params, TemporalParams):
        raise TypeError(f"strategy {strategy!r} needs TemporalParams, got {type(params).__name__}")
    out: list[Choice] = []
    prev_R, gap, streak = None, 0, 0

    def skip(choice: Choice) -> None:
        nonlocal prev_R, gap, streak
        out.append(choice)
        gap += 1
        if gap > params.max_gap:
            prev_R, streak = None, 0

    for sols in candidates:
        ratio = reprojection_ratio(sols)
        if not sols:
            skip(Choice(None, None, ratio, True, "no_solution"))
            continue
        k = 0
        if strategy != "reproj" and prev_R is not None:
            k = int(np.argmin([geodesic_deg(s.R, prev_R) for s in sols]))
            streak = streak + 1 if (k != 0 and ratio >= params.reanchor_ratio) else 0
            if streak >= params.reanchor_frames:
                k, streak = 0, 0
        sol = sols[k]
        if strategy == "temporal_reject":
            reason = None
            if sol.reproj_px > params.max_reproj_px:
                reason = "reproj"
            elif prev_R is not None and geodesic_deg(sol.R, prev_R) > params.max_jump_deg * (gap + 1):
                reason = "jump"
            if reason:
                skip(Choice(k, sol, ratio, True, reason))
                continue
        out.append(Choice(k, sol, ratio, False, None))
        prev_R, gap = sol.R, 0
    return out
