"""E3 pure logic: solver x refine x strategy x class grid, selection rule, per-axis Euler errors, timing."""
import time
from dataclasses import dataclass

import numpy as np

from dronepose.common.geometry import rotmat_to_euler_zyx
from dronepose.pose.disambiguation import STRATEGIES
from dronepose.pose.solvers import SOLVERS, solve

CLASSES = ("gt", "causal")


@dataclass(frozen=True)
class Combo:
    solver: str
    refine: bool
    strategy: str
    cls: str

    @property
    def name(self) -> str:
        return f"{self.solver}{'+lm' if self.refine else ''}+{self.strategy}"

    @property
    def key(self) -> str:
        return f"{self.name}|{self.cls}"


def all_combos() -> list[Combo]:
    """64 combinations, ordered by solver, refine, strategy, class."""
    return [Combo(s, r, st, c) for s in SOLVERS for r in (False, True) for st in STRATEGIES for c in CLASSES]


def reference(cls: str) -> Combo:
    """Bootstrap reference: ippe + global, no refinement, same class."""
    return Combo("ippe", False, "global", cls)


def simplicity_key(c: Combo) -> tuple:
    """Lower is simpler: no refinement first, then the strategy order, then the solver order."""
    return (c.refine, STRATEGIES.index(c.strategy), SOLVERS.index(c.solver))


def choose_combo(means: dict[str, dict], boots: dict[str, list[tuple[float, float, float]]],
                 min_coverage: float = 0.95, max_flip_increase: float = 0.005) -> str:
    """Pick the combination name: best median_rot_sym among those with enough coverage. A combination ties
    with the best only if (a) its bootstrap interval against the best includes zero in every seed and
    (b) its flip rate without drone 08 (`flip_no08`, a fraction) exceeds the best's by at most
    `max_flip_increase`. The simplest tied combination (the best included) wins."""
    by_name = {c.name: c for c in all_combos()}
    ok = [n for n, m in means.items() if m["coverage"] >= min_coverage]
    if not ok:
        raise ValueError(f"no combination reaches coverage >= {min_coverage}")
    best = min(ok, key=lambda n: means[n]["median_rot_sym"])
    tied = [best]
    for n in ok:
        if n == best or n not in boots or not boots[n]:
            continue
        if (all(lo <= 0.0 <= hi for _, lo, hi in boots[n])
                and means[n]["flip_no08"] <= means[best]["flip_no08"] + max_flip_increase):
            tied.append(n)
    return min(tied, key=lambda n: simplicity_key(by_name[n]))


def euler_axis_errors(R_est: np.ndarray, R_gt: np.ndarray) -> np.ndarray:
    """|wrapped difference| of (yaw, pitch, roll) in degrees, in [0, 180]."""
    d = np.asarray(rotmat_to_euler_zyx(R_est)) - np.asarray(rotmat_to_euler_zyx(R_gt))
    return np.abs((d + 180.0) % 360.0 - 180.0)


def mae_medae(values: list[float]) -> tuple[float, float]:
    """Mean and median of `values`."""
    a = np.asarray(values, float)
    return float(a.mean()), float(np.median(a))


def solver_timing(records: list[dict], kpts_by_id: dict[str, np.ndarray], models: dict, solver: str,
                  refine: bool, model_drone: dict[str, str] | None = None, repeats: int = 1) -> dict:
    """Median and p90 (ms) of `solve` calls over frames with finite keypoints; every call is one sample,
    so n = frames * repeats. No disambiguation, only the `solve` call is inside the timer."""
    times = []
    for r in records:
        kp = kpts_by_id.get(r["frame_id"])
        if kp is None:
            continue
        pts = np.asarray(kp, float)[:, :2]
        if not np.isfinite(pts).all():
            continue
        drone = model_drone[r["frame_id"]] if model_drone else r["drone"]
        model = models[drone].kpts_flu
        K = np.asarray(r["K"], float)
        for _ in range(repeats):
            t0 = time.perf_counter()
            solve(solver, model, pts, K, refine=refine)
            times.append((time.perf_counter() - t0) * 1000.0)
    if not times:
        return {"n": 0, "median_ms": float("nan"), "p90_ms": float("nan")}
    return {"n": len(times), "median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}
