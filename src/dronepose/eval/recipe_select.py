"""Choice of the training recipe on the authors' train split of MAV6D (spec §7): the rule is fixed before the data."""
import math

import numpy as np

from dronepose.eval.bootstrap import trajectory_of

MAX_COVERAGE_DROP = 0.05
MAX_FLIP_RISE = 0.03


def paired_groups(rot_ref: dict, rot_cand: dict) -> dict[str, dict[str, list[float]]]:
    """Rotation errors per trajectory, only on frames with a pose in both models."""
    groups: dict[str, dict[str, list[float]]] = {"ref": {}, "cand": {}}
    for frame, a in rot_ref.items():
        b = rot_cand.get(frame)
        if a is None or b is None or math.isnan(a) or math.isnan(b):
            continue
        t = trajectory_of(frame)
        groups["ref"].setdefault(t, []).append(float(a))
        groups["cand"].setdefault(t, []).append(float(b))
    return groups


def paired_median(rot_a: dict, rot_b: dict) -> tuple[float, float, int]:
    """(median of a, median of b, frames) on the frames with a pose in both: a model that drops the hard frames gains nothing."""
    g = paired_groups(rot_a, rot_b)
    a = [v for vs in g["ref"].values() for v in vs]
    b = [v for vs in g["cand"].values() for v in vs]
    return (float(np.median(a)), float(np.median(b)), len(a)) if a else (math.nan, math.nan, 0)


def wins(boot: tuple[float, float, float], cov_ref: float, cov_cand: float, flip_ref: float, flip_cand: float) -> bool:
    """`boot` = (diff, lo, hi) of cand - ref: lower rotation with zero excluded, coverage and flips within the limits."""
    return bool(boot[2] < 0 and cov_ref - cov_cand <= MAX_COVERAGE_DROP + 1e-9
                and flip_cand - flip_ref <= MAX_FLIP_RISE + 1e-9)


def best_of(names: list[str], median: dict[str, float]) -> str:
    return min(names, key=lambda n: median[n])


def decide(step1: dict, step2: dict, combo: dict | None) -> dict:
    """Steps 1-4 of spec §7: `stepN` holds `winner` (and `median` when there is one), `combo` holds `name` and `wins`."""
    a, b = step1.get("winner"), step2.get("winner")
    if a is None and b is None:
        return {"recipe": None, "stop": True, "combo_needed": False}
    if a is None or b is None:
        return {"recipe": a or b, "stop": False, "combo_needed": False}
    if combo is None:
        return {"recipe": None, "stop": False, "combo_needed": True}
    if combo["wins"]:
        return {"recipe": combo["name"], "stop": False, "combo_needed": False}
    best = best_of([a, b], {a: step1["median"], b: step2["median"]})
    return {"recipe": best, "stop": False, "combo_needed": False}
