"""Method selection on the val for E4a / E4b (spec 2026-10-27 §6, the user's rule adapted).

(1) best = lowest mean ADE at 1 s over the metric windows; (2) for every other candidate a paired bootstrap of the
difference with the best (trajectories resampled within each stratum, scene); (3) among the best and the candidates
whose 95% interval contains zero, the simplest in SIMPLICITY wins.
"""
import numpy as np

from dronepose.forecast.metrics import bootstrap_counts

SIMPLICITY = ("cv", "ca", "kf_cv", "kf_ca", "kf_ca_los", "physics", "R-CV-6D", "R-CV-FT", "G-6D")


def select_method(errors: dict[str, np.ndarray], traj, strata, order=SIMPLICITY, b: int = 10000,
                  seed: int = 0) -> dict:
    """errors: method -> ADE at 1 s per window (same windows for every method)."""
    unknown = set(errors) - set(order)
    if unknown:
        raise ValueError(f"methods without a simplicity rank: {sorted(unknown)}")
    inv, counts = bootstrap_counts(traj, strata, b, seed)
    n = counts.shape[1]
    sums = {m: np.bincount(inv, weights=np.asarray(e, float), minlength=n) for m, e in errors.items()}
    n_win = counts @ np.bincount(inv, minlength=n)
    mean = {m: float(np.mean(e)) for m, e in errors.items()}
    best = min(mean, key=mean.get)
    diffs, ties = {}, []
    for m in errors:
        if m == best:
            continue
        boot = (counts @ (sums[m] - sums[best])) / n_win
        lo, hi = np.percentile(boot, [2.5, 97.5])
        diffs[m] = {"mean_diff": mean[m] - mean[best], "ci": [float(lo), float(hi)]}
        if lo <= 0 <= hi:
            ties.append(m)
    chosen = min([best] + ties, key=order.index)
    return {"chosen": chosen, "best": best, "ties": sorted(ties, key=order.index), "ade_1s": mean,
            "diff_vs_best": diffs, "order": list(order)}
