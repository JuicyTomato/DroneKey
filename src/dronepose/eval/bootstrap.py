"""Paired bootstrap over trajectories (scene x drone) for differences of medians (or means)."""
import numpy as np


def trajectory_of(frame_id: str) -> str:
    """Trajectory id (scene_drone) of a frame id, e.g. '14_03_10_000_0005' -> '14_03'."""
    return "_".join(frame_id.split("_")[:2])


def by_trajectory(results: list[dict], key: str) -> dict[str, list[float]]:
    """Non-None values of `key` grouped by trajectory."""
    out: dict[str, list[float]] = {}
    for r in results:
        if r.get(key) is not None:
            out.setdefault(trajectory_of(r["frame_id"]), []).append(float(r[key]))
    return out


def paired_bootstrap(groups: dict[str, dict[str, list[float]]], ref: str, n_boot: int = 2000,
                     seed: int = 0, stat: str = "median") -> dict[str, tuple[float, float, float]]:
    """Per name: (observed `stat` diff vs ref, 2.5th, 97.5th percentile); one resample shared by all names.
    `stat` is "median" (default) or "mean" (e.g. accuracy over 0/1 values)."""
    if stat not in ("median", "mean"):
        raise ValueError(f"unknown stat {stat!r}: expected 'median' or 'mean'")
    agg = np.median if stat == "median" else np.mean
    trajs = sorted(set.intersection(*(set(g) for g in groups.values())))
    rng = np.random.default_rng(seed)

    def med(name: str, ts) -> float:
        return float(agg(np.concatenate([groups[name][t] for t in ts])))

    diffs: dict[str, list[float]] = {n: [] for n in groups}
    for _ in range(n_boot):
        pick = rng.choice(trajs, size=len(trajs), replace=True)
        m = {n: med(n, pick) for n in groups}
        for n in groups:
            diffs[n].append(m[n] - m[ref])
    out = {}
    for n in groups:
        lo, hi = np.percentile(diffs[n], [2.5, 97.5])
        out[n] = (med(n, trajs) - med(ref, trajs), float(lo), float(hi))
    return out
