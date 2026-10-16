"""Phase 3 metrics: ADE/FDE per horizon, orientation errors, skill vs a reference and paired bootstrap per trajectory."""
import numpy as np


def position_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(pred, float) - np.asarray(gt, float), axis=-1)


def ade_at(err: np.ndarray, h: int) -> np.ndarray:
    """Mean error over the first h predicted steps, per window."""
    return np.asarray(err)[:, :h].mean(axis=1)


def fde_at(err: np.ndarray, h: int) -> np.ndarray:
    """Error at predicted step h (1-based), per window."""
    return np.asarray(err)[:, h - 1]


def geodesic_deg_batch(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    cos = (np.einsum("...ji,...ji->...", np.asarray(Ra, float), np.asarray(Rb, float)) - 1) / 2   # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def up_angle_deg(ua: np.ndarray, ub: np.ndarray) -> np.ndarray:
    ua, ub = np.asarray(ua, float), np.asarray(ub, float)
    cos = np.sum(ua * ub, -1) / (np.linalg.norm(ua, axis=-1) * np.linalg.norm(ub, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def skill(err_method: np.ndarray, err_ref: np.ndarray) -> float:
    """Relative improvement over the reference (on the mean error): 1 - sum(method) / sum(ref)."""
    return float(1 - np.sum(err_method) / np.sum(err_ref))


def paired_bootstrap_skill(err_method, err_ref, traj, strata, b: int, seed: int) -> tuple[float, float, float]:
    """Skill with a 95% percentile interval; trajectories resampled with replacement within each stratum (scene).

    Both methods are evaluated on the same resampled trajectories (paired).
    """
    err_method, err_ref = np.asarray(err_method, float), np.asarray(err_ref, float)
    ids, inv = np.unique(np.asarray(traj), return_inverse=True)
    S_m = np.bincount(inv, weights=err_method, minlength=len(ids))
    S_r = np.bincount(inv, weights=err_ref, minlength=len(ids))
    traj_stratum = np.empty(len(ids), dtype=object)
    traj_stratum[inv] = np.asarray(strata)
    rng = np.random.default_rng(seed)
    counts = np.zeros((b, len(ids)))
    rows = np.arange(b)[:, None]
    for s in sorted(set(traj_stratum)):
        members = np.flatnonzero(traj_stratum == s)
        draws = members[rng.integers(0, len(members), size=(b, len(members)))]
        np.add.at(counts, (np.broadcast_to(rows, draws.shape), draws), 1)
    boot = 1 - (counts @ S_m) / (counts @ S_r)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return skill(err_method, err_ref), float(lo), float(hi)


def metric_group(windows) -> np.ndarray:
    """The only place deciding which windows enter averages in metres (drones with a verified size)."""
    return np.asarray(windows.metric, bool)
