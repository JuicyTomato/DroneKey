"""Stratified state at the freeze (spec §5.3) and importance weights back to the natural distribution.

Cells: terciles of five freeze-state variables (|omega_vert|, |omega_horiz|, |a|, |v|, time since the last
intention change), 3^5 = 243 cells, edges and natural probabilities from an auxiliary set generated without
stratification. Acceptance depends only on the cell (and on the order of the candidates): each occupiable cell up
to a common cap, then, if the attempt budget runs out first, a top-up with the next candidates in stream order
that were refused only by the cap. Within a cell the samples stay natural, so p_nat / p_strat weights are unbiased.
"""
import math

import numpy as np

from dronepose.forecast.generator import REASONS, GeneratedBatch, iter_chunks, simulate_candidates

STRAT_VARS = ("omega_vert", "omega_horiz", "accel_norm", "speed", "t_since_change")
OMEGA_GROUPS = ("|ω_vert|", "|ω_oriz|")        # the first two cell variables, named as in the report
LEVEL_NAMES = ("basso", "medio", "alto")


def _values(meta: dict) -> np.ndarray:
    return np.array([abs(meta[v]) if v == "omega_vert" else meta[v] for v in STRAT_VARS], float)


def cell_edges(meta_aux: list[dict]) -> dict[str, tuple[float, float]]:
    X = np.stack([_values(m) for m in meta_aux])
    q = np.quantile(X, [1 / 3, 2 / 3], axis=0)
    return {v: (float(q[0, i]), float(q[1, i])) for i, v in enumerate(STRAT_VARS)}


def _level(x: float, edges: tuple[float, float]) -> int:
    lv = int(np.searchsorted(edges, x, side="right"))
    return 1 if edges[0] == edges[1] == x else lv       # tied edges (a point mass): the tied value is the middle level


def cell_of(meta: dict, edges: dict) -> tuple[int, ...]:
    x = _values(meta)
    return tuple(_level(x[i], edges[v]) for i, v in enumerate(STRAT_VARS))


def cell_key(cell) -> str:
    return "".join(str(c) for c in cell)


def natural_cell_probs(meta_aux: list[dict], edges: dict) -> dict[tuple, float]:
    cells = [cell_of(m, edges) for m in meta_aux]
    uniq, counts = np.unique(np.array(cells), axis=0, return_counts=True)
    return {tuple(int(c) for c in u): float(k / len(cells)) for u, k in zip(uniq, counts)}


def importance_weights(cells: list[tuple], probs: dict[tuple, float]) -> np.ndarray:
    """p_nat(cell) / p_strat(cell), mean 1; 0 for cells the natural distribution never produces."""
    keys = [tuple(c) for c in cells]
    uniq, inv, counts = np.unique(np.array(keys), axis=0, return_inverse=True, return_counts=True)
    p_strat = counts / len(keys)
    p_nat = np.array([probs.get(tuple(int(c) for c in u), 0.0) for u in uniq])
    w = (p_nat / p_strat)[inv.reshape(-1)]
    return w / w.mean() if w.sum() > 0 else w


class _Selector:
    def __init__(self, n: int, probs: dict):
        self.n, self.probs = n, probs
        self.occupiable = [c for c, p in probs.items() if p > 0]
        self.cap = math.ceil(n / max(len(self.occupiable), 1))
        self.count: dict[tuple, int] = {}
        self.accepted: list = []
        self.reserve: list = []

    def offer(self, cell, item) -> None:
        if self.probs.get(cell, 0) <= 0 or len(self.accepted) >= self.n:
            return
        if self.count.get(cell, 0) < self.cap:
            self.count[cell] = self.count.get(cell, 0) + 1
            self.accepted.append((cell, item))
        elif len(self.reserve) < self.n:
            self.reserve.append((cell, item))

    def full(self) -> bool:
        return len(self.accepted) >= self.n

    def top_up(self) -> int:
        k = min(self.n - len(self.accepted), len(self.reserve))
        for cell, item in self.reserve[:k]:
            self.count[cell] = self.count.get(cell, 0) + 1
            self.accepted.append((cell, item))
        return k


def select_stratified(cells: list[tuple], n: int, probs: dict) -> list[int]:
    """Indices of a stratified selection of n from a stream of candidate cells (same rule as the generator)."""
    sel = _Selector(n, probs)
    for i, c in enumerate(cells):
        sel.offer(tuple(c), i)
        if sel.full():
            break
    sel.top_up()
    return sorted(i for _, i in sel.accepted)


def stratified_generate(n: int, seed: int, level: int, cfg: dict, edges: dict, probs: dict, min_per_group: int,
                        max_attempts: int, freeze: bool = True, sim_fn=simulate_candidates, workers: int = 1,
                        **sim_kw) -> tuple[GeneratedBatch, dict]:
    sel = _Selector(n, probs)
    attempts = 0
    n_rej = {r: 0 for r in REASONS}
    rejected_by: dict[str, dict] = {}
    by_omega = {f"{name} {lv}": {"candidates": 0, "rejected": {}} for name in OMEGA_GROUPS for lv in LEVEL_NAMES}
    chunks = iter_chunks(sim_fn, seed, level, cfg, freeze, workers=workers, **sim_kw)
    while not sel.full() and attempts < max_attempts:
        ids, r = next(chunks)
        attempts += len(ids)
        for i, t in enumerate(r["trajs"]):
            cell = cell_of(t.meta, edges)
            groups = [by_omega[f"{name} {LEVEL_NAMES[cell[j]]}"] for j, name in enumerate(OMEGA_GROUPS)]
            for g in groups:
                g["candidates"] += 1
            if r["reject"][i]:
                n_rej[r["reason"][i]] += 1
                key = f"{r['reason'][i]}/{t.meta['mode']}/{t.meta['style']}"
                rejected_by.setdefault(key, {"count": 0})["count"] += 1
                for g in groups:
                    g["rejected"][r["reason"][i]] = g["rejected"].get(r["reason"][i], 0) + 1
            else:
                sel.offer(cell, t)
    chunks.close()
    topped = sel.top_up() if not sel.full() else 0
    pairs = sorted(sel.accepted, key=lambda ct: ct[1].meta["sample_id"])
    cells = [c for c, _ in pairs]
    trajs = [t for _, t in pairs]
    w = importance_weights(cells, probs) if cells else np.zeros(0)
    for t, c, wi in zip(trajs, cells, w):
        t.meta["weight"] = float(wi)
        t.meta["cell"] = cell_key(c)
    per_level = {v: [int(sum(1 for c in cells if c[i] == k)) for k in range(3)] for i, v in enumerate(STRAT_VARS)}
    counts = {c: cells.count(c) for c in set(cells)}
    report = {"n": len(trajs), "n_requested": n, "attempts": attempts, "cap": sel.cap,
              "occupiable": len(sel.occupiable), "filled": sum(1 for c in sel.occupiable if counts.get(c, 0) >= sel.cap),
              "underfilled": sorted(cell_key(c) for c in sel.occupiable if 0 < counts.get(c, 0) < sel.cap),
              "uncovered": sorted(cell_key(c) for c in sel.occupiable if counts.get(c, 0) == 0),
              "topped_up": topped, "per_level": per_level, "min_per_group": min_per_group,
              "quota_ok": all(min(x) >= min_per_group for x in per_level.values()),
              "zero_natural_cells": sorted(cell_key(c) for c in set(cells) if probs.get(c, 0) <= 0),
              "n_rejected": n_rej, "rejected_by": rejected_by, "by_omega": by_omega}
    return GeneratedBatch(trajs=trajs, internals={}, n_rejected=n_rej, rejected_by=rejected_by,
                          n_candidates=attempts), report
