import numpy as np

from dronepose.forecast.stratify import (STRAT_VARS, cell_edges, cell_of, importance_weights, natural_cell_probs,
                                         select_stratified, stratified_generate)


def synthetic_meta(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(size=(n, 5))
    x[:, 1] = x[:, 1] * (1 + 3 * x[:, 0])         # correlated variables: some joint cells rarer than others
    return [{"omega_vert": float(-x[i, 0] if i % 2 else x[i, 0]), "omega_horiz": float(x[i, 1]),
             "accel_norm": float(x[i, 2]), "speed": float(x[i, 3]), "t_since_change": float(x[i, 4])} for i in range(n)]


class FakeSim:
    """Stand-in for simulate_candidates: plausible candidates with synthetic meta, a few rejected."""

    def __init__(self, seed=0):
        self.seed, self.calls = seed, 0

    def __call__(self, cand_ids, seed, level, cfg, freeze=True, split="gen"):
        from dronepose.forecast.dataset import Trajectory
        self.calls += 1
        metas = synthetic_meta(len(cand_ids), int(cand_ids[0]) + 1000 * seed)
        trajs = []
        for i, m in zip(cand_ids, metas):
            m = dict(m, sample_id=int(i), mode="angle", style="normale")
            trajs.append(Trajectory(scene="gen", drone="03", split="gen", motion_type="normale", copies=[], s_c=1.0,
                                    times=np.arange(60) / 30, p_G=np.zeros((60, 3), np.float32),
                                    R_G=np.repeat(np.eye(3, dtype=np.float32)[None], 60, 0), freeze_idx=29, meta=m))
        reject = np.asarray(cand_ids) % 17 == 0
        return {"trajs": trajs, "reject": reject, "reason": np.where(reject, "tilt", ""),
                "internals": {"p": np.zeros((len(cand_ids), 60, 3))}}


CFG = {"generator": {"chunk": 256}}


def test_cell_edges_are_terciles():
    meta = synthetic_meta(3000, 1)
    edges = cell_edges(meta)
    assert set(edges) == set(STRAT_VARS)
    x = np.abs([m["omega_vert"] for m in meta])
    lo, hi = edges["omega_vert"]
    assert abs((x < lo).mean() - 1 / 3) < 0.01 and abs((x >= hi).mean() - 1 / 3) < 0.01
    cells = [cell_of(m, edges) for m in meta]
    assert all(len(c) == 5 and set(c) <= {0, 1, 2} for c in cells)
    probs = natural_cell_probs(meta, edges)
    assert abs(sum(probs.values()) - 1) < 1e-12 and len(probs) <= 243


def test_stratified_quota_per_group():
    aux = synthetic_meta(5000, 2)
    edges, probs = cell_edges(aux), None
    probs = natural_cell_probs(aux, edges)
    batch, report = stratified_generate(300, 7, 1, CFG, edges, probs, min_per_group=30, max_attempts=20000,
                                        sim_fn=FakeSim())
    assert len(batch.trajs) == 300
    for var in STRAT_VARS:
        assert min(report["per_level"][var]) >= 30, (var, report["per_level"][var])
    assert report["quota_ok"]
    assert all("weight" in t.meta for t in batch.trajs)
    assert report["n_rejected"]["tilt"] > 0


def test_stratify_stops_on_attempt_cap():
    aux = synthetic_meta(5000, 3)
    edges = cell_edges(aux)
    probs = natural_cell_probs(aux, edges)
    batch, report = stratified_generate(2000, 7, 1, CFG, edges, probs, min_per_group=200, max_attempts=300,
                                        sim_fn=FakeSim())
    assert report["attempts"] <= 300 + CFG["generator"]["chunk"]
    assert report["uncovered"]
    assert len(batch.trajs) < 2000


def test_importance_weights_finite_and_normalised():
    probs = {(0,) * 5: 0.7, (1,) * 5: 0.3}
    cells = [(0,) * 5] * 10 + [(1,) * 5] * 10 + [(2,) * 5] * 5     # last cell has natural probability 0
    w = importance_weights(cells, probs)
    assert np.isfinite(w).all()
    assert abs(w.mean() - 1) < 1e-12
    assert (w[-5:] == 0).all()
    np.testing.assert_allclose(w[0] / w[10], 0.7 / 0.3)


def test_weighted_mean_recovers_natural_mean():
    pool = synthetic_meta(40000, 4)
    aux = synthetic_meta(20000, 5)
    edges = cell_edges(aux)
    probs = natural_cell_probs(aux, edges)
    cells = [cell_of(m, edges) for m in pool]
    idx = select_stratified(cells, 3000, probs)
    w = importance_weights([cells[i] for i in idx], probs)
    x = np.array([pool[i]["omega_horiz"] for i in idx])
    natural = np.mean([m["omega_horiz"] for m in pool])
    assert abs(np.average(x, weights=w) - natural) / natural < 0.05
    assert abs(x.mean() - natural) / natural > 0.1            # the stratified set alone is biased


def test_parallel_matches_serial():
    aux = synthetic_meta(5000, 6)
    edges = cell_edges(aux)
    probs = natural_cell_probs(aux, edges)
    a, ra = stratified_generate(400, 3, 1, CFG, edges, probs, 10, 4000, sim_fn=FakeSim())
    b, rb = stratified_generate(400, 3, 1, CFG, edges, probs, 10, 4000, sim_fn=FakeSim(), workers=2)
    assert [t.meta for t in a.trajs] == [t.meta for t in b.trajs]
    assert ra["attempts"] == rb["attempts"]


def test_cell_of_tied_edges_puts_the_tie_in_the_middle():
    # a variable with a point mass (e.g. many exact zeros) can give two equal tercile edges: the tied value is
    # neither "low" nor "high"
    edges = {v: (1.0, 2.0) for v in STRAT_VARS}
    edges["speed"] = (0.0, 0.0)
    meta = {"omega_vert": 1.5, "omega_horiz": 1.5, "accel_norm": 1.5, "speed": 0.0, "t_since_change": 1.5}
    assert cell_of(meta, edges) == (1, 1, 1, 1, 1)
    assert cell_of(dict(meta, speed=0.3), edges)[3] == 2
    assert cell_of(dict(meta, speed=2.0), {**edges, "speed": (1.0, 2.0)})[3] == 2    # untied edges: as before


def test_rejections_by_omega_group():
    aux = synthetic_meta(5000, 2)
    edges = cell_edges(aux)
    probs = natural_cell_probs(aux, edges)
    _, report = stratified_generate(300, 7, 1, CFG, edges, probs, 30, 20000, sim_fn=FakeSim())
    by = report["by_omega"]
    assert set(by) == {f"{v} {lv}" for v in ("|ω_vert|", "|ω_oriz|") for lv in ("basso", "medio", "alto")}
    for v in ("|ω_vert|", "|ω_oriz|"):
        groups = [by[f"{v} {lv}"] for lv in ("basso", "medio", "alto")]
        assert sum(g["candidates"] for g in groups) == report["attempts"]
        assert sum(g["rejected"].get("tilt", 0) for g in groups) == report["n_rejected"]["tilt"]
