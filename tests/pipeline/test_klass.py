import numpy as np
import pytest

from dronepose.data.sixdronesyn import DRONE_IDS
from dronepose.keypoints.class_scores import combine, fuse_causal, log_scores, scores_geom
from dronepose.pipeline.klass import CausalClassFuser, klg_frame_scores, load_klg_params

P = {"a_min": 0.1, "w_g": 1.0, "form": "log"}


def _rand_frames(n, seed=0):
    rng = np.random.default_rng(seed)
    probs, reproj = {}, {}
    for i in range(n):
        fid = f"14_02_07_000_{i + 1:04d}"
        p = rng.dirichlet(np.ones(7) * 0.3)
        probs[fid] = None if i in (0, 1, 7) else {d: float(v) for d, v in zip(DRONE_IDS, p)}
        reproj[fid] = None if i in (0, 1, 9) else {d: float(v) for d, v in zip(DRONE_IDS, rng.uniform(0, 30, 7))}
    return probs, reproj


def test_klg_matches_batch_recipe():
    probs, reproj = _rand_frames(12)
    k = {f: p for f, p in probs.items() if p is not None}
    batch = combine([(combine([(log_scores(k, 0.1), 1.0)]), 1.0), (scores_geom(reproj, "log"), 1.0)])
    for fid in probs:
        got = klg_frame_scores(probs[fid], reproj[fid], P)
        if fid not in batch:
            assert got is None
        else:
            assert got.keys() == batch[fid].keys()
            assert all(abs(got[c] - batch[fid][c]) < 1e-12 for c in got)


def test_fuser_matches_fuse_causal():
    probs, reproj = _rand_frames(20, seed=3)
    recs = [{"frame_id": f, "frame_idx": i + 1} for i, f in enumerate(probs)]
    scores = {f: s for f in probs if (s := klg_frame_scores(probs[f], reproj[f], P)) is not None}
    batch = fuse_causal(recs, scores, "S")
    fuser = CausalClassFuser()
    out = [fuser.update(scores.get(r["frame_id"])) for r in recs]
    first = next(i for i, c in enumerate(out) if c is not None)
    assert all(c is None for c in out[:first])
    assert out[first:] == [batch[r["frame_id"]] for r in recs[first:]]


def test_fuser_none_before_first_score():
    f = CausalClassFuser()
    assert f.update(None) is None
    assert f.update({"03": 1.0, "01": 0.0}) == "03"
    assert f.update(None) == "03"


def test_load_klg_params():
    assert load_klg_params() == P
