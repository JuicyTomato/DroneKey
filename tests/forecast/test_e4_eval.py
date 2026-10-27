import numpy as np

from dronepose.forecast.e4_eval import (REFERENCE, degradation, method_errors, pipeline_errors, select_on,
                                        summarize_e4)
from dronepose.forecast.windows import make_windows
from tests.forecast.test_e4 import CFG, _pose_file, _traj
from tests.forecast.test_gaps import PARAMS

LOS = {**PARAMS, "kf_ca_los": {"q": 1e3, "r_los": 1e-2, "r_lat": 1e-4}}


def _trajs():
    return [_traj(scene=s, drone=d) for s in ("03", "09") for d in ("01", "02")] + \
           [_traj(scene="03", drone="08", s_c=None)]


def test_method_errors_reference_and_shapes():
    w = make_windows(_trajs(), CFG.obs_len, CFG.pred_len, CFG.fps)
    err, rot = method_errors(w, PARAMS, CFG)
    assert set(err) == {"cv", "ca", "kf_cv", "kf_ca", "physics", REFERENCE}
    assert all(e.shape == (len(w), 30) for e in err.values())
    np.testing.assert_allclose(err[REFERENCE], err["cv"])                 # cv_k = 2 in PARAMS
    assert set(rot) == {"rot_hold", "rot_const_omega"}


def test_summary_selection_degradation():
    w = make_windows(_trajs(), CFG.obs_len, CFG.pred_len, CFG.fps)
    err, rot = method_errors(w, PARAMS, CFG)
    s = summarize_e4(err, rot, w, CFG, b=200)
    names = [g["group"] for g in s["groups"]]
    assert names[0] == "metri: tutti" and "drone 08 (unità dataset)" in names
    assert REFERENCE not in s["groups"][0]["skill_vs_CV-2"]
    sel = select_on(err, w, ("cv", "ca", "kf_ca"), b=200)
    assert sel["chosen"] in ("cv", "ca", "kf_ca")
    d = degradation(err, err, w, b=200)
    assert all(v["mean"] == 0 for v in d.values())


def test_pipeline_errors_exact_poses_equal_e4a(tmp_path):
    trajs = _trajs()
    path = _pose_file(tmp_path, trajs)
    gt = make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)
    err_a, _ = method_errors(gt, LOS, CFG, los=True)
    err_b, rot_b, idx, info = pipeline_errors(trajs, [path], CFG, lambda s, m: LOS, lambda s: {})
    assert info["n_windows"] == len(idx) and info["n_dropped"] == 0
    for k in err_b:
        np.testing.assert_allclose(err_b[k], err_a[k][idx], atol=1e-9, err_msg=k)


def test_align_window_rows_pairs_by_key():
    from dronepose.forecast.e4_eval import align_window_rows
    a = [(0, "bg_000", 5), (0, "bg_090", 5), (0, "bg_000", 7), (1, "bg_000", 5)]
    b = [(0, "bg_090", 5), (1, "bg_000", 5), (0, "bg_000", 7)]      # (0, bg_000, 5) dropped in b
    ia, ib = align_window_rows(a, b)
    assert [a[i] for i in ia] == [b[i] for i in ib] == sorted(b)
    import pytest
    with pytest.raises(ValueError, match="unique"):
        align_window_rows(a + a[:1], b)


def test_pipeline_errors_keys_option(tmp_path):
    trajs = [_traj()]
    path = _pose_file(tmp_path, trajs)
    out = pipeline_errors(trajs, [path], CFG, lambda s, m: LOS, lambda s: {})
    keyed = pipeline_errors(trajs, [path], CFG, lambda s, m: LOS, lambda s: {}, with_keys=True)
    assert len(out) == 4 and len(keyed) == 5
    assert len(keyed[4]) == len(out[2]) and all(k[2] == g for k, g in zip(keyed[4], out[2]))
