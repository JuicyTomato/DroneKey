import csv
import dataclasses
import json

import numpy as np
from scipy.spatial.transform import Rotation

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import Trajectory
from dronepose.forecast.report import summarize, write_report
from dronepose.forecast.tuning import POSITION_METHODS, predict_all
from dronepose.forecast.windows import make_windows

CFG = load_config()
PARAMS = {"cv_k": 2, "ca_k": 3, "kf_cv": {"q": 100.0, "r": 1e-6}, "kf_ca": {"q": 1000.0, "r": 1e-6}, "ctr_k": 3,
          "acro_k": 3, "acro_kd": "estimate", "acro_gen": {"lam": 5.0, "thrust": "hold_vz", "k": 3, "kd": "estimate"}}
B = 200


def _traj(scene, drone, seed, s_c=1.0, scale=1.0):
    t = np.arange(70) / 30
    rng = np.random.default_rng(seed)
    p = scale * np.stack([np.sin(2 * t + seed), np.cos(3 * t), 0.2 * t], axis=1) + 0.01 * rng.normal(size=(70, 3))
    R = Rotation.from_euler("xz", np.stack([5 * np.sin(t), 20 * t + seed], axis=1), degrees=True).as_matrix()
    motion = "linear" if scene == "03" else "nonlinear"
    return Trajectory(scene=scene, drone=drone, split="val", motion_type=motion, copies=[("01", 0)], s_c=s_c,
                      times=t, p_G=p, R_G=R)


BASE = [_traj("03", "01", 1), _traj("03", "02", 2), _traj("14", "01", 3), _traj("14", "02", 4)]


def _run(trajs):
    w = make_windows(trajs, CFG.obs_len, CFG.pred_len, CFG.fps)
    preds = predict_all(w, PARAMS, CFG)
    return w, preds, summarize(w, preds, CFG, bootstrap_b=B)


def _group(summary, name):
    return next(g for g in summary["groups"] if g["group"] == name)


def test_summary_keeps_08_out_of_metric_groups():
    _, _, a = _run(BASE)
    _, _, b = _run(BASE + [_traj("14", "08", 9, s_c=None, scale=100.0)])
    for name in ("metri: tutti", "metri: lineare", "metri: non lineare", "metri: scena 14"):
        assert _group(a, name)["position"] == _group(b, name)["position"]
    g08 = _group(b, "drone 08 (unità dataset)")
    assert g08["units"] == "dataset" and g08["n_traj"] == 1 and any("nominale" in n for n in g08["notes"])


def test_cv_skill_against_itself_is_zero():
    _, _, s = _run(BASE)
    np.testing.assert_allclose(_group(s, "metri: tutti")["skill"]["cv"]["ade_1s"], [0, 0, 0], atol=1e-12)


def test_per_drone_rows_have_no_ci():
    _, _, s = _run(BASE)
    assert len(_group(s, "drone 01")["skill"]["ca"]["ade_1s"]) == 1
    assert len(_group(s, "metri: tutti")["skill"]["ca"]["ade_1s"]) == 3
    assert len(_group(s, "metri: scena 03")["skill"]["ca"]["fde_1s"]) == 3


def test_summary_values_match_direct_computation():
    w, preds, s = _run(BASE)
    err = np.linalg.norm(preds["const"] - w.fut_p, axis=-1)
    assert np.isclose(_group(s, "metri: tutti")["position"]["const"]["fde_mean"][-1], err[:, 29].mean())
    assert np.isclose(_group(s, "metri: tutti")["position"]["const"]["ade_median"][1], np.median(err[:, :10].mean(1)))


def test_write_report_files(tmp_path):
    w, preds, s = _run(BASE)
    write_report(s, w, preds, tmp_path)
    for name in ("table.json", "table.md", "per_window.csv"):
        assert (tmp_path / name).is_file()
    assert json.loads((tmp_path / "table.json").read_text())["n_windows"] == len(w)
    with open(tmp_path / "per_window.csv") as f:
        assert sum(1 for _ in csv.DictReader(f)) == len(w) * len(POSITION_METHODS)


def test_summary_has_fde_curve_and_figure(tmp_path):
    w, preds, s = _run(BASE)
    g = _group(s, "metri: tutti")
    err = np.linalg.norm(preds["cv"] - w.fut_p, axis=-1)
    np.testing.assert_allclose(g["fde_curve"]["cv"], err[w.metric].mean(0))
    write_report(s, w, preds, tmp_path)
    assert (tmp_path / "ade_fde_vs_horizon.png").is_file()
