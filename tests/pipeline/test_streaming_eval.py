import numpy as np
import pytest

from dronepose.pipeline.streaming_eval import pose_summary, select_obs_mode, select_window


def _res(errs, drone="02", traj="14_02"):
    return [{"frame_id": f"{traj}_07_000_{i:04d}", "rot_err_deg": e, "rot_err_sym_deg": e, "t_err": 0.0,
             "ade": None, "rejected": e is None, "drone": drone} for i, e in enumerate(errs)]


def test_pose_summary_counts_flips():
    s = pose_summary(_res([1, 2, 3, 179, 170, 1, 2, 3, 4, 5]))
    assert s["flip_rate"] == pytest.approx(0.2)
    assert s["coverage"] == pytest.approx(1.0)
    s = pose_summary(_res([1, None, 179]) + _res([170, 1], drone="08", traj="14_08"))
    assert s["flip_rate_no08"] == pytest.approx(0.5)


def _groups(rng, shift):
    return {f"{s}_{d}": list(rng.normal(1 + shift, 0.05, 50)) for s in ("03", "09", "14") for d in ("01", "02", "03")}


def test_select_window_rule():
    rng = np.random.default_rng(0)
    summ = {15: {"coverage": 0.90, "median_rot_err_deg": 0.5}, 30: {"coverage": 0.97, "median_rot_err_deg": 1.0},
            60: {"coverage": 0.97, "median_rot_err_deg": 1.001}, 120: {"coverage": 0.98, "median_rot_err_deg": 2.0}}
    rot = {15: _groups(rng, -0.5), 30: _groups(rng, 0.0), 60: _groups(rng, 0.001), 120: _groups(rng, 1.0)}
    for k in (30, 60):
        summ[k]["median_rot_err_deg"] = float(np.median(np.concatenate(list(rot[k].values()))))
    out = select_window(summ, rot, n_boot=500)
    assert 15 not in out["eligible"]
    assert out["chosen"] == 30          # 30 and 60 tie, smaller wins; 120 clearly worse
    rot[60] = _groups(rng, -0.3)
    summ[60]["median_rot_err_deg"] = float(np.median(np.concatenate(list(rot[60].values()))))
    assert select_window(summ, rot, n_boot=500)["chosen"] == 60


def test_select_obs_mode_tie_goes_to_e(monkeypatch):
    import dronepose.pipeline.streaming_eval as se
    monkeypatch.setattr(se, "degradation", lambda a, b, w, n: {"R-CV-6D": {"mean": -0.01, "ci": [-0.02, 0.001]}})
    assert select_obs_mode({}, {}, None, 10)["chosen"] == "e"
    monkeypatch.setattr(se, "degradation", lambda a, b, w, n: {"R-CV-6D": {"mean": -0.01, "ci": [-0.02, -0.001]}})
    assert select_obs_mode({}, {}, None, 10)["chosen"] == "r"


def _flip_groups(rng, p):
    return {f"{s}_{d}": [float(x) for x in (rng.random(60) < p)] for s in ("03", "09", "14") for d in ("01", "02", "03")}


def test_select_window_flips_rule():
    from dronepose.pipeline.streaming_eval import select_window_flips
    rng = np.random.default_rng(0)
    flips = {15: _flip_groups(rng, 0.02), 30: _flip_groups(rng, 0.02), 60: _flip_groups(rng, 0.02),
             120: _flip_groups(rng, 0.30)}
    rot = {15: _groups(rng, 0.0), 30: _groups(rng, 0.2), 60: _groups(rng, 0.0), 120: _groups(rng, 0.0)}
    summ = {W: {"coverage": 0.97, "median_rot_err_deg": float(np.median(np.concatenate(list(rot[W].values())))),
                "flip_rate_no08": float(np.mean(np.concatenate(list(flips[W].values()))))} for W in flips}
    summ[15]["coverage"] = 0.90                                  # not eligible
    summ[120]["median_rot_err_deg"] = 0.0                        # lowest rotation but clearly worse on flips
    out = select_window_flips(summ, flips, rot, n_boot=500)
    assert out["eligible"] == [30, 60, 120]
    assert 120 not in out["tied"] and 30 in out["tied"] and 60 in out["tied"]
    # among 30 and 60 (tied on flips) the lower median rotation wins; 60 has lower rotation than 30
    assert out["chosen"] == 60
    summ[30]["median_rot_err_deg"] = summ[60]["median_rot_err_deg"]
    assert select_window_flips(summ, flips, rot, n_boot=500)["chosen"] == 30      # tie on rotation: smaller W
    assert set(out["ci_vs_best"]) == {30, 60, 120}
