import pytest

from dronepose.pipeline.streaming_eval import seed_summary


def _res(rot_off, rot_st, ade_off, ade_st):
    return {"streaming_row": "W60", "window_selection": {"chosen": 60},
            "summary": {"offline": {"median_rot_err_deg": rot_off, "flip_rate_no08": 0.002, "coverage": 0.97},
                        "W60": {"median_rot_err_deg": rot_st, "flip_rate_no08": 0.03, "coverage": 0.97}},
            "rot_vs_offline": {"W60": [rot_st - rot_off, -0.01, 0.05]},
            "flip_no08_vs_offline": {"W60": [0.028, 0.0, 0.08]},
            "forecast": {"offline": {"R-CV-6D": {"ade_1s": ade_off}}, "W60": {"R-CV-6D": {"ade_1s": ade_st}}},
            "forecast_streaming_minus_offline": {"R-CV-6D": {"mean": ade_st - ade_off, "ci": [0.001, 0.01]}}}


def test_seed_summary_mean_and_sd():
    out = seed_summary({0: _res(1.0, 1.1, 0.13, 0.14), 1: _res(2.0, 2.3, 0.15, 0.16)})
    assert out["per_seed"][1]["rot_stream"] == pytest.approx(2.3)
    assert out["mean"]["rot_delta"] == pytest.approx(0.2)
    assert out["sd"]["rot_delta"] == pytest.approx(0.1414, abs=1e-3)
    assert out["mean"]["ade_delta"] == pytest.approx(0.01)
    assert out["n_sig"]["ade_delta"] == 2
