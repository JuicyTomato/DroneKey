import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.eval.e3 import (Combo, all_combos, choose_combo, euler_axis_errors, mae_medae, reference,
                               simplicity_key, solver_timing)


class _Model:
    kpts_flu = np.array([[0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [-0.1, -0.1, 0.0], [-0.1, 0.1, 0.0]])


def test_all_combos_count_and_unique():
    cs = all_combos()
    assert len(cs) == 64
    assert len({c.key for c in cs}) == 64
    assert sum(c.cls == "gt" for c in cs) == 32


def test_names():
    assert Combo("sqpnp", True, "reproj", "causal").name == "sqpnp+lm+reproj"
    assert Combo("ippe", False, "global", "gt").key == "ippe+global|gt"


def test_reference():
    assert reference("causal").name == "ippe+global"
    assert reference("gt").cls == "gt"


def test_simplicity_order():
    cs = {c.name: c for c in all_combos()}
    names = ["epnp+reproj", "ippe+lm+reproj", "ippe+global", "ippe+reproj"]
    # global constraints: refine False < True first, then strategy, then solver
    assert sorted(names, key=lambda n: simplicity_key(cs[n])) == [
        "ippe+reproj", "epnp+reproj", "ippe+global", "ippe+lm+reproj"]


def _means(**kw):
    return {n: {"coverage": v[1], "median_rot_sym": v[0], "flip_no08": v[2] if len(v) > 2 else 0.0}
            for n, v in kw.items()}


def test_choose_best_when_separated():
    means = _means(**{"ippe+global": (1.50, 1.0), "ippe+reproj": (2.0, 1.0), "epnp+global": (1.2, 1.0)})
    boots = {"ippe+global": [(0.3, 0.2, 0.4)] * 3, "ippe+reproj": [(0.8, 0.5, 1.1)] * 3}
    assert choose_combo(means, boots) == "epnp+global"


def test_choose_simplest_on_tie():
    means = _means(**{"ippe+global": (1.50, 1.0), "ippe+reproj": (1.52, 1.0)})
    boots = {"ippe+reproj": [(0.02, -0.01, 0.05)] * 3}
    assert choose_combo(means, boots) == "ippe+reproj"


def test_choose_tie_rejected_when_more_flips():
    means = _means(**{"ippe+global": (1.50, 1.0, 0.0), "ippe+reproj": (1.52, 1.0, 0.012)})
    boots = {"ippe+reproj": [(0.02, -0.01, 0.05)] * 3}
    assert choose_combo(means, boots) == "ippe+global"


def test_choose_tie_accepted_within_flip_margin():
    means = _means(**{"ippe+global": (1.50, 1.0, 0.0), "ippe+reproj": (1.52, 1.0, 0.004)})
    boots = {"ippe+reproj": [(0.02, -0.01, 0.05)] * 3}
    assert choose_combo(means, boots) == "ippe+reproj"


def test_choose_tie_needs_all_seeds():
    means = _means(**{"ippe+global": (1.50, 1.0), "ippe+reproj": (1.52, 1.0)})
    boots = {"ippe+reproj": [(0.02, -0.01, 0.05), (0.02, -0.01, 0.05), (0.1, 0.05, 0.2)]}
    assert choose_combo(means, boots) == "ippe+global"


def test_choose_excludes_low_coverage():
    means = _means(**{"ippe+global": (1.50, 0.97), "epnp+global": (1.0, 0.93)})
    assert choose_combo(means, {}) == "ippe+global"
    with pytest.raises(ValueError):
        choose_combo(_means(**{"ippe+global": (1.0, 0.5)}), {})


def test_euler_wraps():
    Ra = Rotation.from_euler("ZYX", [179, 0, 0], degrees=True).as_matrix()
    Rb = Rotation.from_euler("ZYX", [-179, 0, 0], degrees=True).as_matrix()
    e = euler_axis_errors(Ra, Rb)
    assert e[0] == pytest.approx(2.0, abs=1e-6)
    assert e[1] == pytest.approx(0.0, abs=1e-6) and e[2] == pytest.approx(0.0, abs=1e-6)


def test_mae_medae():
    assert mae_medae([1, 2, 9]) == (4.0, 2.0)


def test_solver_timing():
    from dronepose.common.geometry import project
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    R = Rotation.from_euler("ZYX", [10, 20, 5], degrees=True).as_matrix()
    pts = project(K, (R @ _Model.kpts_flu.T).T + np.array([0.0, 0.0, 2.0]))
    recs = [{"frame_id": f"f{i}", "drone": "01", "K": K} for i in range(4)]
    kp = {"f0": np.c_[pts, np.ones(4)], "f1": np.c_[pts, np.ones(4)], "f2": np.full((4, 3), np.nan)}
    out = solver_timing(recs, kp, {"01": _Model(), "02": _Model()}, "ippe", False, model_drone={f"f{i}": "02" for i in range(4)},
                        repeats=2)
    assert out["n"] == 4  # f0, f1 x 2 repeats; f2 non-finite and f3 missing are skipped
    assert 0 < out["median_ms"] <= out["p90_ms"]
