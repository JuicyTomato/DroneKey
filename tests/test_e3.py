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


def _load_eval_e3():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("eval_e3", Path("scripts/eval_e3.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_e3 = _load_eval_e3()


def test_eval_e3_rejects_test_without_flag():
    records = [{"frame_id": "06_01_a_1", "drone": "01", "scene": "06", "split": "test", "frame_idx": 1}]
    with pytest.raises(ValueError, match="test split"):
        eval_e3.run([("s0", "missing.jsonl")], "missing_dets.jsonl", split="test", allow_test=False, records=records)


def test_eval_e3_compare_needs_chosen_and_known_names():
    with pytest.raises(ValueError, match="--chosen"):
        eval_e3.run([("s0", "k.jsonl")], "d.jsonl", compare_dronekeypp=True, records=[])
    with pytest.raises(ValueError, match="unknown"):
        eval_e3.select_combos(["ippe+nonsense"])
    assert {c.name for c in eval_e3.select_combos(["epnp+reproj"])} == {"epnp+reproj", "ippe+global"}
    assert len(eval_e3.select_combos(None)) == 64


def test_parse_seed():
    assert eval_e3.parse_seed("s0=a/b.jsonl") == ("s0", "a/b.jsonl")
    with pytest.raises(ValueError):
        eval_e3.parse_seed("s0")


def _fake(cov, rot, flip=0.0, groups=None):
    """Per-seed evaluation as `Context.evaluate` returns it: table rows all/no08 and rot groups by trajectory."""
    return {"table": {"all": {"coverage": cov, "median_rot_sym": rot}, "no08": {"flip_rate": flip}},
            "rot": groups}


def test_select_on_val_wiring():
    rng = np.random.default_rng(0)
    base = {f"t{i}": list(rng.normal(2.0, 0.3, 40)) for i in range(6)}
    far = {t: [v + 3.0 for v in vs] for t, vs in base.items()}
    seeds = ["s0", "s1", "s2"]

    def seed_data(better_rot):
        return {
            "ippe+global": _fake(0.97, 2.0, groups=base),
            "ippe+reproj": _fake(0.97, 2.0 - better_rot, groups={t: [v - better_rot for v in vs] for t, vs in base.items()}),
            "epnp+reproj": _fake(0.97, 5.0, groups=far),
            "sqpnp+global": _fake(0.80, 0.1, groups=base),          # best median but coverage too low
            "iterative+global": _fake(0.97, None, groups={}),       # no pose: excluded, no crash
        }
    # clearly separated: the best (ippe+reproj, 1.0 deg better in every seed) wins; ippe+global is not tied with it
    sel = eval_e3.select_on_val(seeds, {s: seed_data(1.0) for s in seeds})
    assert sel["best"] == "ippe+reproj" and sel["chosen"] == "ippe+reproj"
    assert set(sel["boots_vs_best"]["s0"]) == {"ippe+global", "ippe+reproj", "epnp+reproj"}
    assert sel["means"]["ippe+global"]["coverage"] == pytest.approx(0.97)
    # identical distributions: tie in every seed, the simplest (ippe+reproj < ippe+global) is kept, and no
    # tie is declared for the clearly worse epnp+reproj
    sel = eval_e3.select_on_val(seeds, {s: seed_data(0.0) for s in seeds})
    assert sel["chosen"] == "ippe+reproj"
    # tie broken by flip rate: the simpler combination has 2 points more flips than the best
    data = {s: seed_data(0.0) for s in seeds}
    for s in seeds:
        data[s]["ippe+reproj"] = _fake(0.97, 2.0, flip=0.02, groups=base)
        data[s]["ippe+global"] = _fake(0.97, 1.99, flip=0.0, groups=base)
    assert eval_e3.select_on_val(seeds, data)["chosen"] == "ippe+global"


def test_select_on_val_no_coverage_is_an_error_not_a_crash():
    groups = {"t0": [1.0, 2.0]}
    sel = eval_e3.select_on_val(["s0"], {"s0": {"ippe+global": _fake(0.92, 1.0, groups=groups)}})
    assert "error" in sel and "0.95" in sel["error"]
