import pytest

from dronepose.eval.class_select import GRID, Cell, choose_cell, latency_ok_from, loso_maps, param_grid, \
    pose_ok_from, tune_all, tune_loso


def _cells(*names):
    by = {c.name: c for c in GRID}
    return {n: by[n] for n in names}


def test_grid_names_and_costs():
    names = {c.name for c in GRID}
    assert {"C1+V", "C1+F", "C1+W", "C1+L", "C1+L+g", "K+L", "K+CM+L+g", "G+F", "G+L"} <= names
    assert "C1+V+g" not in names and "G+W" not in names
    by = {c.name: c for c in GRID}
    assert by["C1+L+g"].cost == 0 and by["K+L"].cost == 1 and by["K+CM+W"].cost == 1
    assert by["C1+V"].n_params == 0 and by["C1+L+g"].n_params > by["C1+L"].n_params


def test_param_grid_values():
    assert param_grid(Cell("C1", "V", False)) == [{}]
    g = param_grid(Cell("C1", "L", True))
    assert {"a_min", "w_g", "form"} <= set(g[0])
    assert len({p["w_g"] for p in g}) == 5 and {p["form"] for p in g} == {"log", "quad"}
    assert all("sigma" in p for p in g if p["form"] == "quad") and not any("sigma" in p for p in g if p["form"] == "log")


def _boots(table):
    return lambda ref: table[ref]


def test_choose_cell_prefers_cost0_in_tie():
    cells = _cells("K+L", "C1+L+g", "C1+V")
    acc = {"K+L": [0.95] * 3, "C1+L+g": [0.93] * 3, "C1+V": [0.85] * 3}
    table = {"K+L": {"K+L": [(0, 0, 0)] * 3, "C1+L+g": [(-0.02, -0.04, 0.001)] * 3,
                     "C1+V": [(-0.1, -0.15, -0.05)] * 3}}
    assert choose_cell(acc, _boots(table), cells, {n: True for n in cells}) == "C1+L+g"


def test_choose_cell_k_wins_when_significant():
    cells = _cells("K+L", "C1+L+g")
    acc = {"K+L": [0.95] * 3, "C1+L+g": [0.90] * 3}
    table = {"K+L": {"K+L": [(0, 0, 0)] * 3, "C1+L+g": [(-0.05, -0.08, -0.02)] * 3}}
    assert choose_cell(acc, _boots(table), cells, {n: True for n in cells}) == "K+L"


def test_choose_cell_tie_needs_zero_in_only_one_seed():
    cells = _cells("K+L", "C1+L+g")
    acc = {"K+L": [0.95] * 3, "C1+L+g": [0.90] * 3}
    table = {"K+L": {"K+L": [(0, 0, 0)] * 3,
                     "C1+L+g": [(-0.05, -0.08, -0.02), (-0.05, -0.08, 0.01), (-0.05, -0.08, -0.02)]}}
    assert choose_cell(acc, _boots(table), cells, {n: True for n in cells}) == "C1+L+g"


def test_choose_cell_respects_pose_constraint():
    cells = _cells("K+L", "C1+L+g")
    acc = {"K+L": [0.95] * 3, "C1+L+g": [0.90] * 3}
    table = {"C1+L+g": {"C1+L+g": [(0, 0, 0)] * 3, "K+L": [(0.05, 0.02, 0.08)] * 3}}
    assert choose_cell(acc, _boots(table), cells, {"K+L": False, "C1+L+g": True}) == "C1+L+g"


def test_choose_cell_residual_tie_prefers_no_geometry_then_fewer_params():
    cells = _cells("C1+L", "C1+L+g", "C1+W")
    acc = {"C1+L": [0.9] * 3, "C1+L+g": [0.9] * 3, "C1+W": [0.9] * 3}
    tie = [(0.0, -0.01, 0.01)] * 3
    table = {n: {m: tie for m in cells} for n in cells}
    assert choose_cell(acc, _boots(table), cells, {n: True for n in cells}) == "C1+W"


def _recs(scene, n=4):
    return [{"frame_id": f"{scene}_01_05_180_{i:04d}", "frame_idx": i, "drone": "01"} for i in range(1, n + 1)]


def test_tune_loso_never_uses_held_out_scene():
    recs = _recs("03") + _recs("09") + _recs("14")
    grid = [{"p": "a"}, {"p": "b"}]

    def build(params):
        # p=a is right only on scene 03, p=b only on 09 and 14
        good = {"a": {"03"}, "b": {"09", "14"}}[params["p"]]
        return {0: {r["frame_id"]: "01" if r["frame_id"][:2] in good else "02" for r in recs}}

    chosen = tune_loso(build, recs, grid)
    # fold 03 picks b although a is the right one on 03; folds 09 and 14 tie (one scene each) -> first, a
    assert chosen == {"03": {"p": "b"}, "09": {"p": "a"}, "14": {"p": "a"}}


def test_tune_loso_ties_go_to_first_params():
    recs = _recs("03") + _recs("09") + _recs("14")
    build = lambda params: {0: {r["frame_id"]: "01" for r in recs}}  # noqa: E731
    assert tune_loso(build, recs, [{"p": 1}, {"p": 2}])["09"] == {"p": 1}


def test_tune_loso_rejects_empty_grid():
    with pytest.raises(ValueError):
        tune_loso(lambda p: {}, _recs("03"), [])


def test_tune_all_uses_every_scene():
    recs = _recs("03") + _recs("09") + _recs("14")
    good = {"a": {"03"}, "b": {"09", "14"}}
    build = lambda params: {0: {r["frame_id"]: "01" if r["frame_id"][:2] in good[params["p"]] else "02"  # noqa: E731
                                for r in recs}}
    assert tune_all(build, recs, [{"p": "a"}, {"p": "b"}]) == {"p": "b"}


def test_loso_maps_takes_each_scene_from_its_fold():
    recs = _recs("03", 1) + _recs("09", 1)
    build = lambda params: {0: {r["frame_id"]: params["c"] for r in recs}}  # noqa: E731
    out = loso_maps(build, recs, {"03": {"c": "05"}, "09": {"c": "06"}})
    assert out == {0: {recs[0]["frame_id"]: "05", recs[1]["frame_id"]: "06"}}


def test_pose_ok_from():
    worse = [(0.2, 0.05, 0.4)] * 3
    mixed = [(0.2, 0.05, 0.4), (0.1, -0.02, 0.3), (0.2, 0.05, 0.4)]
    assert not pose_ok_from(worse, flip=0.002, flip_base=0.002)
    assert pose_ok_from(mixed, flip=0.002, flip_base=0.002)
    assert not pose_ok_from(mixed, flip=0.008, flip_base=0.002)   # +0.6 points
    assert pose_ok_from([(-0.1, -0.2, -0.01)] * 3, flip=0.007, flip_base=0.002)   # +0.5 points: allowed


def test_latency_ok_from():
    cells = _cells("C1+W+g", "K+L")
    assert latency_ok_from(cells, None) == {"C1+W+g": True, "K+L": False}       # not measured: K not allowed
    assert latency_ok_from(cells, 29.9) == {"C1+W+g": True, "K+L": False}
    assert latency_ok_from(cells, 30.0) == {"C1+W+g": True, "K+L": True}
