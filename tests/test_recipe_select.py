import importlib.util
import json
from pathlib import Path

import pytest

from dronepose.eval.recipe_select import best_of, decide, paired_groups, wins

_spec = importlib.util.spec_from_file_location("split_authors", Path("scripts/real/split_authors.py"))
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)
_spec = importlib.util.spec_from_file_location("select_recipe", Path("scripts/real/select_recipe.py"))
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)


def test_paired_groups_drop_frames_missing_in_one():
    g = paired_groups({"a_1_1": 5.0, "a_1_2": None, "b_1_1": 7.0}, {"a_1_1": 3.0, "a_1_2": 2.0, "b_1_1": float("nan")})
    assert g == {"ref": {"a_1": [5.0]}, "cand": {"a_1": [3.0]}}


@pytest.mark.parametrize("boot,cov,flip,ok", [
    ((-5, -8, -2), 0.50, 0.30, True),
    ((-5, -8, 0.1), 0.50, 0.30, False),
    ((-5, -8, -2), 0.449, 0.30, False),
    ((-5, -8, -2), 0.50, 0.331, False),
    ((-5, -8, -2), 0.45, 0.33, True),
])
def test_wins(boot, cov, flip, ok):
    assert wins(boot, 0.50, cov, 0.30, flip) is ok


def test_best_of():
    assert best_of(["a", "b"], {"a": 3.0, "b": 2.0}) == "b"


def test_decide_no_winner_stops():
    assert decide({"winner": None}, {"winner": None}, None) == {"recipe": None, "stop": True, "combo_needed": False}


def test_decide_one_winner_skips_combo():
    assert decide({"winner": "cp90"}, {"winner": None}, None) == {"recipe": "cp90", "stop": False, "combo_needed": False}
    assert decide({"winner": None}, {"winner": "bl10"}, None)["recipe"] == "bl10"


def test_decide_two_winners_needs_combo_then_uses_it():
    assert decide({"winner": "cp90"}, {"winner": "bl20"}, None)["combo_needed"] is True
    a, b = {"winner": "cp90", "median": 20.0}, {"winner": "bl20", "median": 25.0}
    assert decide(a, b, {"name": "c90b20", "wins": False})["recipe"] == "cp90"
    assert decide(a, b, {"name": "c90b20", "wins": True})["recipe"] == "c90b20"
    assert decide(b, a, {"name": "c90b20", "wins": False})["recipe"] == "cp90"


def _write_gt(tmp_path):
    gt, box = tmp_path / "gt.jsonl", tmp_path / "box.jsonl"
    rows = [{"frame_id": "a_1_1", "authors_split": "train"}, {"frame_id": "b_1_1", "authors_split": "test"}]
    gt.write_text("".join(json.dumps(r) + "\n" for r in rows))
    box.write_text("".join(json.dumps({"frame_id": r["frame_id"], "box": [0, 0, 1, 1]}) + "\n" for r in rows))
    return gt, box


def test_authors_test_needs_flag(tmp_path):
    with pytest.raises(ValueError, match="--allow-test"):
        sa.write_split("test", tmp_path / "o")


def test_write_split_keeps_only_that_split(tmp_path):
    gt, box = _write_gt(tmp_path)
    sa.write_split("train", tmp_path / "o", gt=gt, gtbox=box)
    assert [json.loads(l)["frame_id"] for l in open(tmp_path / "o" / "gt.jsonl")] == ["a_1_1"]
    assert [json.loads(l)["frame_id"] for l in open(tmp_path / "o" / "gtbox.jsonl")] == ["a_1_1"]


def test_write_split_limit(tmp_path):
    gt, box = _write_gt(tmp_path)
    sa.write_split("train", tmp_path / "o", gt=gt, gtbox=box, limit=0)
    assert (tmp_path / "o" / "gt.jsonl").read_text() == ""


def test_select_cli_end_to_end(tmp_path):
    frames = [f"s{i}_1_{j}" for i in range(6) for j in range(5)]
    dumps = tmp_path / "frames"
    dumps.mkdir()
    (dumps / "ref.json").write_text(json.dumps({f: 30.0 for f in frames}))
    (dumps / "good.json").write_text(json.dumps({f: 10.0 for f in frames}))
    (dumps / "same.json").write_text(json.dumps({f: 30.0 for f in frames}))
    entry = {"real": {"all": {"coverage": 0.5, "flip_rate": 0.3, "median_rot_sym": 30.0}}, "kpts2d": {"all": {"px_median": 50.0}}}
    good = {**entry, "real": {"all": {**entry["real"]["all"], "median_rot_sym": 10.0}}}
    (tmp_path / "e2.json").write_text(json.dumps({"models": {"ref": entry, "good": good, "same": entry}}))
    out = tmp_path / "s.json"
    sr.main(["--e2", str(tmp_path / "e2.json"), "--dumps", str(dumps), "--ref", "ref", "--cand", "good", "--cand", "same", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["winner"] == "good" and r["median"] == pytest.approx(10.0)
    assert r["candidates"]["same"]["wins"] is False
    d = tmp_path / "d.json"
    (tmp_path / "none.json").write_text(json.dumps({"winner": None}))
    sr.main(["--decide", str(out), str(tmp_path / "none.json"), "--out", str(d)])
    assert json.loads(d.read_text())["recipe"] == "good"


def _dumps(tmp_path, **models):
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    for name, vals in models.items():
        (d / f"{name}.json").write_text(json.dumps(vals))
    return d


def test_head_to_head_uses_only_shared_frames(tmp_path):
    # hard frames: a poses them badly, b drops them, so b has the lower median on its own frames
    hard, easy = [f"s_1_{i}" for i in range(6)], [f"s_1_{i}" for i in range(6, 10)]
    a = {**{f: 50.0 for f in hard}, **{f: 8.0 for f in easy}}
    b = {**{f: None for f in hard}, **{f: 10.0 for f in easy}}
    assert sr.head_to_head(["a", "b"], _dumps(tmp_path, a=a, b=b)) == "a"


def test_step_records_best_even_without_winners(tmp_path):
    frames = [f"s{i}_1_{j}" for i in range(6) for j in range(5)]
    dumps = _dumps(tmp_path, ref={f: 30.0 for f in frames}, x={f: 31.0 for f in frames}, y={f: 32.0 for f in frames})
    entry = {"real": {"all": {"coverage": 0.5, "flip_rate": 0.3, "median_rot_sym": 30.0}}, "kpts2d": {"all": {"px_median": 50.0}}}
    r = sr.compare({"models": {"ref": entry, "x": entry, "y": entry}}, dumps, "ref", ["x", "y"])
    assert r["winner"] is None and r["best"] == "x"


def test_compare_refuses_results_that_include_the_test_split(tmp_path):
    entry = {"real": {"all": {}, "authors_test": {}}, "kpts2d": {"all": {}}}
    with pytest.raises(ValueError, match="test"):
        sr.compare({"models": {"ref": entry}}, tmp_path, "ref", [])


def test_two_winners_report_the_paired_reference(tmp_path):
    frames = [f"s{i}_1_{j}" for i in range(6) for j in range(5)]
    dumps = _dumps(tmp_path, cp90={f: 10.0 for f in frames}, bl20={f: 9.0 for f in frames})
    s1, s2 = tmp_path / "1.json", tmp_path / "2.json"
    s1.write_text(json.dumps({"winner": "cp90", "median": 1.0}))
    s2.write_text(json.dumps({"winner": "bl20", "median": 2.0}))
    out = tmp_path / "d.json"
    sr.main(["--decide", str(s1), str(s2), "--dumps", str(dumps), "--out", str(out)])
    d = json.loads(out.read_text())
    assert d["combo_needed"] is True and d["reference"] == "bl20"
