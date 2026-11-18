import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts/real")
_spec = importlib.util.spec_from_file_location("final_test", Path("scripts/real/final_test.py"))
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)
FROZEN = {"ref": "yolo_s0", "support": [], "detector": "d.pt", "det_conf": 0.1, "yolo": {"new_s0": "y.pt"}, "dk": {"dk_new_s0": "k.pt"}, "refs_yolo": {"yolo_s0": "ry.pt"},
          "refs_dk": {"dk_s0": "rk.pt"}, "refs_a_kpts": {"yolo_s0": "a.jsonl", "dk_s0": "b.jsonl"}}


def test_frozen_ref_must_be_a_yolo_reference():
    with pytest.raises(ValueError, match="ref"):
        ft.check_frozen({**FROZEN, "ref": "nope"})


def test_every_reference_needs_its_setting_a_export():
    with pytest.raises(ValueError, match="dk_s0"):
        ft.check_frozen({**FROZEN, "refs_a_kpts": {"yolo_s0": "a.jsonl"}})


def test_frozen_lists_missing_files(tmp_path):
    (tmp_path / "d.pt").write_bytes(b"x")
    frozen = {**FROZEN, "detector": str(tmp_path / "d.pt")}
    miss = ft.missing_files(frozen)
    assert "y.pt" in miss and "a.jsonl" in miss and str(tmp_path / "d.pt") not in miss


def test_support_models_stay_out_of_the_group(tmp_path):
    frozen = {**FROZEN, "support": ["final"], "refs_yolo": {"yolo_s0": "ry.pt", "final": "f.pt"},
              "refs_a_kpts": {**FROZEN["refs_a_kpts"], "final": "c.jsonl"}}
    b = " ".join(ft.build_commands(frozen, tmp_path, tmp_path / "split")[-1])
    assert "yolo_e2=yolo_s0 " in b and "final=" in b


def test_needs_allow_test(tmp_path):
    with pytest.raises(ValueError, match="--allow-test"):
        ft.guard(False, tmp_path / "o", smoke=False)


def test_refuses_existing_out(tmp_path):
    with pytest.raises(ValueError, match="una volta sola"):
        ft.guard(True, tmp_path, smoke=False)


def test_smoke_refuses_real_dir():
    with pytest.raises(ValueError, match="smoke"):
        ft.guard(False, ft.TEST_OUT, smoke=True)


def test_smoke_and_allowed_pass(tmp_path):
    ft.guard(False, tmp_path / "smoke", smoke=True)
    ft.guard(True, tmp_path / "test", smoke=False)


def test_commands_cover_settings_and_references(tmp_path):
    cmds = [" ".join(c) for c in ft.build_commands(FROZEN, tmp_path, tmp_path / "split")]
    assert sum("export_frames_c.py" in c for c in cmds) == 1
    assert sum("export_two_stage.py" in c for c in cmds) == 3   # new A, new B, ref B
    assert sum("export_dronekey.py" in c for c in cmds) == 3
    assert sum("eval_e2.py" in c for c in cmds) == 2
    assert not any("--allow-test" in c for c in cmds)


def test_detector_runs_first_and_evals_last(tmp_path):
    cmds = [" ".join(c) for c in ft.build_commands(FROZEN, tmp_path, tmp_path / "split")]
    assert "export_frames_c.py" in cmds[0] and "eval_e2.py" in cmds[-1] and "eval_e2.py" in cmds[-2]


def test_only_setting_b_gets_predicted_class(tmp_path):
    cmds = [" ".join(c) for c in ft.build_commands(FROZEN, tmp_path, tmp_path / "split")]
    a, b = cmds[-2], cmds[-1]
    assert "e2_a.json" in a and "--class-from" not in a
    assert "e2_b.json" in b and "--class-from" in b


def test_interpreters(tmp_path):
    cmds = ft.build_commands(FROZEN, tmp_path, tmp_path / "split")
    for c in cmds:
        expected = ft.YOLO_PY if any("export_frames_c" in x or "export_two_stage" in x for x in c) else ft.PY
        assert c[0] == expected


def test_evals_use_the_split_records(tmp_path):
    cmds = ft.build_commands(FROZEN, tmp_path, tmp_path / "split")
    for c in cmds[-2:]:
        assert c[c.index("--canonical") + 1] == str(tmp_path / "split" / "gt.jsonl")


def test_detector_uses_frozen_threshold(tmp_path):
    det = ft.build_commands({**FROZEN, "det_conf": 0.1}, tmp_path, tmp_path / "split")[0]
    assert det[det.index("--conf") + 1] == "0.1"


def test_frozen_needs_detector_threshold():
    with pytest.raises(ValueError, match="det_conf"):
        ft.check_frozen({k: v for k, v in FROZEN.items() if k != "det_conf"})
