import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "test_class_script", Path(__file__).resolve().parents[1] / "scripts" / "class" / "test_class.py")
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)


def test_test_needs_allow_test(tmp_path):
    with pytest.raises(ValueError, match="--allow-test"):
        tc.guard(False, tmp_path / "out", smoke=False)


def test_test_refuses_existing_out(tmp_path):
    with pytest.raises(ValueError, match="una volta sola"):
        tc.guard(True, tmp_path, smoke=False)


def test_smoke_refuses_real_test_dir():
    with pytest.raises(ValueError, match="smoke"):
        tc.guard(False, tc.TEST_OUT, smoke=True)


def test_smoke_and_allowed_test_pass(tmp_path):
    tc.guard(False, tmp_path / "smoke", smoke=True)
    tc.guard(True, tmp_path / "test", smoke=False)


@pytest.mark.skipif(not tc.PARAMS.exists(), reason="frozen parameters are outside git")
def test_frozen_cells_are_the_kept_variants():
    cells = tc.frozen_cells()
    assert list(cells) == ["C1+V", "C1+W+g", "K+L+g"]
    assert cells["C1+W+g"][1] == {"w_g": 0.1, "form": "quad", "sigma": 1.0}
    assert cells["K+L+g"][1] == {"a_min": 0.1, "w_g": 1.0, "form": "log"}
