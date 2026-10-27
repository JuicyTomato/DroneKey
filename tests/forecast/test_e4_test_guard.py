import subprocess
import sys
from pathlib import Path

import pytest

from dronepose.forecast.e4_eval import guard_test_run

SCRIPT = Path("scripts/forecast/e4/e4_test.py")


def test_refuses_without_flag(tmp_path):
    with pytest.raises(ValueError):
        guard_test_run(False, tmp_path / "test")


def test_refuses_when_already_run(tmp_path):
    out = tmp_path / "test"
    out.mkdir()
    with pytest.raises(ValueError):
        guard_test_run(True, out)


def test_allows_once(tmp_path):
    guard_test_run(True, tmp_path / "test")


def test_script_refuses_without_flag(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path / "test")], capture_output=True, text=True)
    assert r.returncode != 0 and "allow-test" in r.stderr
    assert not (tmp_path / "test").exists()


def test_script_refuses_existing_out(tmp_path):
    (tmp_path / "test").mkdir()
    r = subprocess.run([sys.executable, str(SCRIPT), "--allow-test", "--out", str(tmp_path / "test")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "una volta sola" in r.stderr


def test_smoke_refuses_real_test_dir():
    with pytest.raises(ValueError):
        guard_test_run(False, Path("experiments/forecast/e4/test"), smoke=True)


def test_smoke_allowed_elsewhere(tmp_path):
    guard_test_run(False, tmp_path / "smoke", smoke=True)


def test_script_smoke_without_out_refused():
    r = subprocess.run([sys.executable, str(SCRIPT), "--smoke-val"], capture_output=True, text=True)
    assert r.returncode != 0 and "smoke" in r.stderr


def test_t3_pairing_and_validation():
    from dronepose.forecast.e4_eval import t3_e4a_key, validate_choices
    assert t3_e4a_key("kf_ca_los") == "kf_ca" and t3_e4a_key("R-CV-6D") == "R-CV-6D"
    e4a = ("cv", "ca", "kf_cv", "kf_ca", "physics", "G-6D", "R-CV-6D", "R-CV-FT")
    validate_choices({"E4a": "R-CV-FT", "E4b": "kf_ca_los"}, e4a, e4a + ("kf_ca_los",))
    with pytest.raises(ValueError):
        validate_choices({"E4a": "kf_ca_los", "E4b": "cv"}, e4a, e4a + ("kf_ca_los",))
    with pytest.raises(ValueError):
        validate_choices({"E4a": "cv", "E4b": "mystery"}, e4a, e4a + ("kf_ca_los",))
