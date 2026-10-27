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
