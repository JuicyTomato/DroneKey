import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "sigma_stats", Path(__file__).resolve().parents[2] / "scripts" / "dronekey" / "sigma_stats.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _crops(root, split):
    (root / "labels" / split).mkdir(parents=True)
    for i in range(3):
        vals = " ".join(f"{x + 0.01 * i} {y} 2" for x, y in [(0.5, 0.2), (0.8, 0.5), (0.5, 0.8), (0.3, 0.5)])
        (root / "labels" / split / f"f_{i}.txt").write_text(f"0 0.5 0.5 0.9 0.9 {vals}\n")


def test_sigma_stats_refuses_test_split_without_allow_test(tmp_path):
    root = tmp_path / "crops"
    _crops(root, "test")
    args = ["--crops-root", str(root), "--split", "test", "--out-dir", str(tmp_path / "out")]
    with pytest.raises(ValueError, match="allow-test"):
        S.main(args)
    assert not (tmp_path / "out").exists()
    S.main([*args, "--allow-test"])
    assert json.loads((tmp_path / "out" / "sigma_stats_crops.json").read_text())["split"] == "test"


def test_sigma_stats_train_split_needs_no_flag(tmp_path):
    root = tmp_path / "crops"
    _crops(root, "train")
    S.main(["--crops-root", str(root), "--out-dir", str(tmp_path / "out")])
    assert (tmp_path / "out" / "sigma_stats_crops.json").exists()
