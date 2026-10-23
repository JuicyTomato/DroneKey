import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronepose.keypoints.dronekey.config import load_config

pytest.importorskip("torch")
pytest.importorskip("torchvision")

_spec = importlib.util.spec_from_file_location("overfit", Path("scripts/dronekey/overfit.py"))
overfit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(overfit)

KP = [(0.5, 0.2), (0.8, 0.5), (0.5, 0.8), (0.3, 0.5)]


def _crops(root, n=8):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        rng = np.random.default_rng(1)
        for i in range(n):
            cv2.imwrite(str(root / "images" / split / f"f_{i:04d}.jpg"), rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
            vals = " ".join(f"{x + 0.01 * i} {y} 2" for x, y in KP)
            (root / "labels" / split / f"f_{i:04d}.txt").write_text(f"0 0.5 0.5 0.9 0.9 {vals}\n")
    return root


def test_overfit_smoke(tmp_path):
    root = _crops(tmp_path / "crops")
    out = tmp_path / "overfit_paper.json"
    overfit.main(["--crops-root", str(root), "--variant", "paper", "--n", "8", "--epochs", "3", "--device", "cpu",
                  "--no-pretrained", "--tiny", "--out", str(out)])
    res = json.loads(out.read_text())
    assert {"variant", "n", "epochs", "lr", "cr_softplus", "curve", "final_px_median", "dead_relu_frac",
            "passed", "seconds"} <= set(res)
    assert res["variant"] == "paper" and res["n"] == 8 and res["epochs"] == 3 and res["cr_softplus"] is load_config()["model"]["cr_softplus"]
    assert len(res["curve"]) == 3 and res["final_px_median"] == res["curve"][-1]
    assert 0.0 <= res["dead_relu_frac"] <= 1.0
    assert res["passed"] == (res["final_px_median"] < 1.0)
    assert (tmp_path / "overfit_paper_run" / "history.json").exists()
    assert not list((tmp_path / "overfit_paper_run").glob("*.pt"))
