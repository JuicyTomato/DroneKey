import json
from pathlib import Path

import numpy as np
import pytest

from dronepose.forecast.config import load_config
from dronepose.forecast.dataset import load_trajectories
from dronepose.forecast.e4 import load_pose_windows
from dronepose.forecast.nets.evaluate import predict_run
from dronepose.forecast.nets.features import build_batch, fill_lost
from dronepose.pipeline.decide import write_pose_file
from dronepose.pipeline.forecast_now import Forecaster
from tests.pipeline.test_decide import _fake_results

RUN = Path("experiments/forecast/e4/runs")
needs = pytest.mark.skipif(not RUN.exists(), reason="E4 runs missing")


@needs
def test_forecaster_matches_bulk(tmp_path):
    cfg = load_config(Path("configs/forecast.json"))
    tr = [t for t in load_trajectories(Path("data/processed/forecast"), "val") if t.drone != "08" and t.scene == "14"][0]
    rng = np.random.default_rng(1)
    by_seq, recs = {}, {}
    for bg, deg in tr.copies[:1]:
        seq = f"{tr.scene}_{tr.drone}_{bg}_{int(deg):03d}"
        res = _fake_results(tr, seq, rng, drop=(5, 6))
        for k, r in enumerate(res):                       # a smooth path, so the forecast is meaningful
            if r["t"] is not None:
                r["t"] = [0.01 * k, 0.002 * k * k / 30, 10.0]
        by_seq[seq] = res
        recs.update({r["frame_id"]: {"frame_id": r["frame_id"], "drone": tr.drone} for r in res})
    tr.copies = tr.copies[:1]
    path = tmp_path / "p.jsonl"
    write_pose_file(by_seq, recs, "val", 0, "t", path)
    w, _ = load_pose_windows(path, [tr], cfg)
    w = w.subset(np.arange(5))
    f = Forecaster("14")
    batch = build_batch(w, {**f.params, "cv_k": 2}, cfg, mask_channel=True)
    bulk = predict_run(f.run, batch, f.device)
    fi = fill_lost(w.obs_valid)
    for i in range(len(w)):
        anchor = w.obs_p[i, fi[i, -1]] + w.last_p[i]
        got = f.predict(w.obs_p[i] + w.last_p[i], w.obs_R[i], w.obs_valid[i], tr.drone)
        assert np.allclose(got, bulk[i] + anchor, atol=1e-5)


@needs
def test_forecaster_none_cases():
    f = Forecaster("14")
    v = np.zeros(15, bool)
    v[-1] = True
    assert f.predict(np.zeros((15, 3)), np.tile(np.eye(3), (15, 1, 1)), v, "02") is None
    assert f.predict(np.zeros((15, 3)), np.tile(np.eye(3), (15, 1, 1)), np.ones(15, bool), "08") is None


def test_final_split_paths(tmp_path, monkeypatch):
    import dronepose.pipeline.forecast_now as fn
    cfg = tmp_path / "pipeline.json"
    cfg.write_text(json.dumps({"forecast_params": str(tmp_path), "forecast_runs": str(tmp_path / "runs"), "seed": 0}))
    read = []

    def fake_read(self):
        read.append(Path(self).name)
        return "{}"

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    monkeypatch.setattr(Path, "read_text", lambda self: fake_read(self) if self.name.startswith("params_") else
                        Path.read_bytes(self).decode())
    monkeypatch.setattr(fn, "load_config", lambda p: None)
    monkeypatch.setattr(fn, "load_checkpoint", stop)
    for scene, tag in (("final", "final"), ("03", "fold03")):
        read.clear()
        with pytest.raises(Stop):
            fn.Forecaster(scene, cfg_path=cfg)
        assert read == [f"params_{tag}.json"]
        assert tag in fn._run_name(scene, 0)
    assert "foldfinal" not in fn._run_name("final", 0)


@pytest.mark.skipif(not Path("experiments/forecast/e4/runs/R-CV_noisy_gru_6d_final_s0/best.pt").exists(),
                    reason="final run missing")
def test_forecaster_final_builds():
    f = Forecaster("final")
    assert f.run.name == "R-CV_noisy_gru_6d_final_s0"
