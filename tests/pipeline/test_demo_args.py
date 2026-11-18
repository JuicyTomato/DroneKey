import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dronepose.data import mav6d
from dronepose.pipeline.sources import MAV6DSource, SixDroneSynSource

spec = importlib.util.spec_from_file_location("demo", Path(__file__).resolve().parents[2] / "scripts" / "demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

KPTS = [[0.15, 0.15, 0.0], [0.15, -0.15, 0.0], [-0.15, -0.15, 0.001], [-0.15, 0.15, 0.001]]
CAM_FROM_VICON = [[0.6685859, -0.74342, 0.01787715, -0.38], [0.01558769, -0.01002444, -0.99982825, 1.6],
                  [0.74347153, 0.66874974, 0.004886, 3.0]]


def _rec(seq, split, i, step):
    ts = 1_600_000_000_000_000_000 + i * step
    return {"frame_id": f"{seq}_{ts}", "image_path": f"/nonexistent/{i}.jpg", "sequence": seq, "frame_idx": i,
            "authors_split": split, "K": mav6d.K.tolist(), "dist": mav6d.DIST.tolist(),
            "kpts2d": [[10.0, 20.0]] * 4, "R_gt": np.eye(3).tolist(), "t_gt": [0.1, 0.2, 2.0],
            "drone": "05", "in_image": True}


@pytest.fixture
def mav_args(tmp_path):
    step = round(1e9 / 24)
    recs = [_rec("01_0101", "train", i, step) for i in range(4)] + [_rec("02_0201", "test", i, step) for i in range(3)]
    gt = tmp_path / "gt.jsonl"
    gt.write_text("\n".join(json.dumps(r) for r in recs))
    hub = tmp_path / "hub.json"
    hub.write_text(json.dumps({"kpts_flu": KPTS, "cam_from_vicon": CAM_FROM_VICON}))
    return ["--source", "mav6d", "--gt", str(gt), "--hub-model", str(hub), "--mode", "streaming", "--out", "x.mp4"]


def test_mav6d_test_sequence_exits(mav_args):
    args = demo.build_parser().parse_args(mav_args + ["--sequence", "02_0201"])
    with pytest.raises(SystemExit, match="test"):
        demo.resolve_source(args)


def test_mav6d_missing_sequence_exits(mav_args):
    args = demo.build_parser().parse_args(mav_args + ["--sequence", "01_01"])
    with pytest.raises(SystemExit, match="not found"):
        demo.resolve_source(args)


def test_default_6dronesyn_unchanged(tmp_path):
    canon = tmp_path / "c.jsonl"
    canon.write_text(json.dumps({"sequence": "14_02_07_000"}) + "\n")
    args = demo.build_parser().parse_args(["--sequence", "14_02", "--mode", "offline", "--out", "x.mp4",
                                           "--canonical", str(canon)])
    assert args.source == "6dronesyn" and args.fps is None and args.max_frames is None
    from unittest import mock
    with mock.patch.object(SixDroneSynSource, "__init__", lambda self, p: None), \
            mock.patch.object(SixDroneSynSource, "sequences", lambda self: ["14_02_07_000"]):
        src, seq, fps = demo.resolve_source(args)
    assert isinstance(src, SixDroneSynSource) and seq == "14_02_07_000" and fps == 30


def test_mav6d_fps_from_source(mav_args):
    args = demo.build_parser().parse_args(mav_args + ["--sequence", "01_0101"])
    src, seq, fps = demo.resolve_source(args)
    assert isinstance(src, MAV6DSource) and seq == "01_0101" and fps == 24.0
    args = demo.build_parser().parse_args(mav_args + ["--sequence", "01_0101", "--fps", "12"])
    assert demo.resolve_source(args)[2] == 12


def test_shown_forecast():
    fut = np.ones((30, 3))
    pose, nopose = {"R": np.eye(3)}, {"R": None}
    assert demo.shown_forecast(nopose, fut, True) is None
    assert demo.shown_forecast(nopose, fut, False) is fut
    assert demo.shown_forecast(pose, fut, True) is fut
    bad = fut.copy()
    bad[3, 1] = np.nan
    assert demo.shown_forecast(pose, bad, False) is None
    assert demo.shown_forecast(pose, None, True) is None


def test_no_forecast_flag(mav_args):
    base = mav_args + ["--sequence", "01_0101"]
    assert demo.build_parser().parse_args(base).no_forecast is False
    assert demo.build_parser().parse_args(base + ["--no-forecast"]).no_forecast is True


def test_no_forecast_header_has_no_previsione():
    from dronepose.pipeline.draw import band_lines
    from dronepose.pipeline.sources import Frame
    fr = Frame("f", 1, np.zeros((10, 10, 3), np.uint8), np.eye(3), {})
    res = {"R": None, "t": None, "drone_model": None}
    assert "previsione" in band_lines(fr, res, None, "offline", None)[1]
    assert "previsione" not in " ".join(band_lines(fr, res, None, "offline", 5.0, show_forecast=False))
