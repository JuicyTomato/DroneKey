import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dronepose.data.crops import label_box_xyxy

_spec = importlib.util.spec_from_file_location(
    "eval_2dronekey", Path(__file__).resolve().parents[2] / "scripts" / "dronekey" / "eval_2dronekey.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

GT = [[900.0, 400.0], [1100.0, 420.0], [1080.0, 560.0], [880.0, 540.0]]
SIDE = 100.0


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def files(tmp_path, preds: dict, split="val", n_gt=None):
    ids = list(preds) if n_gt is None else [f"{i:05d}" for i in range(n_gt)]
    gt = write_jsonl(tmp_path / f"gt_{split}.jsonl", [{"frame_id": i, "kpts2d": GT} for i in ids])
    crops = write_jsonl(tmp_path / f"crops_{split}.jsonl", [{"frame_id": i, "box": [800.0, 300.0, SIDE]} for i in ids])
    pred = write_jsonl(tmp_path / "pred.jsonl",
                       [{"frame_id": i, "kpts_pred": [[x, y, 1.0] for x, y in p]} for i, p in preds.items()])
    return pred, gt, crops


def shifted(dx):
    return [[x + dx, y] for x, y in GT]


def test_eval_2dk_two_readings_known_values(tmp_path):
    pred, gt, crops = files(tmp_path, {"00000": GT, "00001": shifted(0.02 * SIDE)})
    out = E.run(pred, gt, crops)
    x0, y0, x1, y1 = label_box_xyxy(np.array(GT))
    area = (x1 - x0) * (y1 - y0)
    o_i = [1.0, np.exp(-(2.0 ** 2) / (0.2 * area))]
    o_ii = [1.0, np.exp(-0.05)]                                    # (d/side)^2 / (0.2 * 0.2^2) = 4e-4 / 0.008
    for key, o in (("reading_i", o_i), ("reading_ii", o_ii)):
        assert out[key]["sr90"] == 1.0 and out[key]["sr95"] == 1.0
        assert out[key]["ap_paper"] == pytest.approx(1.0)
        assert out[key]["mean_oks"] == pytest.approx(np.mean(o))
    assert out["px_median"] == pytest.approx(1.0)                   # frames at 0 and 2 px -> median 1
    assert out["n_frames"] == 2 and out["n_missing"] == 0 and out["split"] == "val"


def test_eval_2dk_readings_differ_when_threshold_crossed(tmp_path):
    # d = 0.25 * side: reading (ii) OKS = exp(-0.0625/0.008) ~ 4e-4 -> miss; reading (i) with a big area stays high
    pred, gt, crops = files(tmp_path, {"00000": shifted(0.25 * SIDE)})
    out = E.run(pred, gt, crops)
    assert out["reading_ii"]["sr90"] == 0.0 and out["reading_ii"]["ap_paper"] == 0.0
    assert out["reading_i"]["sr90"] == 1.0


def test_eval_2dk_missing_and_nonfinite_count_as_miss(tmp_path):
    pred, gt, crops = files(tmp_path, {"00000": GT, "00001": GT}, n_gt=4)      # 00002, 00003 have no prediction
    rows = [json.loads(line) for line in pred.read_text().splitlines()]
    rows[1]["kpts_pred"][0][0] = float("nan")                                   # json.dumps writes NaN
    write_jsonl(pred, rows)
    out = E.run(pred, gt, crops)
    assert out["n_frames"] == 4 and out["n_missing"] == 3                       # 2 absent + 1 non-finite
    for key in ("reading_i", "reading_ii"):
        assert out[key]["sr90"] == 0.25 and out[key]["ap_paper"] == pytest.approx(0.25)
    assert out["px_median"] == 0.0                                              # only the one valid frame


def test_eval_2dk_refuses_test(tmp_path):
    pred, gt, crops = files(tmp_path, {"00000": GT}, split="test")
    with pytest.raises(ValueError, match="test"):
        E.run(pred, gt, crops)
    assert E.run(pred, gt, crops, allow_test=True)["split"] == "test"


def test_eval_2dk_main_writes_json_and_table(tmp_path, capsys):
    pred, gt, crops = files(tmp_path, {"00000": GT})
    out = tmp_path / "sub" / "out.json"
    E.main(["--pred", str(pred), "--gt", str(gt), "--crops", str(crops), "--out", str(out)])
    assert json.loads(out.read_text())["reading_i"]["ap_paper"] == pytest.approx(1.0)
    txt = capsys.readouterr().out
    assert "99.68" in txt and "99.8" in txt and "98.5" in txt
