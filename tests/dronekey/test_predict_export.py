import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from dronepose.data.crops import CROP_SIZE, CropBox, crop_box_from_kpts, crop_to_frame, label_box_xyxy  # noqa: E402
from dronepose.keypoints.detections import Detection  # noqa: E402
from dronepose.keypoints.dronekey import predict as P  # noqa: E402
from dronepose.keypoints.dronekey.config import load_config, resolve_variant  # noqa: E402
from dronepose.keypoints.dronekey.data import CropDataset, preprocess_crop  # noqa: E402
from dronepose.keypoints.dronekey.model import TinyBackbone, build_model  # noqa: E402

CFG = load_config()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "dronekey"))

KP = [[900.0, 400.0], [1100.0, 420.0], [1080.0, 560.0], [880.0, 540.0]]


class HalfPredictor:
    """Stands in for the network: always (0.5, 0.5) normalized to the crop, for all four propellers."""

    def predict_crops(self, crops_bgr, boxes, batch: int = 64):
        assert all(c.shape == (CROP_SIZE, CROP_SIZE, 3) for c in crops_bgr)
        return [np.hstack([crop_to_frame(np.full((4, 2), 0.5 * CROP_SIZE), b), np.ones((4, 1))]) for b in boxes]


def _record(i, split="val", kp=KP):
    return {"frame_id": f"01_01_01_000_{i:04d}", "split": split, "image_path": f"/nowhere/{i}.png",
            "kpts2d": [[x + 3 * i, y] for x, y in kp]}


def _blank(path):
    return np.zeros((1080, 1920, 3), np.uint8)


def _crops_root(tmp_path, recs, split="val"):
    root = tmp_path / "crops"
    root.mkdir()
    with open(root / f"crops_{split}.jsonl", "w") as f:
        for r in reversed(recs):  # unsorted on purpose
            b = crop_box_from_kpts(np.array(r["kpts2d"]))
            f.write(json.dumps({"frame_id": r["frame_id"], "box": [b.x0, b.y0, b.side]}) + "\n")
    return root


def _centers(rows):
    return np.array([np.array(r["kpts_pred"])[:, :2].mean(0) for r in rows])


def test_setting_a_maps_crop_center_to_frame(tmp_path):
    recs = [_record(i) for i in range(3)]
    root = _crops_root(tmp_path, recs)
    rows = P.rows_setting_a(HalfPredictor(), root, "val", None, read_image=lambda p: np.zeros((256, 256, 3), np.uint8))
    assert [r["frame_id"] for r in rows] == sorted(r["frame_id"] for r in recs)   # ordered by frame_id
    for row, rec in zip(rows, recs):
        b = crop_box_from_kpts(np.array(rec["kpts2d"]))
        k = np.array(row["kpts_pred"])
        assert k.shape == (4, 3) and (k[:, 2] == 1.0).all()
        np.testing.assert_allclose(k[:, :2], [[b.x0 + b.side / 2, b.y0 + b.side / 2]] * 4, atol=1e-6)


def test_setting_a_selects_frame_ids_and_reads_crop_path(tmp_path):
    recs = [_record(i) for i in range(3)]
    root = _crops_root(tmp_path, recs)
    seen = []
    rows = P.rows_setting_a(HalfPredictor(), root, "val", [recs[2]["frame_id"], recs[0]["frame_id"]],
                            read_image=lambda p: seen.append(Path(p)) or np.zeros((256, 256, 3), np.uint8))
    assert [r["frame_id"] for r in rows] == [recs[0]["frame_id"], recs[2]["frame_id"]]
    assert seen[0] == root / "images" / "val" / f"{recs[0]['frame_id']}.jpg"


def test_setting_b_with_gt_box_matches_a(tmp_path):
    recs = [_record(i) for i in range(4)]
    root = _crops_root(tmp_path, recs)
    a = P.rows_setting_a(HalfPredictor(), root, "val", None, read_image=lambda p: np.zeros((256, 256, 3), np.uint8))
    dets = {r["frame_id"]: Detection(tuple(label_box_xyxy(np.array(r["kpts2d"]))), 0.9, "x") for r in recs}
    b = P.rows_setting_b(HalfPredictor(), recs, dets, read_image=_blank)
    assert [r["frame_id"] for r in b] == [r["frame_id"] for r in a]
    assert np.abs(_centers(a) - _centers(b)).max() < 1.0


def test_export_b_missing_box_rows():
    recs = [_record(i) for i in range(4)]
    dets = {recs[0]["frame_id"]: Detection(tuple(label_box_xyxy(np.array(recs[0]["kpts2d"]))), 0.9, "x"),
            recs[1]["frame_id"]: Detection(None, 0.0, None)}   # recs[2], recs[3] absent from the file
    rows = P.rows_setting_b(HalfPredictor(), recs, dets, read_image=_blank)
    assert [r["frame_id"] for r in rows] == [r["frame_id"] for r in recs]
    assert np.isfinite(np.array(rows[0]["kpts_pred"])).all()
    for r in rows[1:]:
        k = np.array(r["kpts_pred"])
        assert k.shape == (4, 3) and np.isnan(k[:, :2]).all() and (k[:, 2] == 0).all()


def test_export_refuses_test(tmp_path):
    recs = [_record(0, split="test")]
    root = _crops_root(tmp_path, recs, split="test")
    with pytest.raises(ValueError):
        P.rows_setting_a(HalfPredictor(), root, "test", None, read_image=lambda p: np.zeros((256, 256, 3), np.uint8))
    with pytest.raises(ValueError):
        P.rows_setting_b(HalfPredictor(), recs, {}, read_image=_blank)
    rows = P.rows_setting_a(HalfPredictor(), root, "test", None, allow_test=True,
                            read_image=lambda p: np.zeros((256, 256, 3), np.uint8))
    assert len(rows) == 1


def _tiny_model_cfg():
    cfg = resolve_variant(CFG, "paper")
    cfg["model"].update(d=16, n_layers=2, nhead=2, ffn=32)
    return cfg


def test_preprocess_crop_matches_dataset_eval_path(tmp_path):
    root = tmp_path / "c"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "images" / "val" / "f_0.jpg"), img)
    (root / "labels" / "val" / "f_0.txt").write_text("0 .5 .5 .9 .9 " + " ".join(f"{x} {y} 2" for x, y in [(.5, .2), (.8, .5), (.5, .8), (.3, .5)]))
    cfg = _tiny_model_cfg()
    ds = CropDataset(root, "val", "none", train=False, input_cfg=cfg["input"])
    x_ds = ds[0][0]
    x = preprocess_crop(cv2.imread(str(root / "images" / "val" / "f_0.jpg")), cfg["input"])
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32
    assert torch.equal(x, x_ds)


def test_predict_crops_real_model_shapes(tmp_path):
    cfg = _tiny_model_cfg()
    torch.manual_seed(0)
    model = build_model(cfg["model"], pretrained=False, tiny=True)
    ck = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "run_cfg": cfg, "epoch": 1, "val_px_median": 1.0}, ck)
    pred = P.load_predictor(ck, None, "cpu", tiny=True)
    assert not pred.model.training
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8) for _ in range(5)]
    boxes = [CropBox(100.0 * i, 50.0, 400.0) for i in range(5)]
    out = pred.predict_crops(crops, boxes, batch=2)   # 3 chunks
    assert len(out) == 5 and all(o.shape == (4, 3) and (o[:, 2] == 1.0).all() for o in out)
    # batching does not change the result, and the mapping is crop_to_frame(8 values * 256)
    one = pred.predict_crops(crops[:1], boxes[:1], batch=64)[0]
    np.testing.assert_allclose(out[0], one, atol=1e-4)
    x = preprocess_crop(crops[0], cfg["input"])[None]
    with torch.no_grad():
        y = model.eval()(x).numpy().reshape(4, 2) * CROP_SIZE
    np.testing.assert_allclose(one[:, :2], crop_to_frame(y, boxes[0]), atol=1e-3)
    assert pred.predict_crops([], []) == []


def test_load_predictor_variant_mismatch(tmp_path):
    cfg = _tiny_model_cfg()
    model = build_model(cfg["model"], pretrained=False, tiny=True)
    ck = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "run_cfg": cfg, "epoch": 1, "val_px_median": 1.0}, ck)
    P.load_predictor(ck, None, "cpu", variant="paper", tiny=True)   # agrees on wiring/head (d differs: ignored)
    with pytest.raises(ValueError):
        P.load_predictor(ck, None, "cpu", variant="code", tiny=True)


def test_export_cli_end_to_end_tiny(tmp_path):
    import export_dronekey as E

    cfg = _tiny_model_cfg()
    torch.manual_seed(0)
    model = build_model(cfg["model"], pretrained=False, tiny=True)
    ck = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "run_cfg": cfg, "epoch": 1, "val_px_median": 1.0}, ck)
    recs = [_record(i) for i in range(3)]
    root = _crops_root(tmp_path, recs)
    (root / "images" / "val").mkdir(parents=True)
    for r in recs:
        cv2.imwrite(str(root / "images" / "val" / f"{r['frame_id']}.jpg"), np.full((256, 256, 3), 90, np.uint8))
    out = tmp_path / "o.jsonl"
    E.main(["--weights", str(ck), "--variant", "paper", "--crops-root", str(root), "--split", "val",
            "--setting", "A", "--out", str(out), "--device", "cpu", "--tiny"])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["frame_id"] for r in rows] == sorted(r["frame_id"] for r in recs)
    assert all(np.array(r["kpts_pred"]).shape == (4, 3) for r in rows)
    with pytest.raises(ValueError):
        E.main(["--weights", str(ck), "--variant", "paper", "--crops-root", str(root), "--split", "test",
                "--setting", "A", "--out", str(out), "--device", "cpu", "--tiny"])


def test_export_cli_setting_b_tiny(tmp_path):
    import export_dronekey as E

    cfg = _tiny_model_cfg()
    model = build_model(cfg["model"], pretrained=False, tiny=True)
    ck = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "run_cfg": cfg, "epoch": 1, "val_px_median": 1.0}, ck)
    recs = [_record(i) for i in range(3)]
    png = tmp_path / "f.png"
    cv2.imwrite(str(png), np.full((1080, 1920, 3), 80, np.uint8))
    for r in recs:
        r["image_path"] = str(png)
    canon = tmp_path / "canonical_val.jsonl"
    canon.write_text("".join(json.dumps(r) + "\n" for r in recs))
    det = tmp_path / "det.jsonl"
    det.write_text("".join(json.dumps({"frame_id": r["frame_id"], "box_conf": 0.9, "cls": 0,
                                       "box": None if i == 2 else label_box_xyxy(np.array(r["kpts2d"])).tolist()}) + "\n"
                           for i, r in enumerate(recs)))
    out = tmp_path / "b.jsonl"
    E.main(["--weights", str(ck), "--variant", "paper", "--split", "val", "--setting", "B", "--det", str(det),
            "--canonical", str(canon), "--out", str(out), "--device", "cpu", "--tiny"])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["frame_id"] for r in rows] == [r["frame_id"] for r in recs]
    assert np.isfinite(np.array(rows[0]["kpts_pred"])).all() and np.isnan(np.array(rows[2]["kpts_pred"])[:, :2]).all()
