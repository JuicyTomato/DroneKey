import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronepose.data.crops import (CROP_SIZE, CropBox, crop_box_from_det_box, crop_box_from_kpts, crop_to_frame,
                                  frame_to_crop, label_box_xyxy)
from dronepose.data.export_yolo import yolo_label_line
from dronepose.eval.detection import detection_metrics, iou_xyxy
from dronepose.keypoints.detections import Detection
from dronepose.keypoints.two_stage import all_boxes, crop_from_detection, top_detection_row

KP = np.array([[900.0, 500.0], [1100.0, 500.0], [1080.0, 560.0], [920.0, 560.0]])
KP_EDGE = np.array([[10.0, 500.0], [210.0, 500.0], [190.0, 560.0], [30.0, 560.0]])   # label box clipped at x = 0
WH = np.array([1920.0, 1080.0, 1920.0, 1080.0])


def test_label_box_matches_yolo_label_line():
    for k in (KP, KP_EDGE):
        cx, cy, w, h = (float(v) for v in yolo_label_line(0, k, size=(1920, 1080)).split()[1:5])
        want = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        np.testing.assert_allclose(label_box_xyxy(k) / WH, want, atol=1e-5)


def test_exact_box_gives_setting_a_crop():
    a, b = crop_box_from_det_box(label_box_xyxy(KP)), crop_box_from_kpts(KP)
    np.testing.assert_allclose([a.x0, a.y0, a.side], [b.x0, b.y0, b.side], atol=1e-9)


def test_crop_from_detection_border_is_black():
    img = np.full((1080, 1920, 3), 200, np.uint8)
    crop, box = crop_from_detection(img, [-60.0, 400.0, 140.0, 600.0])        # partly outside on the left
    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)
    assert box.x0 < 0
    cols_out = int(np.floor(-box.x0 * CROP_SIZE / box.side)) - 1              # margin for interpolation / JPEG
    assert cols_out > 0
    assert crop[:, :cols_out].max() <= 2                                       # JPEG noise only
    assert crop[CROP_SIZE // 2, CROP_SIZE - 5].min() > 150                    # inside stays bright


def test_crop_roundtrip_coordinates():
    box = crop_box_from_det_box(label_box_xyxy(KP_EDGE))
    assert box.x0 < 0                                                          # box reaches outside the frame
    np.testing.assert_allclose(crop_to_frame(frame_to_crop(KP_EDGE, box), box), KP_EDGE, atol=1e-6)
    b = CropBox(-100.0, -50.0, 300.0)
    np.testing.assert_allclose(crop_to_frame(frame_to_crop(KP_EDGE, b), b), KP_EDGE, atol=1e-6)


def test_top_detection_row_picks_max_conf_and_scales():
    xyxy = np.array([[0.0, 0, 10, 10], [10.0, 20, 30, 40]])
    conf = np.array([0.4, 0.9])
    cls = np.array([1.0, 3.0])
    kxy = np.stack([np.zeros((4, 2)), np.arange(8.0).reshape(4, 2)])
    kconf = np.stack([np.full(4, 0.1), np.full(4, 0.7)])
    row = top_detection_row("f", xyxy, conf, cls, kxy, kconf, 3.0)
    assert row["box"] == [30.0, 60.0, 90.0, 120.0]
    assert row["box_conf"] == pytest.approx(0.9)
    assert row["cls"] == 3 and isinstance(row["cls"], int)
    np.testing.assert_allclose(np.array(row["kpts_pred"])[:, :2], np.arange(8.0).reshape(4, 2) * 3)
    np.testing.assert_allclose(np.array(row["kpts_pred"])[:, 2], 0.7)
    json.dumps(row, allow_nan=True)
    row = top_detection_row("f", xyxy, conf, cls, kxy, None, 1.0)               # no keypoint confidences
    assert [k[2] for k in row["kpts_pred"]] == [1.0] * 4


def test_top_detection_row_empty():
    row = top_detection_row("f", np.zeros((0, 4)), np.zeros(0), np.zeros(0), np.zeros((0, 4, 2)), None, 3.0)
    assert row["box"] is None and row["cls"] is None and row["box_conf"] == 0.0
    k = np.array(row["kpts_pred"])
    assert k.shape == (4, 3) and np.isnan(k[:, :2]).all() and (k[:, 2] == 0).all()


def test_iou_and_detection_metrics():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    recs = [{"frame_id": f"f{i}", "drone": d, "kpts2d": KP.tolist()} for i, d in enumerate(["03", "03", "01", "02"])]
    box = tuple(label_box_xyxy(KP))
    dets = {"f0": Detection(box, 0.9, "03"),
            "f1": Detection(box, 0.8, "06"),                  # confusion 03 -> 06
            "f2": Detection(None, 0.0, None)}                 # no detection; f3 has no row at all
    m = detection_metrics(dets, recs)
    assert m["no_det_rate"] == pytest.approx(0.5)
    assert m["iou_median"] == pytest.approx(1.0) and m["iou_ge_05"] == 1.0
    assert m["class_acc"] == pytest.approx(0.5)
    assert m["class_acc_by_drone"] == {"03": 0.5}
    assert m["confusion"] == {"03": {"03": 1, "06": 1}}


def test_detection_metrics_all_missing():
    m = detection_metrics({}, [{"frame_id": "a", "drone": "01", "kpts2d": KP.tolist()}])
    assert m["no_det_rate"] == 1.0 and m["iou_median"] is None and m["class_acc"] is None


@pytest.mark.dataset
def test_gt_box_crops_reproduce_setting_a():
    root = Path("data/processed")
    crops = root / "yolo_crops/images/val"
    if not crops.is_dir():
        pytest.skip("data/processed/yolo_crops missing")
    with open(root / "canonical_val.jsonl") as f:
        recs = [json.loads(l) for l in f][::997][:8]
    assert len(recs) == 8
    for r in recs:
        img = cv2.imread(r["image_path"])
        crop, _ = crop_from_detection(img, label_box_xyxy(np.array(r["kpts2d"], float)))
        ref = cv2.imread(str(crops / f"{r['frame_id']}.jpg"))
        assert np.abs(crop.astype(int) - ref.astype(int)).mean() < 2


def test_all_boxes_sorted_and_scaled():
    xyxy = np.array([[0, 0, 10, 10], [1, 2, 3, 4]], float)
    out = all_boxes(xyxy, np.array([0.3, 0.8]), np.array([2.0, 5.0]), 3.0)
    assert out == [[3.0, 6.0, 9.0, 12.0, 0.8, 5.0], [0.0, 0.0, 30.0, 30.0, 0.3, 2.0]]
    assert all_boxes(np.zeros((0, 4)), np.zeros(0), np.zeros(0), 3.0) == []
