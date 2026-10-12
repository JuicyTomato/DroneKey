import numpy as np
import pytest
import yaml

from dronepose.data.crops import CROP_SIZE, CropBox, crop_box_from_kpts, crop_image, crop_to_frame, frame_to_crop
from dronepose.data.export_yolo import CLASS_IDS, write_data_yaml, yolo_label_line

KP = np.array([[900.0, 500.0], [1100.0, 500.0], [1080.0, 560.0], [920.0, 560.0]])


def test_box_is_square_centred_and_scaled():
    b = crop_box_from_kpts(KP)
    assert b.side == pytest.approx(320.0)
    assert (b.x0 + b.side / 2, b.y0 + b.side / 2) == (pytest.approx(1000.0), pytest.approx(530.0))


def test_min_side_for_tiny_drones():
    assert crop_box_from_kpts(np.array([[10.0, 10], [12, 10], [12, 11], [10, 11]])).side == 64.0


def test_frame_crop_roundtrip():
    b = crop_box_from_kpts(KP)
    np.testing.assert_allclose(crop_to_frame(frame_to_crop(KP, b), b), KP, atol=1e-9)


def test_crop_outside_image_is_padded_and_keypoint_lands_where_mapped():
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[3:8, 3:8] = 255                                  # bright dot around pixel (5, 5)
    b = CropBox(x0=-100.0, y0=-100.0, side=256.0)        # mostly outside the image
    crop = crop_image(img, b)
    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)
    u, v = frame_to_crop(np.array([[5.0, 5.0]]), b)[0]
    assert crop[int(round(v)), int(round(u))].max() == 255
    assert crop[10, 10].max() == 0                       # padded area is black


def test_yolo_label_line_format():
    b = crop_box_from_kpts(KP)
    line = yolo_label_line(CLASS_IDS["05"], frame_to_crop(KP, b))
    vals = line.split()
    assert len(vals) == 1 + 4 + 4 * 3 and vals[0] == "4"
    nums = np.array(vals[1:], float)
    assert np.all((nums >= 0) & (nums <= 2))
    assert np.all(nums[4 + 2::3] == 2)                   # visibility v=2 for all propellers


def test_class_ids_and_yaml(tmp_path):
    assert CLASS_IDS == {"01": 0, "02": 1, "03": 2, "04": 3, "05": 4, "06": 5, "08": 6}
    write_data_yaml(tmp_path / "data.yaml", "/kaggle/input/x")
    y = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert y["kpt_shape"] == [4, 3] and y["flip_idx"] == [1, 0, 3, 2] and len(y["names"]) == 7
