import numpy as np

from dronepose.data.crops import CropBox, frame_to_crop
from dronepose.keypoints.export import crop_result_to_frame

BOX = CropBox(x0=100.0, y0=50.0, side=512.0)
KP = np.array([[200.0, 100.0], [300.0, 100.0], [300.0, 180.0], [200.0, 180.0]])


def test_best_detection_is_mapped_back_to_frame():
    good = frame_to_crop(KP, BOX, 256)
    other = good + 30.0
    out = crop_result_to_frame(np.stack([other, good]), np.array([[0.5] * 4, [0.9] * 4]), np.array([0.2, 0.8]), BOX, 256)
    np.testing.assert_allclose(out[:, :2], KP, atol=1e-9)
    np.testing.assert_allclose(out[:, 2], 0.9)


def test_missing_keypoint_confidence_defaults_to_one():
    out = crop_result_to_frame(frame_to_crop(KP, BOX, 256)[None], None, np.array([0.7]), BOX, 256)
    np.testing.assert_allclose(out[:, 2], 1.0)


def test_no_detection():
    for xy, conf in ((None, None), (np.zeros((0, 4, 2)), np.zeros(0))):
        out = crop_result_to_frame(xy, None, conf, BOX, 256)
        assert np.isnan(out[:, :2]).all() and (out[:, 2] == 0).all()
