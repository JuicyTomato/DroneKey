import numpy as np
import pytest

from dronepose.data.crops import CropBox
from dronepose.data.masks import (area_limits, diff_mask, geom_key, keypoints_in_mask, mask_iou, native_to_crop,
                                  native_to_frame, soften)


def test_geom_key():
    assert geom_key("01_05_13_000_0100") == "01_05_000_0100"


def test_keypoints_in_mask_uses_radius():
    m = np.zeros((20, 20), np.uint8)
    m[10, 10] = 1
    assert keypoints_in_mask(m, np.array([[12.0, 10.0]]), radius=3)
    assert not keypoints_in_mask(m, np.array([[18.0, 18.0]]), radius=3)


def test_keypoints_outside_image_are_not_in_mask():
    m = np.ones((20, 20), np.uint8)
    assert not keypoints_in_mask(m, np.array([[30.0, 10.0]]))


def test_area_limits_are_per_drone():
    assert area_limits({"05": [0.1, 0.2, 0.3], "08": [0.4]}) == pytest.approx({"05": 0.5, "08": 1.0})


def test_keypoints_alone_do_not_accept_a_full_mask():
    m = np.ones((10, 10), np.uint8)
    assert keypoints_in_mask(m, np.array([[5.0, 5.0]])) and m.mean() > area_limits({"05": [0.1]})["05"]


def test_diff_mask_finds_common_object():
    a, b = np.full((40, 40, 3), 30, np.uint8), np.full((40, 40, 3), 200, np.uint8)
    a[15:25, 15:25] = 120
    b[15:25, 15:25] = 120
    m = diff_mask([a, b], np.array([[20.0, 20.0]]))
    assert m[20, 20] == 1 and 96 <= m.sum() <= 100   # l'apertura ellittica smussa gli angoli


def test_diff_mask_drops_components_far_from_keypoints():
    a, b = np.full((60, 60, 3), 30, np.uint8), np.full((60, 60, 3), 200, np.uint8)
    for x in (a, b):
        x[5:15, 5:15] = 120
        x[40:50, 40:50] = 120
    m = diff_mask([a, b], np.array([[45.0, 45.0]]))
    assert m[45, 45] == 1 and m[10, 10] == 0


def test_mask_iou():
    a, b = np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8)
    a[:2] = 1
    b[1:3] = 1
    assert mask_iou(a, b) == pytest.approx(1 / 3)
    assert mask_iou(np.zeros((4, 4)), np.zeros((4, 4))) == 0


def test_soften_keeps_inside_and_softens_edge():
    m = np.zeros((20, 20), np.uint8)
    m[5:15, 5:15] = 255
    s = soften(m)
    assert s[10, 10] == 255 and s[0, 0] == 0 and 0 < s[10, 4] < 255


def test_native_to_crop_size():
    assert native_to_crop(np.full((512, 512), 255, np.uint8)).shape == (256, 256)


def test_native_to_frame_matches_box():
    out = native_to_frame(np.full((300, 300), 255, np.uint8), CropBox(300, 150, 300))
    assert out.shape == (360, 640) and out[100, 150] > 200 and out[10, 10] == 0


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_select_renders_one_per_geometry():
    b = _load("scripts/aug/build_sam_inputs.py", "build_sam_inputs")
    recs = [{"frame_id": f"01_05_{bg}_000_0100"} for bg in ("13", "14", "15")] + [{"frame_id": "02_04_13_000_0001"}]
    assert [r["frame_id"] for r in b.select_renders(recs, 1)] == ["01_05_14_000_0100", "02_04_13_000_0001"]


def test_sam_prompts_stay_inside_crop():
    s = _load("scripts/kaggle/sam_masks.py", "sam_masks")
    box, pts, labels = s.prompts([[10, 50], [90, 50], [50, 10], [50, 90]], 100)
    assert box == [0, 0, 100, 100] and labels == [1, 1, 1, 1] and len(pts) == 4
