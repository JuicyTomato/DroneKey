import cv2
import numpy as np
import pytest

from dronepose.keypoints.dronekey import augment as aug
from dronepose.keypoints.recipes import RECIPES

FLIP = [1, 0, 3, 2]


def _apply(M, x, y):
    return (M @ np.array([x, y, 1.0]))[:2]


def test_affine_matrix_rotation_90():
    # cv2 convention (y down): +90 maps the offset (10, 0) from the centre to (0, -10)
    M = aug.affine_matrix(256, 256, 90.0, 1.0, 0.5, 0.5)
    assert M.shape == (3, 3)
    np.testing.assert_allclose(_apply(M, 138, 128), (128, 118), atol=1e-4)


def test_affine_matrix_scale_about_center():
    M = aug.affine_matrix(256, 256, 0.0, 2.0, 0.5, 0.5)
    np.testing.assert_allclose(_apply(M, 138, 128), (148, 128), atol=1e-4)


def test_affine_matrix_translation():
    M = aug.affine_matrix(256, 256, 0.0, 1.0, 0.6, 0.4)
    np.testing.assert_allclose(_apply(M, 100, 100), (100 + 0.1 * 256, 100 - 0.1 * 256), atol=1e-4)


def test_random_affine_moves_keypoints_with_image():
    img = np.zeros((256, 256, 3), np.uint8)
    kpts = np.array([[100.0, 120.0], [180.0, 90.0], [60.0, 200.0], [200.0, 200.0]])
    for k in kpts:
        cv2.circle(img, (int(k[0]), int(k[1])), 6, (255, 255, 255), -1)
    out, kp = aug.random_affine(img, kpts, 30.0, 0.1, 0.3, np.random.default_rng(3))
    assert out.shape == img.shape and not np.allclose(kp, kpts)
    for x, y in kp:
        assert out[int(round(y)), int(round(x))].min() > 200  # disc follows its keypoint (+-1 px)


def test_random_affine_border_is_114():
    img = np.full((64, 64, 3), 255, np.uint8)
    out, _ = aug.random_affine(img, np.zeros((0, 2)), 0.0, 0.5, 0.0, np.random.default_rng(1))  # shifted: border shows
    assert (out == 114).any() and (out == 255).any()


def test_hsv_zero_gain_identity():
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    out = aug.augment_hsv(img.copy(), 0.0, 0.0, 0.0, np.random.default_rng(0))
    np.testing.assert_array_equal(out, img)


def test_hsv_changes_image():
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    out = aug.augment_hsv(img.copy(), 0.015, 0.7, 0.4, np.random.default_rng(0))
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.array_equal(out, img)


def _hsv_of(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(float)


def test_hsv_matches_ultralytics_8_4_formula():
    gains = (0.3, 0.3, 0.3)
    r = np.random.default_rng(11).uniform(-1, 1, 3) * np.array(gains)  # same draw as augment_hsv
    h0, s0, v0 = 60, 200, 180
    img = np.tile(cv2.cvtColor(np.array([[[h0, s0, v0]]], np.uint8), cv2.COLOR_HSV2BGR), (8, 8, 1))
    out = aug.augment_hsv(img.copy(), *gains, np.random.default_rng(11))
    assert (out == out[0, 0]).all()  # constant image stays constant
    h, s, v = _hsv_of(out)
    assert abs(h - (h0 + r[0] * 180) % 180) <= 2.5
    assert abs(s - min(s0 * (1 + r[1]), 255)) <= 3
    assert abs(v - min(v0 * (1 + r[2]), 255)) <= 3
    assert abs(r[0] * 180) > 5  # the hue shift under test is not negligible


def test_hsv_pure_white_stays_white():
    img = np.full((8, 8, 3), 255, np.uint8)
    out = aug.augment_hsv(img.copy(), 0.5, 0.9, 0.0, np.random.default_rng(4))  # v gain 0: brightness unchanged
    np.testing.assert_array_equal(out, img)


def test_fliplr_reorders():
    img = np.zeros((256, 256, 3), np.uint8)
    img[:, :10] = 255
    kpts = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
    out, kp = aug.fliplr(img, kpts, FLIP)
    np.testing.assert_allclose(kp, [[256 - 30, 40], [256 - 10, 20], [256 - 70, 80], [256 - 50, 60]])
    assert out[:, -10:].min() == 255 and out[:, :10].max() == 0


def test_recipe_none_identity():
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    kpts = np.random.default_rng(1).uniform(0, 64, (4, 2))
    out, kp = aug.apply_recipe(img, kpts, "none", np.random.default_rng(2))
    np.testing.assert_array_equal(out, img)
    np.testing.assert_array_equal(kp, kpts)


def test_apply_recipe_reads_recipes_and_is_seeded():
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    kpts = np.random.default_rng(1).uniform(10, 50, (4, 2))
    a = aug.apply_recipe(img, kpts, "final", np.random.default_rng(5), flip_idx=FLIP)
    b = aug.apply_recipe(img, kpts, "final", np.random.default_rng(5), flip_idx=FLIP)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.allclose(a[1], kpts)
    with pytest.raises(KeyError):
        aug.apply_recipe(img, kpts, "nope", np.random.default_rng(0), flip_idx=FLIP)
    assert "final" in RECIPES


def test_default_albumentations_w1_optional():
    t = aug.default_albumentations_w1()
    assert t is None or len(t) == 4


@pytest.mark.parametrize("recipe", ["w1", "final"])
def test_albumentations_path_runs(recipe):
    pytest.importorskip("albumentations")  # optional dependency; installed on Kaggle, absent locally
    img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    kpts = np.random.default_rng(1).uniform(30, 220, (4, 2))
    out, kp = aug.apply_recipe(img, kpts, recipe, np.random.default_rng((0, 1, 2)))
    assert out.shape == img.shape and out.dtype == np.uint8 and kp.shape == (4, 2)
    out2, kp2 = aug.apply_recipe(img, kpts, recipe, np.random.default_rng((0, 1, 2)))
    np.testing.assert_array_equal(out, out2)  # fixed (seed, epoch, index) -> same sample
    np.testing.assert_array_equal(kp, kp2)


@pytest.mark.parametrize("recipe", ["w1", "final"])
def test_pixel_step_keeps_shape_and_is_seeded(recipe):
    pytest.importorskip("albumentations")
    img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    changed = 0
    for i in range(40):
        a = aug._pixel_step(img, recipe, np.random.default_rng((0, 0, i)))
        b = aug._pixel_step(img, recipe, np.random.default_rng((0, 0, i)))
        assert a.shape == img.shape and a.dtype == np.uint8
        np.testing.assert_array_equal(a, b)
        changed += not np.array_equal(a, img)
    if recipe == "final":  # p = 0.3 per transform: some draw must alter the pixels
        assert changed > 0


def test_pixel_step_does_not_move_keypoints(monkeypatch):
    pytest.importorskip("albumentations")
    monkeypatch.setitem(RECIPES, "pixel_only", {"fliplr": 0.0, "degrees": 0.0, "scale": 0.0, "translate": 0.0,
                                                "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0})
    img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    kpts = np.random.default_rng(1).uniform(30, 220, (4, 2))
    for i in range(10):  # identity warp: only the Albumentations list (that of "final") can act
        _, kp = aug.apply_recipe(img, kpts, "pixel_only", np.random.default_rng((0, 0, i)))
        np.testing.assert_allclose(kp, kpts, atol=1e-3)
