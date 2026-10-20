import numpy as np
import pytest

from dronepose.keypoints.recipes import RECIPES, albumentations_for


def test_w1_matches_chapter_8():
    assert RECIPES["w1"]["degrees"] == 30.0 and RECIPES["w1"]["scale"] == 0.3 and RECIPES["w1"]["mosaic"] == 0.0
    assert albumentations_for("w1") is None


def test_final_keeps_geometry():
    for k in ("fliplr", "degrees", "scale", "translate"):
        assert RECIPES["final"][k] == RECIPES["w1"][k]


def test_final_has_no_blur_and_runs():
    A = pytest.importorskip("albumentations")
    tf = albumentations_for("final", p=1.0)
    assert not any("Blur" in type(t).__name__ for t in tf)
    img = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    out = A.Compose(tf)(image=img)["image"]
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_default_probability_is_0_3():
    pytest.importorskip("albumentations")
    assert all(t.p == 0.3 for t in albumentations_for("final"))


def test_unknown_recipe_raises():
    with pytest.raises(KeyError):
        albumentations_for("nope")
