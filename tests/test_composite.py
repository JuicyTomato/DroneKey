import cv2
import numpy as np
import pytest

from dronepose.keypoints.composite import (Composer, composer_for, motion_kernel, paste, sample_background)
from dronepose.keypoints.recipes import COMPOSITE, RECIPES, CompositeSpec


@pytest.fixture
def comp_dirs(tmp_path):
    masks, bgs = tmp_path / "masks", tmp_path / "bgs"
    masks.mkdir()
    bgs.mkdir()
    m = np.zeros((256, 256), np.uint8)
    m[100:156, 100:156] = 255
    cv2.imwrite(str(masks / "01_05_000_0100.png"), m)
    for i, v in enumerate((200, 200, 200)):
        cv2.imwrite(str(bgs / f"{i}.jpg"), np.full((360, 640, 3), v, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return masks, bgs


def test_recipes_add_only_new_names():
    assert RECIPES["cp50"] == RECIPES["final"] and COMPOSITE["cp90"].p_paste == 0.9
    assert COMPOSITE["bl20"] == CompositeSpec(0.0, 0.5, 20.0)
    assert COMPOSITE["c50b10"] == CompositeSpec(0.5, 0.5, 10.0)
    assert set(COMPOSITE) == {"cp50", "cp90", "bl10", "bl20", "c50b10", "c50b20", "c90b10", "c90b20"}
    assert composer_for("final", None, None) is None and composer_for("w1", None, None) is None


@pytest.mark.parametrize("L,a", [(1, 0), (7, 33), (20, 90)])
def test_motion_kernel_normalised_and_centred(L, a):
    k = motion_kernel(L, a)
    assert k.shape[0] % 2 == 1 and k.sum() == pytest.approx(1)
    assert np.allclose(k, k[::-1, ::-1], atol=1e-5)


def test_paste_keeps_drone_and_replaces_rest():
    img, bg = np.full((8, 8, 3), 10, np.uint8), np.full((8, 8, 3), 200, np.uint8)
    a = np.zeros((8, 8), float)
    a[2:4, 2:4] = 1
    out = paste(img, a, bg)
    assert (out[2:4, 2:4] == 10).all() and (out[6:, 6:] == 200).all()


def test_paste_removes_black_padding():
    img = np.zeros((8, 8, 3), np.uint8)
    assert (paste(img, np.zeros((8, 8)), np.full((8, 8, 3), 200, np.uint8)) == 200).all()


def test_sample_background_upscales_small_source():
    bg = np.random.default_rng(0).integers(0, 255, (30, 40, 3), dtype=np.uint8)
    assert sample_background(bg, 256, 256, np.random.default_rng(1)).shape == (256, 256, 3)


def test_sample_background_changes_with_rng():
    bg = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    a = sample_background(bg, 256, 256, np.random.default_rng(1))
    b = sample_background(bg, 256, 256, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_missing_mask_leaves_image_unchanged(comp_dirs):
    c = Composer(CompositeSpec(1.0, 0.0, 0.0), *comp_dirs)
    img = np.full((256, 256, 3), 7, np.uint8)
    assert np.array_equal(c(img.copy(), "99_99_01_000_0001", np.random.default_rng(0)), img)


def test_empty_mask_dir_raises(tmp_path, comp_dirs):
    (tmp_path / "vuota").mkdir()
    with pytest.raises(FileNotFoundError):
        Composer(CompositeSpec(1.0, 0.0, 0.0), tmp_path / "vuota", comp_dirs[1])


def test_empty_background_dir_raises(tmp_path, comp_dirs):
    (tmp_path / "vuota").mkdir()
    with pytest.raises(FileNotFoundError):
        Composer(CompositeSpec(1.0, 0.0, 0.0), comp_dirs[0], tmp_path / "vuota")


def test_paste_needs_dirs():
    with pytest.raises(ValueError):
        Composer(CompositeSpec(1.0, 0.0, 0.0), None, None)


def test_mask_shared_by_the_three_renders(comp_dirs):
    c = Composer(CompositeSpec(1.0, 0.0, 0.0), *comp_dirs)
    img = np.full((256, 256, 3), 7, np.uint8)
    for bg in ("13", "14", "15"):
        assert not np.array_equal(c(img.copy(), f"01_05_{bg}_000_0100", np.random.default_rng(0)), img)


def test_mask_resized_to_image_of_same_aspect(comp_dirs):
    c = Composer(CompositeSpec(1.0, 0.0, 0.0), *comp_dirs)
    out = c(np.zeros((512, 512, 3), np.uint8), "01_05_13_000_0100", np.random.default_rng(0))
    assert out.shape == (512, 512, 3)


def test_mask_of_other_aspect_raises(comp_dirs):
    c = Composer(CompositeSpec(1.0, 0.0, 0.0), *comp_dirs)
    with pytest.raises(ValueError, match="aspect"):
        c(np.zeros((360, 640, 3), np.uint8), "01_05_13_000_0100", np.random.default_rng(0))


def test_background_scale_varies_also_on_frames():
    x = np.arange(640)
    bg = np.repeat(np.repeat((((x // 40) % 2) * 255).astype(np.uint8)[None, :, None], 480, 0), 3, 2)
    widths = set()
    for s in range(30):
        out = sample_background(bg, 360, 640, np.random.default_rng(s))
        widths.add(int(np.abs(np.diff((out[180, :, 0] > 127).astype(int))).sum()))
    assert len(widths) > 1          # il numero di strisce visibili cambia: la finestra cambia scala


def test_blur_alone_needs_no_mask_and_acts_on_whole_image():
    c = Composer(CompositeSpec(0.0, 1.0, 20.0), None, None)
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert not np.array_equal(c(img.copy(), "x", np.random.default_rng(1)), img)


def test_pasted_blur_moves_only_the_drone_layer(comp_dirs):
    c = Composer(CompositeSpec(1.0, 1.0, 20.0), *comp_dirs)
    img = np.zeros((256, 256, 3), np.uint8)
    out = c(img, "01_05_13_000_0100", np.random.default_rng(3))
    far = np.ones((256, 256), bool)
    far[70:186, 70:186] = False
    assert np.abs(out[far].astype(int) - 200).max() <= 3
    assert out[128, 128].max() < 150          # il drone (nero) resta, lo sfondo non lo copre


def test_blur_scale_shrinks_length():
    assert Composer(CompositeSpec(0, 1, 20), None, None, blur_scale=0.4).blur_max == pytest.approx(8.0)


def test_counts():
    c = Composer(CompositeSpec(0, 1, 20), None, None)
    assert c.n_masks == 0 and c.n_backgrounds == 0


def test_appledouble_files_are_ignored(comp_dirs):
    masks, bgs = comp_dirs
    (masks / "._01_05_000_0100.png").write_bytes(b"\x00")
    (bgs / "._0.jpg").write_bytes(b"\x00")
    c = Composer(CompositeSpec(1.0, 0.0, 0.0), masks, bgs)
    assert c.n_masks == 1 and c.n_backgrounds == 3
