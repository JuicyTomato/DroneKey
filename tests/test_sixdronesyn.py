import numpy as np

from dronepose.common.geometry import geodesic_deg
from dronepose.data.sixdronesyn import (SPLIT_BY_SCENE, corrected_rotation, iter_annotation_files,
                                        iter_frames, to_circular)
from tests.fixtures.fake_dataset import default_sequences


def test_annotation_files_skip_macos_junk(fake_root):
    files = iter_annotation_files(fake_root)
    assert len(files) == 4
    assert all(not f.name.startswith("._") for f in files)


def test_frames_fields_and_count(fake_root):
    frames = list(iter_frames(fake_root))
    assert len(frames) == 3 + 2 + 2 + 3           # all annotated frames, missing PNG included
    f = next(fr for fr in frames if fr.scene == "03")
    assert f.drone == "01" and f.degree == 90 and f.background == "05"
    assert f.kpts2d.shape == (4, 2) and f.kpts3d_cam.shape == (4, 3)
    assert f.image_path.name == f"{f.frame_idx:04d}.png"
    assert f.image_path.parent.name == "03_01_05_090"


def test_scene_comes_from_seq_type_not_folder(fake_root):
    scenes = {fr.scene for fr in iter_frames(fake_root)}
    assert "13" in scenes and "12" not in scenes     # folder is scene_13/scene_12 like the real dataset
    assert SPLIT_BY_SCENE["13"] == "train"


def test_split_table():
    assert {s for s, v in SPLIT_BY_SCENE.items() if v == "test"} == {"06", "07"}
    assert {s for s, v in SPLIT_BY_SCENE.items() if v == "val"} == {"03", "09", "14"}
    assert {s for s, v in SPLIT_BY_SCENE.items() if v == "train"} == {"01", "02", "04", "05", "08", "11", "12", "13"}


def test_to_circular_swaps_only_z_order_drones():
    a = np.arange(8).reshape(4, 2)
    np.testing.assert_array_equal(to_circular("01", a), a[[0, 1, 3, 2]])
    np.testing.assert_array_equal(to_circular("02", a), a)
    np.testing.assert_array_equal(to_circular("03", a), a)


def test_corrected_rotation_recovers_truth_including_drone_02(fake_root):
    truth = {(s.scene, s.drone): s.R_true for s in default_sequences()}
    for fr in iter_frames(fake_root):
        assert geodesic_deg(corrected_rotation(fr), truth[(fr.scene, fr.drone)]) < 1e-4
