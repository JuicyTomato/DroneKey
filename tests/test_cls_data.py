import json

from dronepose.keypoints.cls_data import build_imagefolder, drone_of_frame


def test_drone_of_frame():
    assert drone_of_frame("01_04_01_000_0001") == "04"


def test_build_imagefolder_links_by_drone(tmp_path):
    crops = tmp_path / "yolo_crops"
    ids = {"train": ["01_04_01_000_0001", "01_04_01_000_0002", "02_08_03_000_0001"], "val": ["03_01_05_180_0001"]}
    for split, fids in ids.items():
        (crops / "images" / split).mkdir(parents=True)
        with open(crops / f"crops_{split}.jsonl", "w") as f:
            for fid in fids:
                (crops / "images" / split / f"{fid}.jpg").write_bytes(b"jpg")
                f.write(json.dumps({"frame_id": fid, "box": [0, 0, 1]}) + "\n")
    out = tmp_path / "cls"
    assert build_imagefolder(crops, out) == {"train": 3, "val": 1}
    link = out / "train" / "04" / "01_04_01_000_0002.jpg"
    assert link.is_symlink() and link.resolve() == (crops / "images" / "train" / "01_04_01_000_0002.jpg").resolve()
    assert sorted(p.name for p in (out / "train").iterdir()) == ["04", "08"]
    assert (out / "val" / "01" / "03_01_05_180_0001.jpg").is_symlink()


def test_build_imagefolder_limit(tmp_path):
    crops = tmp_path / "yolo_crops"
    (crops / "images" / "train").mkdir(parents=True)
    with open(crops / "crops_train.jsonl", "w") as f:
        for i in range(5):
            fid = f"01_04_01_000_{i:04d}"
            (crops / "images" / "train" / f"{fid}.jpg").write_bytes(b"jpg")
            f.write(json.dumps({"frame_id": fid}) + "\n")
    assert build_imagefolder(crops, tmp_path / "cls", splits=("train",), limit=2) == {"train": 2}
