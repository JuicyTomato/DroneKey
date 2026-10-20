import csv
import json

import yaml

from dronepose.common.registry import data_fingerprint, make_run_info, registry_row, update_registry


def _dataset(root):
    (root / "images" / "val").mkdir(parents=True)
    (root / "data.yaml").write_text("names: [drone]\n")
    (root / "crops_val.jsonl").write_text('{"a": 1}\n')
    (root / "images" / "val" / "a.jpg").write_bytes(b"img")
    return root


def test_fingerprint_changes_with_index_not_images(tmp_path):
    d = _dataset(tmp_path)
    h0 = data_fingerprint(d)
    (d / "images" / "val" / "a.jpg").write_bytes(b"other image")
    assert data_fingerprint(d) == h0
    (d / "crops_val.jsonl").write_text('{"a": 2}\n')
    assert data_fingerprint(d) != h0
    assert len(h0) == 64


def test_make_run_info_fields():
    args = {"name": "r", "seed": 3, "recipe": "final", "setting": "crop", "model": "yolo26s-pose.pt"}
    info = make_run_info(args, "abc123-dirty", "f" * 64, {"ultralytics": "8.4.178"}, "Tesla T4")
    assert set(info) == {"name", "created", "commit", "data_hash", "versions", "gpu", "args", "seed", "recipe", "setting"}
    assert (info["name"], info["seed"], info["recipe"], info["setting"]) == ("r", 3, "final", "crop")
    assert info["commit"] == "abc123-dirty" and info["gpu"] == "Tesla T4" and info["args"] == args
    json.dumps(info)


def _results(run, rows):
    cols = ["epoch", "time", "metrics/mAP50-95(P)"]
    with open(run / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["  " + c for c in cols])   # some Ultralytics versions pad the headers
        w.writerows(rows)


def test_registry_row_from_run_info(tmp_path):
    run = tmp_path / "r1"
    run.mkdir()
    (run / "run_info.json").write_text(json.dumps({"name": "r1", "seed": 1, "recipe": "final", "setting": "crop"}))
    _results(run, [[1, 3600, 0.2], [2, 7200, 0.5], [3, 10800, 0.4]])
    row = registry_row(run)
    assert row["name"] == "r1" and row["recipe"] == "final" and "ricostruito" not in row
    assert row["epochs_done"] == 3 and row["best_epoch"] == 2
    assert row["train_hours"] == 3.0


def test_registry_row_reconstructed(tmp_path):
    run = tmp_path / "old"
    run.mkdir()
    (run / "args.yaml").write_text(yaml.safe_dump({"name": "w1_x", "seed": 7, "model": "m.pt", "imgsz": 256}))
    _results(run, [[1, 1800, 0.1], [2, 3600, 0.3]])
    row = registry_row(run)
    assert row["ricostruito"] is True and row["name"] == "w1_x" and row["seed"] == 7
    assert row["epochs_done"] == 2 and row["best_epoch"] == 2 and row["train_hours"] == 1.0


def test_update_registry_replaces_by_name(tmp_path):
    reg = tmp_path / "sub" / "registry.jsonl"
    update_registry(reg, [{"name": "a", "v": 1}, {"name": "b", "v": 1}])
    update_registry(reg, [{"name": "a", "v": 2}])
    rows = [json.loads(line) for line in reg.read_text().splitlines()]
    assert rows == [{"name": "a", "v": 2}, {"name": "b", "v": 1}]
