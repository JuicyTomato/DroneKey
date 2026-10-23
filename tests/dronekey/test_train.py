import copy
import json
import sys

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from dronepose.keypoints.dronekey import train as T  # noqa: E402
from dronepose.keypoints.dronekey.config import load_config, resolve_variant  # noqa: E402
from dronepose.keypoints.dronekey.data import CropDataset  # noqa: E402
from dronepose.keypoints.dronekey.model import build_model  # noqa: E402

CFG = load_config()
KP = [(0.5, 0.2), (0.8, 0.5), (0.5, 0.8), (0.3, 0.5)]


def _crops(root, n=8):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        rng = np.random.default_rng(1)
        for i in range(n):
            cv2.imwrite(str(root / "images" / split / f"f_{i:04d}.jpg"), rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
            vals = " ".join(f"{x + 0.01 * i} {y} 2" for x, y in KP)
            (root / "labels" / split / f"f_{i:04d}.txt").write_text(f"0 0.5 0.5 0.9 0.9 {vals}\n")
    return root


def _run_cfg(epochs=2, lr=1e-3):
    cfg = resolve_variant(CFG, "G4")
    cfg["model"].update(d=16, n_layers=2, nhead=2, ffn=32)
    cfg["train"].update(epochs=epochs, lr=lr, batch=4, workers=0, amp=False)
    cfg.update(seed=0, name="t", variant="G4", dataset="6dronesyn", recipe="none")
    return cfg


def _setup(tmp_path, epochs=2, lr=1e-3):
    cfg = _run_cfg(epochs, lr)
    root = tmp_path / "crops" if (tmp_path / "crops").exists() else _crops(tmp_path / "crops")
    train = CropDataset(root, "train", "none", True, cfg["input"], seed=0)
    val = CropDataset(root, "val", "none", False, cfg["input"])
    torch.manual_seed(0)
    return cfg, build_model(cfg["model"], pretrained=False, tiny=True), train, val


def test_frame_px_errors_known():
    t = torch.rand(3, 8)
    e = T.frame_px_errors(t + torch.tensor([0.01, 0.0] * 4), t, size=256)
    torch.testing.assert_close(e, torch.full((3,), 2.56))


def test_fit_writes_history_last_best(tmp_path):
    cfg, model, train, val = _setup(tmp_path)
    out = tmp_path / "run"
    hist = T.fit(cfg, model, train, val, out, torch.device("cpu"))
    assert len(hist) == 2
    on_disk = json.loads((out / "history.json").read_text())
    assert [r["epoch"] for r in on_disk] == [1, 2]
    assert set(on_disk[0]) == {"epoch", "train_loss", "val_loss", "val_px_median", "val_px_p90", "scale", "gate_mean", "seconds"}
    assert len(on_disk[0]["gate_mean"]) == 2 and on_disk[0]["scale"] > on_disk[1]["scale"]
    assert (out / "last.pt").exists() and (out / "best.pt").exists()
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 2 and len(last["history"]) == 2 and len(last["config_hash"]) == 64
    assert {"model", "optimizer", "scaler", "rng", "best_px", "run_cfg"} <= set(last)


def test_resume_bit_identical_cpu(tmp_path):
    cfg, m1, tr, va = _setup(tmp_path)
    T.fit(cfg, m1, tr, va, tmp_path / "a", torch.device("cpu"))
    cfg, m2, tr, va = _setup(tmp_path)
    T.fit(cfg, m2, tr, va, tmp_path / "b", torch.device("cpu"), max_hours=1e-9)
    assert torch.load(tmp_path / "b" / "last.pt", map_location="cpu", weights_only=False)["epoch"] == 1
    cfg, m3, tr, va = _setup(tmp_path)
    hist = T.fit(cfg, m3, tr, va, tmp_path / "b", torch.device("cpu"), resume_from=tmp_path / "b" / "last.pt")
    assert len(hist) == 2
    sa, sc = m1.state_dict(), m3.state_dict()
    assert all(torch.equal(sa[k], sc[k]) for k in sa)
    h1 = json.loads((tmp_path / "a" / "history.json").read_text())
    for r1, r3 in zip(h1, hist):
        assert (r1["train_loss"], r1["val_px_median"]) == (r3["train_loss"], r3["val_px_median"])


def test_resume_rejects_config_change(tmp_path):
    cfg, model, tr, va = _setup(tmp_path)
    T.fit(cfg, model, tr, va, tmp_path / "r", torch.device("cpu"), max_hours=1e-9)
    cfg, model, tr, va = _setup(tmp_path, lr=5e-4)
    with pytest.raises(ValueError, match="config"):
        T.fit(cfg, model, tr, va, tmp_path / "r", torch.device("cpu"), resume_from=tmp_path / "r" / "last.pt")


def test_max_hours_stops_after_epoch(tmp_path):
    cfg, model, tr, va = _setup(tmp_path, epochs=3)
    hist = T.fit(cfg, model, tr, va, tmp_path / "m", torch.device("cpu"), max_hours=1e-9)
    assert len(hist) == 1 and (tmp_path / "m" / "last.pt").exists()


def test_best_selected_by_px_not_loss(tmp_path, monkeypatch):
    vals = iter([5.0, 3.0, 4.0])  # best px at epoch 2; the fake val loss (10 - px) is lowest at epoch 1

    def fake_eval(model, loader, loss, device, **kw):
        px = next(vals)
        return {"loss": 10.0 - px, "px_median": px, "px_p90": px + 1, "gate_mean": None}

    monkeypatch.setattr(T, "evaluate", fake_eval)
    cfg, model, tr, va = _setup(tmp_path, epochs=3)
    T.fit(cfg, model, tr, va, tmp_path / "bp", torch.device("cpu"))
    best = torch.load(tmp_path / "bp" / "best.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == 2 and best["val_px_median"] == 3.0
    assert "model" in best and "run_cfg" in best


def test_resume_ignores_workers_change(tmp_path):
    cfg, model, tr, va = _setup(tmp_path)
    T.fit(cfg, model, tr, va, tmp_path / "w", torch.device("cpu"), max_hours=1e-9)
    cfg, model, tr, va = _setup(tmp_path)
    cfg["train"]["workers"] = 2
    hist = T.fit(cfg, model, tr, va, tmp_path / "w", torch.device("cpu"), resume_from=tmp_path / "w" / "last.pt")
    assert len(hist) == 2


def test_fit_without_checkpoints_writes_only_history(tmp_path):
    cfg, model, train, val = _setup(tmp_path)
    out = tmp_path / "run"
    hist = T.fit(cfg, model, train, val, out, torch.device("cpu"), save_checkpoints=False)
    assert len(hist) == 2
    assert (out / "history.json").exists()
    assert not list(out.glob("*.pt")) and not list(out.glob("*.tmp*"))


def test_train_script_requires_albumentations_for_w1_final(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "kaggle" / "train_dronekey.py"
    spec = importlib.util.spec_from_file_location("train_dronekey_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, "albumentations", None)  # `import albumentations` -> ImportError
    for recipe in ("w1", "final"):
        with pytest.raises(SystemExit, match="albumentations mancante"):
            mod.require_albumentations(recipe)
    mod.require_albumentations("none")  # unaffected


def _load_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "kaggle" / "train_dronekey.py"
    spec = importlib.util.spec_from_file_location("train_dronekey_script_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ORIG_BUILD_RUN_CFG: dict = {}


def _run_script(mod, monkeypatch, root, project, *extra):
    """Run train_dronekey.main() on tiny crops with a tiny model (CPU)."""
    import dronepose.keypoints.dronekey.model as M

    orig_build = build_model  # module-level original (M.build_model may already be patched by a previous call)

    def small_cfg(args, name):
        cfg = _ORIG_BUILD_RUN_CFG[mod](args, name)
        cfg["model"].update(d=16, n_layers=2, nhead=2, ffn=32)
        return cfg

    _ORIG_BUILD_RUN_CFG.setdefault(mod, mod.build_run_cfg)
    monkeypatch.setattr(mod, "build_run_cfg", small_cfg)
    monkeypatch.setattr(M, "build_model", lambda m, pretrained=True: orig_build(m, pretrained=False, tiny=True))
    argv = ["train_dronekey.py", "--dataset", "6dronesyn", "--variant", "G4", "--recipe", "none", "--epochs", "2",
            "--batch", "4", "--workers", "0", "--device", "cpu", "--data-root", str(root), "--project", str(project), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()


def test_script_resume_from_readonly_folder(tmp_path, monkeypatch):
    import os
    import shutil

    root = _crops(tmp_path / "crops")
    (root / "data.yaml").write_text("names: [drone]\n")
    mod = _load_script()
    first = tmp_path / "first"
    _run_script(mod, monkeypatch, root, first, "--name", "dk_t", "--max-hours", "0")  # stops after epoch 1
    src = tmp_path / "input" / "ds"  # simulates /kaggle/input/<dataset>
    src.mkdir(parents=True)
    for f in ("last.pt", "best.pt", "run_info.json", "history.json"):
        shutil.copyfile(first / "dk_t" / f, src / f)
    assert len(json.loads((src / "history.json").read_text())) == 1
    before = {p.name: p.stat().st_mtime_ns for p in src.iterdir()}
    project = tmp_path / "working"
    os.chmod(src, 0o555)
    try:
        _run_script(mod, monkeypatch, root, project, "--resume-from", str(src / "last.pt"))  # no --name
        assert sorted(p.name for p in src.iterdir()) == sorted(before)
        assert {p.name: p.stat().st_mtime_ns for p in src.iterdir()} == before
    finally:
        os.chmod(src, 0o755)
    out = project / "dk_t"
    assert [r["epoch"] for r in json.loads((out / "history.json").read_text())] == [1, 2]
    assert (out / "last.pt").exists() and (out / "best.pt").exists()
    assert torch.load(out / "last.pt", map_location="cpu", weights_only=False)["epoch"] == 2
    info = json.loads((out / "run_info.json").read_text())
    assert len(info["resumed"]) == 1 and info["name"] == "dk_t" and info["epochs_done"] == 2


def test_script_resume_with_different_name_errors(tmp_path, monkeypatch):
    root = _crops(tmp_path / "crops")
    (root / "data.yaml").write_text("names: [drone]\n")
    mod = _load_script()
    _run_script(mod, monkeypatch, root, tmp_path / "first", "--name", "dk_t", "--max-hours", "0")
    project = tmp_path / "working"
    with pytest.raises(SystemExit, match="differs from the checkpoint"):
        _run_script(mod, monkeypatch, root, project, "--name", "other", "--resume-from", str(tmp_path / "first" / "dk_t" / "last.pt"))
    assert not project.exists()


def test_script_resume_config_mismatch_leaves_no_trace(tmp_path, monkeypatch):
    root = _crops(tmp_path / "crops")
    (root / "data.yaml").write_text("names: [drone]\n")
    mod = _load_script()
    _run_script(mod, monkeypatch, root, tmp_path / "first", "--name", "dk_t", "--max-hours", "0")
    project = tmp_path / "working"
    with pytest.raises(SystemExit, match="config hash mismatch"):
        _run_script(mod, monkeypatch, root, project, "--resume-from", str(tmp_path / "first" / "dk_t" / "last.pt"), "--seed", "5")
    assert not project.exists()
