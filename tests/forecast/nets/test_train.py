import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dronepose.forecast.nets.features import BASES, Batch, Normaliser, rotate_yaw
from dronepose.forecast.nets.train import (load_checkpoint, rotate_yaw_torch, subset_indices, train_run,
                                          weighted_ade)

CFG = {"hidden": 16, "layers": 1, "head": [32], "gate_hidden": 8, "lr": 3e-3, "batch": 16, "max_epochs": 30,
       "patience": 100, "pred_len": 30}
CPU = torch.device("cpu")


def _pool(W=64, K=1, seed=0, noise=0.0):
    """Synthetic pool: constant-velocity futures; 'phys' is off by 0.6 x velocity (learnable from x)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(W, 3)).astype(np.float32)
    t_obs = (np.arange(15) - 14)[None, :, None] / 30
    t_fut = np.arange(1, 31)[None, :, None] / 30
    pool = []
    for k in range(K):
        jitter = rng.normal(scale=0.01, size=(W, 15, 3)).astype(np.float32) * (k + 1)
        pos = v[:, None] * t_obs + jitter
        x = np.concatenate([pos, np.broadcast_to(v[:, None], (W, 15, 3)), np.tile([0, 0, 1], (W, 15, 1))], -1)
        target = v[:, None] * t_fut + noise * rng.normal(size=(W, 30, 3))
        phys = target + 0.6 * v[:, None] * np.ones((1, 30, 1))
        bases = {b: phys.astype(np.float32) for b in BASES}
        pool.append(Batch(x=x.astype(np.float32), bases=bases, target=target.astype(np.float32),
                          weight=rng.uniform(0.5, 2, W).astype(np.float32), mode_onehot=np.zeros((W, 3), np.float32),
                          class_onehot=np.zeros((W, 7), np.float32), traj=np.arange(W), metric=np.ones(W, bool)))
    return pool


def test_weighted_ade():
    pred = torch.zeros(2, 3, 3)
    target = torch.zeros(2, 3, 3)
    target[0, :, 0] = 1.0                     # ADE 1
    target[1, :, 1] = torch.tensor([0.0, 3.0, 6.0])  # ADE 3
    w = torch.tensor([1.0, 3.0])
    assert weighted_ade(pred, target, w).item() == pytest.approx((1 * 1 + 3 * 3) / 4)


def test_torch_rotation_matches_numpy():
    b = _pool(W=8)[0]
    ang = np.linspace(0, 6, 8)
    ref = rotate_yaw(b, ang)
    x, bases, target = rotate_yaw_torch(torch.tensor(b.x), {k: torch.tensor(v) for k, v in b.bases.items()},
                                        torch.tensor(b.target), torch.tensor(ang, dtype=torch.float32))
    np.testing.assert_allclose(x.numpy(), ref.x, atol=1e-5)
    np.testing.assert_allclose(target.numpy(), ref.target, atol=1e-5)
    np.testing.assert_allclose(bases["phys"].numpy(), ref.bases["phys"], atol=1e-5)


def test_loss_decreases_on_tiny_set(tmp_path):
    pool = _pool()
    res = train_run("R-F", "clean", 0, CFG, pool, pool, tmp_path, device=CPU)
    hist = json.loads((tmp_path / "history.json").read_text())["history"]
    assert hist[0]["epoch"] == 0 and hist[0]["train_loss"] is None
    assert res["best_select_ade_1s"] < 0.5 * hist[0]["select_ade_1s"]


def test_epoch0_residual_equals_base(tmp_path):
    pool = _pool(W=32)
    train_run("R-F", "clean", 0, dict(CFG, max_epochs=1), pool, pool, tmp_path, device=CPU)
    hist = json.loads((tmp_path / "history.json").read_text())["history"]
    b = pool[0]
    ade = np.linalg.norm(b.bases["phys"] - b.target, axis=-1).mean(1)
    assert hist[0]["select_ade_1s"] == pytest.approx(float(np.average(ade, weights=b.weight)), rel=1e-4)


def test_early_stopping_and_best_checkpoint(tmp_path):
    train = _pool(W=64)
    select = _pool(W=64, seed=5, noise=3.0)               # nothing to learn on the selection set
    res = train_run("R-F", "clean", 0, dict(CFG, patience=2, max_epochs=60, lr=3e-2), train, select, tmp_path,
                    device=CPU)
    hist = json.loads((tmp_path / "history.json").read_text())["history"]
    sel = [h["select_ade_1s"] for h in hist]
    best = int(np.argmin(sel))
    assert res["done"] and res["epochs"] < 60
    assert res["best_epoch"] == best and len(hist) - 1 == best + 2
    ck = load_checkpoint(tmp_path / "best.pt")
    assert ck["epoch"] == best


def test_resume_continues_history(tmp_path):
    pool = _pool(W=48, K=3)
    cfg = dict(CFG, max_epochs=5)
    train_run("R-F", "noisy", 1, cfg, pool, pool, tmp_path / "a", device=CPU)
    r1 = train_run("R-F", "noisy", 1, cfg, pool, pool, tmp_path / "b", device=CPU, stop_after=2)
    assert not r1["done"] and r1["epochs"] == 2
    train_run("R-F", "noisy", 1, cfg, pool, pool, tmp_path / "b", device=CPU)
    ha = json.loads((tmp_path / "a" / "history.json").read_text())["history"]
    hb = json.loads((tmp_path / "b" / "history.json").read_text())["history"]
    assert [h["epoch"] for h in hb] == list(range(6))
    for x, y in zip(ha, hb):
        assert x["select_ade_1s"] == pytest.approx(y["select_ade_1s"], rel=1e-6)
    wa, wb = (load_checkpoint(tmp_path / d / "last.pt")["model"] for d in ("a", "b"))
    for k in wa:
        torch.testing.assert_close(wa[k], wb[k])


def test_checkpoint_contains_normaliser(tmp_path):
    pool = _pool(W=32)
    train_run("G", "clean", 0, dict(CFG, max_epochs=2), pool, pool, tmp_path, device=CPU)
    expected = Normaliser.fit(pool[0]).to_json()
    for name in ("best.pt", "last.pt"):
        ck = load_checkpoint(tmp_path / name)
        assert ck["normaliser"] == expected
        assert ck["arm"] == "G" and ck["regime"] == "clean" and ck["rnn"] == "gru" and ck["use_mode"] is False


def test_n_train_uses_fixed_random_subset(tmp_path):
    # train_big is ordered by sample id and the importance weights fall along that order: a prefix would be biased
    pool = _pool(W=40)
    train_run("R-F", "clean", 0, dict(CFG, max_epochs=1), pool, pool, tmp_path, device=CPU, n_train=10)
    ck = load_checkpoint(tmp_path / "last.pt")
    idx = subset_indices(40, 10)
    assert ck["n_train"] == 10 and ck["normaliser"] == Normaliser.fit(pool[0].subset(idx)).to_json()
    assert not np.array_equal(np.sort(idx), np.arange(10)) and len(set(idx)) == 10
    np.testing.assert_array_equal(idx, subset_indices(40, 10))                      # fixed, whatever the seed of the run
    assert set(subset_indices(40, 10)) <= set(subset_indices(40, 20))              # nested: 2500 inside 10000


def test_resume_refuses_other_config(tmp_path):
    pool = _pool(W=16)
    train_run("R-F", "clean", 0, dict(CFG, max_epochs=3), pool, pool, tmp_path, device=CPU, stop_after=1)
    with pytest.raises(ValueError):
        train_run("R-F", "clean", 0, dict(CFG, max_epochs=100), pool, pool, tmp_path, device=CPU)


def test_n_train_larger_than_pool_refused(tmp_path):
    with pytest.raises(ValueError):
        train_run("R-F", "clean", 0, CFG, _pool(W=16), _pool(W=16), tmp_path, device=CPU, n_train=32)
