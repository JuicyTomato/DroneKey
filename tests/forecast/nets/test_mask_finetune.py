import dataclasses
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dronepose.forecast.archive_noise import load_tracts
from dronepose.forecast.nets.evaluate import predict_run
from dronepose.forecast.nets.features import (Batch, Normaliser, archive_copy, build_archive_pool, build_batch,
                                              fill_lost, rotate_yaw)
from dronepose.forecast.nets.train import load_checkpoint, rotate_yaw_torch, run_name, train_run
from dronepose.forecast.windows import make_windows
from tests.forecast.nets.test_features import CFG, PARAMS, _gen_trajs
from tests.forecast.nets.test_train import CFG as NET_CFG, CPU, _pool
from tests.forecast.test_archive_noise import _fake_archive


def _w():
    return make_windows(_gen_trajs(), CFG.obs_len, CFG.pred_len, CFG.fps)


def _with_mask(w, lost):
    v = np.ones(w.obs_p.shape[:2], bool)
    v[:, lost] = False
    return dataclasses.replace(w, obs_valid=v)


def test_build_batch_default_unchanged():
    w = _w()
    a = build_batch(w, PARAMS, CFG)
    b = build_batch(dataclasses.replace(w, obs_valid=np.ones(w.obs_p.shape[:2], bool)), PARAMS, CFG)
    assert a.x.shape[-1] == 9
    assert np.array_equal(a.x, b.x) and np.array_equal(a.target, b.target)
    assert all(np.array_equal(a.bases[k], b.bases[k]) for k in a.bases)


def test_mask_channel_all_ones_without_mask():
    b = build_batch(_w(), PARAMS, CFG, mask_channel=True)
    assert b.x.shape[-1] == 10 and (b.x[..., 9] == 1).all()
    np.testing.assert_array_equal(b.x[..., :9], build_batch(_w(), PARAMS, CFG).x)


def test_fill_lost():
    v = np.array([[False, True, False, True, False]])
    np.testing.assert_array_equal(fill_lost(v), [[1, 1, 1, 3, 3]])


def test_anchor_is_last_valid():
    w = _w()
    b = build_batch(_with_mask(w, [-1]), PARAMS, CFG, mask_channel=True)
    np.testing.assert_allclose(b.x[:, -1, :3], 0, atol=1e-7)
    np.testing.assert_allclose(b.x[:, -2, :3], 0, atol=1e-7)
    np.testing.assert_allclose(b.target, w.fut_p - w.obs_p[:, -2:-1], atol=1e-6)
    np.testing.assert_array_equal(b.x[:, -1, 9], 0)
    np.testing.assert_array_equal(b.x[:, -2, 9], 1)


def test_lost_frames_do_not_leak():
    w = _with_mask(_w(), [3, -1])
    dirty = dataclasses.replace(w, obs_p=w.obs_p + np.where(w.obs_valid[..., None], 0, 10.0))
    a, b = build_batch(w, PARAMS, CFG, mask_channel=True), build_batch(dirty, PARAMS, CFG, mask_channel=True)
    np.testing.assert_allclose(a.x, b.x, atol=1e-6)
    for k in a.bases:
        np.testing.assert_allclose(a.bases[k], b.bases[k], atol=1e-6)


def test_mask_channel_not_rotated():
    b = build_batch(_with_mask(_w(), [2]), PARAMS, CFG, mask_channel=True)
    np.testing.assert_array_equal(rotate_yaw(b, np.full(len(b), 1.0)).x[..., 9], b.x[..., 9])
    x = torch.as_tensor(b.x)
    xr, _, _ = rotate_yaw_torch(x, {}, torch.zeros(len(b), 30, 3), torch.full((len(b),), 1.0))
    assert torch.equal(xr[..., 9], x[..., 9])


def test_normaliser_mask_untouched():
    n = Normaliser.fit(build_batch(_with_mask(_w(), [2]), PARAMS, CFG, mask_channel=True))
    assert len(n.x_mean) == 10 and n.x_mean[9] == 0 and n.x_std[9] == 1


def test_archive_copy_reproducible_and_masked(tmp_path):
    tr = load_tracts([_fake_archive(tmp_path, n=20)])
    tr = dataclasses.replace(tr, dp=np.random.default_rng(0).normal(scale=0.05, size=tr.dp.shape))  # varies per frame
    w = _w()
    a = archive_copy(w, PARAMS, CFG, tr, 0, 3, distance_cond=False)
    b = archive_copy(w, PARAMS, CFG, tr, 0, 3, distance_cond=False)
    c = archive_copy(w, PARAMS, CFG, tr, 0, 4, distance_cond=False)
    assert np.array_equal(a.x, b.x) and not np.array_equal(a.x, c.x) and a.x.shape[-1] == 10
    pool = build_archive_pool(w, PARAMS, CFG, tr, K=2, seed=0, distance_cond=True)
    assert len(pool) == 2 and all(len(p) == len(w) for p in pool)


def test_archive_copy_skips_tracts_with_too_few_valid(tmp_path):
    good = _fake_archive(tmp_path, n=15, seed=0)
    bad = _fake_archive(tmp_path, n=15, seed=1, valid=False)
    b = archive_copy(_w(), PARAMS, CFG, load_tracts([good, bad]), 0, 0, distance_cond=False)
    assert (b.x[..., 9] == 1).all()


def _pool10(**kw):
    pool = _pool(**kw)
    return [dataclasses.replace(b, x=np.concatenate([b.x, np.ones(b.x.shape[:2] + (1,), np.float32)], -1))
            for b in pool]


def test_finetune_zero_mask_weights(tmp_path):
    cfg = {**NET_CFG, "max_epochs": 1}
    src = tmp_path / "src"
    train_run("R-CV", "clean", 0, cfg, _pool(), _pool(seed=1), src, device=CPU)
    ck9 = load_checkpoint(src / "best.pt")
    dst = tmp_path / "ft"
    train_run("R-CV", "clean", 0, cfg, _pool10(), _pool10(seed=1), dst, device=CPU, init_from=src / "best.pt",
              lr=1e-4, n_in=10, stop_after=0)                     # epoch 0 only: the initialised model
    ck10 = load_checkpoint(dst / "best.pt")
    w_ih = ck10["model"]["encoder.weight_ih_l0"]
    assert w_ih.shape[1] == 10 and torch.all(w_ih[:, 9] == 0)
    assert torch.equal(w_ih[:, :9], ck9["model"]["encoder.weight_ih_l0"])
    assert ck10["normaliser"]["x_mean"][:9] == ck9["normaliser"]["x_mean"]
    b9, b10 = _pool(seed=2)[0], _pool10(seed=2)[0]
    np.testing.assert_allclose(predict_run(dst, b10, CPU), predict_run(src, b9, CPU), atol=1e-5)
    meta = json.loads((dst / "history.json").read_text())
    assert meta["n_in"] == 10


def test_resume_refuses_other_init(tmp_path):
    cfg = {**NET_CFG, "max_epochs": 1}
    train_run("G", "clean", 0, cfg, _pool(), _pool(seed=1), tmp_path / "a", device=CPU)
    train_run("G", "clean", 0, {**cfg, "max_epochs": 2}, _pool10(), _pool10(seed=1), tmp_path / "b", device=CPU,
              n_in=10, stop_after=1)
    with pytest.raises(ValueError):
        train_run("G", "clean", 0, {**cfg, "max_epochs": 2}, _pool10(), _pool10(seed=1), tmp_path / "b", device=CPU,
                  n_in=10, lr=1e-4)


def test_run_name_tag():
    assert run_name("R-F", "noisy", 1, tag="fold03") == "R-F_noisy_gru_fold03_s1"
    assert run_name("R-F", "noisy", 1) == "R-F_noisy_gru_s1"
