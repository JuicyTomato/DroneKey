"""Training of one run (plan step 3.3, Task 7): weighted ADE loss, early stopping on the selection set, resumable.

Every epoch: for each training sample one copy of the pool (the only one when clean), a random yaw rotation
(U(0, 2 pi), same rotation for inputs, bases and target), minibatches of cfg["batch"]. Selection: weighted ADE at
1 s in metres on copy 0 of the selection pool, without rotation; epoch 0 = the untrained net (a residual arm = its
base). last.pt (model, optimiser, scheduler, random states, epoch, history, normaliser) is written every epoch and
resumed when present; best.pt keeps the epoch with the lowest selection ADE. The normaliser is fitted on the training
pool only (copy 0 of the samples used; n_train = a fixed random subset).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from dronepose.forecast.nets.device import pick_device, seed_everything
from dronepose.forecast.nets.features import Batch, Normaliser
from dronepose.forecast.nets.models import TrajNet

EVAL_CHUNK = 4096


def run_name(arm, regime, seed, rnn="gru", use_mode=False, n_train=None, tag=None) -> str:
    """Directory of a run under experiments/forecast/nets/runs (tag: fold / set of the 2026-10-27 runs)."""
    return (f"{arm}_{regime}_{rnn}{'_mode' if use_mode else ''}{f'_n{n_train}' if n_train else ''}"
            f"{f'_{tag}' if tag else ''}_s{seed}")


def subset_indices(W: int, n: int, seed: int = 0) -> np.ndarray:
    """Fixed random subset of n training samples (learning curve), nested across n. Not a prefix: train_big is
    ordered by sample id and its importance weights fall along that order (common cells fill first)."""
    return np.random.default_rng(seed).permutation(W)[:n]


def weighted_ade(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    ade = torch.linalg.vector_norm(pred - target, dim=-1).mean(dim=1)
    return (weight * ade).sum() / weight.sum()


def rotate_yaw_torch(x, bases: dict, target, angles):
    """Torch twin of features.rotate_yaw (one angle per sample about e3)."""
    c, s = torch.cos(angles), torch.sin(angles)

    def rot(p):
        shape = (-1,) + (1,) * (p.dim() - 2)
        cc, ss = c.reshape(shape), s.reshape(shape)
        return torch.stack([cc * p[..., 0] - ss * p[..., 1], ss * p[..., 0] + cc * p[..., 1], p[..., 2]], dim=-1)

    x = torch.cat([rot(x[..., i:i + 3]) for i in (0, 3, 6)] + [x[..., 9:]], dim=-1)
    return x, {k: rot(v) for k, v in bases.items()}, rot(target)


def needed_bases(arm: str, use_mode: bool) -> tuple[str, ...]:
    phys = "phys_mode" if use_mode else "phys"
    return {"G": (), "R-CV": ("cv",), "R-F": (phys,), "R-F-P": (phys, "kf_ca")}[arm]


def _tensors(pool: list[Batch], norm: Normaliser, names, device, idx=None) -> dict:
    """Pool -> stacked normalised tensors on the device: x (K,W,N,9), bases (K,W,H,3), target (K,W,H,3).
    Filled copy by copy into one preallocated array per field (the noisy train pool is ~1 GB)."""
    sel = (lambda a: a) if idx is None else (lambda a: a[idx])
    K, b0 = len(pool), pool[0]

    def stack(get, fn):
        first = sel(get(b0))
        arr = np.empty((K,) + first.shape, np.float32)
        for k, b in enumerate(pool):
            arr[k] = fn(sel(get(b)))
        return torch.from_numpy(arr).to(device)

    s = norm.pos_scale
    to = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    return {"x": stack(lambda b: b.x, lambda a: (a - norm.x_mean) / norm.x_std),
            "target": stack(lambda b: b.target, lambda a: a / s),
            "bases": {k: stack(lambda b, k=k: b.bases[k], lambda a: a / s) for k in names},
            "weight": to(sel(b0.weight)), "mode_class": to(np.concatenate([sel(b0.mode_onehot), sel(b0.class_onehot)], 1)),
            "K": K, "W": len(sel(b0.weight))}


def _forward(model, t, k, idx, use_mode, angles=None):
    x, bases, target = t["x"][k, idx], {n: v[k, idx] for n, v in t["bases"].items()}, t["target"][k, idx]
    if angles is not None:
        x, bases, target = rotate_yaw_torch(x, bases, target, angles)
    out = model(x, bases, t["mode_class"][idx] if use_mode else None)
    return out, target


@torch.no_grad()
def _select_ade(model, t, use_mode, pos_scale) -> float:
    model.eval()
    num, den = 0.0, 0.0
    for s in range(0, t["W"], EVAL_CHUNK):
        idx = torch.arange(s, min(s + EVAL_CHUNK, t["W"]), device=t["x"].device)
        out, target = _forward(model, t, 0, idx, use_mode)
        ade = torch.linalg.vector_norm(out["pred"] - target, dim=-1).mean(1)
        w = t["weight"][idx]
        num += float((w * ade).sum())
        den += float(w.sum())
    model.train()
    return pos_scale * num / den


def build_model(arm, cfg, rnn="gru", use_mode=False, n_in=9) -> TrajNet:
    return TrajNet(arm, rnn=rnn, hidden=cfg["hidden"], layers=cfg["layers"], H=cfg.get("pred_len", 30),
                   use_mode=use_mode, head=tuple(cfg["head"]), gate_hidden=cfg.get("gate_hidden", 64), n_in=n_in)


def _init_state(init_from: Path, n_in: int) -> tuple[dict, Normaliser]:
    """Weights and normaliser of a run to fine-tune; a 9-channel net gets zero input weights for the mask channel
    (and mask mean 0 / std 1), so before training it predicts exactly as the source net."""
    ck = load_checkpoint(init_from)
    state = dict(ck["model"])
    for key in [k for k in state if k.startswith("encoder.weight_ih_l0")]:
        w = state[key]
        if w.shape[1] < n_in:
            state[key] = torch.cat([w, torch.zeros(w.shape[0], n_in - w.shape[1], dtype=w.dtype)], dim=1)
    norm = Normaliser.from_json(ck["normaliser"])
    extra = n_in - len(norm.x_mean)
    norm = Normaliser(np.r_[norm.x_mean, np.zeros(extra)], np.r_[norm.x_std, np.ones(extra)], norm.pos_scale)
    return state, norm


def load_checkpoint(path: Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train_run(arm: str, regime: str, seed: int, cfg_nets: dict, train_pool: list[Batch], select_pool: list[Batch],
              out_dir: Path, rnn: str = "gru", use_mode: bool = False, n_train: int | None = None, device=None,
              stop_after: int | None = None, init_from: Path | None = None, lr: float | None = None,
              n_in: int = 9) -> dict:
    """Train (or resume) one run; stop_after = epochs to run in this call (tests of the resume). init_from: fine-tune
    from a run's best.pt, with its normaliser (spec 2026-10-27 §5.4); lr overrides cfg_nets["lr"]."""
    if regime not in ("clean", "noisy"):
        raise ValueError(f"unknown regime {regime!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    t_start = time.time()
    if n_train is not None and n_train > len(train_pool[0]):
        raise ValueError(f"n_train {n_train} > {len(train_pool[0])} training samples")
    idx = None if n_train is None else subset_indices(len(train_pool[0]), n_train)
    fit_on = train_pool[0] if idx is None else train_pool[0].subset(idx)
    init_state, norm = _init_state(Path(init_from), n_in) if init_from else (None, Normaliser.fit(fit_on))
    names = needed_bases(arm, use_mode)
    tr = _tensors(train_pool, norm, names, device, idx)
    sel = _tensors(select_pool[:1], norm, names, device)

    seed_everything(seed)
    model = build_model(arm, cfg_nets, rnn, use_mode, n_in).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    lr0 = cfg_nets["lr"] if lr is None else lr
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg_nets["max_epochs"])
    rng = np.random.default_rng([seed, 3])
    meta = {"arm": arm, "regime": regime, "seed": seed, "rnn": rnn, "use_mode": use_mode, "n_train": n_train,
            "cfg": cfg_nets, "normaliser": norm.to_json(), "n_in": n_in,
            "init_from": None if init_from is None else str(init_from), "lr": lr}
    defaults = {"n_in": 9, "init_from": None, "lr": None}        # checkpoints written before these keys
    last = out / "last.pt"
    if last.exists():
        ck = load_checkpoint(last)
        if any(ck.get(k, defaults.get(k)) != meta[k] for k in ("normaliser", "arm", "regime", "seed", "rnn",
                                                                "use_mode", "n_train", "cfg", "n_in", "init_from", "lr")):
            raise ValueError(f"{last}: checkpoint of another run (arm, settings or training data differ)")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        state = ck["state"]
    else:
        state = {"history": [{"epoch": 0, "train_loss": None, "lr": lr0,
                              "select_ade_1s": _select_ade(model, sel, use_mode, norm.pos_scale), "seconds": 0.0}],
                 "best_epoch": 0, "best": None, "bad": 0, "done": False}
        state["best"] = state["history"][0]["select_ade_1s"]
        _save({**meta, "epoch": 0, "model": model.state_dict()}, out / "best.pt")
    ran = 0
    while not state["done"] and (stop_after is None or ran < stop_after):
        epoch = len(state["history"])
        t0 = time.time()
        W, K = tr["W"], tr["K"]
        copy = rng.integers(K, size=W) if K > 1 else np.zeros(W, int)
        perm = rng.permutation(W)
        angles = rng.uniform(0, 2 * np.pi, W)
        losses, wsum = 0.0, 0.0
        for s in range(0, W, cfg_nets["batch"]):
            ib = perm[s:s + cfg_nets["batch"]]
            i_t = torch.as_tensor(ib, device=device)
            k_t = torch.as_tensor(copy[ib], device=device)
            a_t = torch.as_tensor(angles[ib], dtype=torch.float32, device=device)
            o, target = _forward(model, tr, k_t, i_t, use_mode, a_t)
            loss = weighted_ade(o["pred"], target, tr["weight"][i_t])
            opt.zero_grad()
            loss.backward()
            opt.step()
            bw = tr["weight"][i_t].sum()
            losses = losses + loss.detach() * bw                 # tensors: no device sync per step
            wsum = wsum + bw
        sched.step()
        losses, wsum = float(losses), float(wsum)
        ade = _select_ade(model, sel, use_mode, norm.pos_scale)
        state["history"].append({"epoch": epoch, "train_loss": norm.pos_scale * losses / wsum,
                                 "lr": opt.param_groups[0]["lr"], "select_ade_1s": ade, "seconds": time.time() - t0})
        if ade < state["best"]:
            state.update(best=ade, best_epoch=epoch, bad=0)
            _save({**meta, "epoch": epoch, "model": model.state_dict()}, out / "best.pt")
        else:
            state["bad"] += 1
        state["done"] = state["bad"] >= cfg_nets["patience"] or epoch >= cfg_nets["max_epochs"]
        _save({**meta, "epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
               "scheduler": sched.state_dict(), "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
               "state": state}, last)
        ran += 1
    res = {**{k: meta[k] for k in ("arm", "regime", "seed", "rnn", "use_mode", "n_train", "n_in", "init_from", "lr")},
           "best_epoch": state["best_epoch"], "best_select_ade_1s": state["best"], "epochs": len(state["history"]) - 1,
           "done": state["done"], "device": str(device), "seconds_this_call": time.time() - t_start}
    (out / "history.json").write_text(json.dumps({**res, "history": state["history"]}, indent=1))
    return res
