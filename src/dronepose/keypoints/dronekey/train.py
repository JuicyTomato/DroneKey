"""Resumable DroneKey training loop (spec 5.5): checkpoint every epoch, best = min median val px error."""
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dronepose.keypoints.dronekey.loss import KeypointLoss


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def config_hash(run_cfg: dict) -> str:
    """SHA-256 of the sorted run_cfg; train.workers is excluded (it cannot change results)."""
    cfg = copy.deepcopy(run_cfg)
    cfg.get("train", {}).pop("workers", None)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def frame_px_errors(pred: torch.Tensor, target: torch.Tensor, size: int = 256) -> torch.Tensor:
    """Per frame: median over the 4 propellers of the Euclidean error in pixels of the `size` crop. [B,8] -> [B]."""
    d = (pred.float() - target.float()).reshape(-1, 4, 2) * size
    return torch.quantile(d.norm(dim=2), 0.5, dim=1)  # interpolated median (mean of the middle two)


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")


@torch.no_grad()
def evaluate(model, loader, loss, device, amp: bool = False, size: int = 256) -> dict:
    """Val loss (at the current S(t)), median/p90 of the per-frame px error, mean gate per layer (None without gate)."""
    model.eval()
    tot, n, errs, gate_sum = 0.0, 0, [], None
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with _autocast(device, amp):
            pred, aux = model.forward_aux(x)
        tot += loss(pred, y).item() * x.shape[0]
        n += x.shape[0]
        errs.append(frame_px_errors(pred, y, size).cpu())
        if "gate" in aux:
            g = aux["gate"].float().sum(dim=0).cpu()
            gate_sum = g if gate_sum is None else gate_sum + g
    e = torch.cat(errs).numpy()
    return {"loss": tot / n, "px_median": float(np.median(e)), "px_p90": float(np.percentile(e, 90)),
            "gate_mean": None if gate_sum is None else (gate_sum / n).tolist()}


def _rng_state(device: torch.device) -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":  # only the training GPU: the device count may differ between sessions
        st["cuda"] = torch.cuda.get_rng_state(device)
    return st


def _set_rng_state(st: dict, device: torch.device) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["cuda"], device)


def _save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # a crash mid-write never corrupts the previous checkpoint


def fit(run_cfg: dict, model, train_ds, val_ds, out_dir: Path, device: torch.device,
        resume_from: Path | None = None, max_hours: float | None = None, save_checkpoints: bool = True) -> list[dict]:
    """Train from run_cfg (resolved variant + seed/name overrides); returns the history rows.

    max_hours is a per-session budget: wall time measured from the start of this fit() call (not summed over resumes).
    save_checkpoints=False writes only history.json (no last.pt / best.pt; not resumable).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tc, seed, size = run_cfg["train"], run_cfg["seed"], run_cfg["input"]["size_in"]
    epochs, chash = tc["epochs"], config_hash(run_cfg)
    use_amp = device.type == "cuda" and bool(tc.get("amp"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # no cudnn.deterministic: bit-exact resume only on CPU (spec 5.5)
    lc = run_cfg["loss"]
    loss = KeypointLoss(lc["kind"], lc["alpha"], lc["D"], lc["eps"], epochs=epochs)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, best_px, start = [], float("inf"), 0
    if resume_from is not None:
        ck = torch.load(resume_from, map_location="cpu", weights_only=False)
        if ck["config_hash"] != chash:
            raise ValueError(f"{resume_from}: run config differs from the checkpoint's (config hash mismatch)")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if ck["scaler"]:  # empty when the checkpoint came from a disabled scaler (CPU/MPS)
            scaler.load_state_dict(ck["scaler"])
        _set_rng_state(ck["rng"], device)
        history, best_px, start = ck["history"], ck["best_px"], ck["epoch"]
    else:
        seed_everything(seed)
    pin = device.type == "cuda"
    val_loader = DataLoader(val_ds, batch_size=tc["batch"], shuffle=False, num_workers=tc["workers"], pin_memory=pin)
    t_start = time.time()
    for e in range(start, epochs):
        t0 = time.time()
        loss.set_epoch(e)
        train_ds.set_epoch(e)
        loader = DataLoader(train_ds, batch_size=tc["batch"], shuffle=True, num_workers=tc["workers"],
                            generator=torch.Generator().manual_seed(seed * 100003 + e), drop_last=False, pin_memory=pin)
        model.train()
        tot, n = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                pred = model(x)
            l = loss(pred, y)  # float32, outside autocast
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            tot += l.item() * x.shape[0]
            n += x.shape[0]
        ev = evaluate(model, val_loader, loss, device, amp=use_amp, size=size)
        row = {"epoch": e + 1, "train_loss": tot / n, "val_loss": ev["loss"], "val_px_median": ev["px_median"],
               "val_px_p90": ev["px_p90"], "scale": loss.scale, "gate_mean": ev["gate_mean"],
               "seconds": time.time() - t0}
        history.append(row)
        improved = ev["px_median"] < best_px
        if improved:
            best_px = ev["px_median"]
        if improved and save_checkpoints:
            _save({"model": model.state_dict(), "run_cfg": run_cfg, "epoch": e + 1, "val_px_median": best_px},
                  out_dir / "best.pt")
        if save_checkpoints:
            _save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                   "rng": _rng_state(device), "epoch": e + 1, "history": history, "best_px": best_px,
                   "config_hash": chash, "run_cfg": run_cfg}, out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
        print(f"epoch {e + 1}/{epochs} train {row['train_loss']:.4f} val {row['val_loss']:.4f} "
              f"px med {row['val_px_median']:.2f} p90 {row['val_px_p90']:.2f} S {row['scale']:.3f} "
              f"{row['seconds']:.0f}s{' *' if improved else ''}", flush=True)
        if max_hours is not None and (time.time() - t_start) > max_hours * 3600 and e + 1 < epochs:
            print(f"max_hours {max_hours} exceeded: stopping after epoch {e + 1} (resume with --resume-from)", flush=True)
            break
    return history
