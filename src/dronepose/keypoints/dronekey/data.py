"""Crop dataset for DroneKey: 256 px crops + YOLO labels, Ultralytics-style augmentation, 224 px normalized tensors."""
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dronepose.keypoints.dronekey.augment import apply_recipe


def read_keypoints(label: Path) -> np.ndarray:
    """(4, 2) normalized x, y from a YOLO pose label `cls cx cy w h (x y v) * 4` (class, box, visibility ignored)."""
    lines = [ln for ln in Path(label).read_text().splitlines() if ln.strip()]
    if len(lines) != 1:
        raise ValueError(f"{label}: expected exactly one label line, found {len(lines)}")
    try:
        vals = np.array(lines[0].split(), np.float64)
    except ValueError as e:
        raise ValueError(f"{label}: non-numeric value ({e})") from e
    if vals.size != 5 + 4 * 3 or not np.isfinite(vals).all():
        raise ValueError(f"{label}: expected 17 finite values (cls, box, 4 x (x y v)), found {vals.size}")
    return vals[5:].reshape(4, 3)[:, :2]


T, R = TypeVar("T"), TypeVar("R")


def _parallel_map(fn: Callable[[T], R], xs: Sequence[T], threads: int, desc: str) -> list[R]:
    """`fn` over `xs` on a thread pool (file IO releases the GIL), results in input order; the first failing item re-raises.
    Prints one line per 10 000 items and a final one with the elapsed seconds."""
    t0, out = time.perf_counter(), []
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        for i, r in enumerate(ex.map(fn, xs), 1):
            out.append(r)
            if i % 10_000 == 0 and i < len(xs):
                print(f"{desc} {i}/{len(xs)} ({time.perf_counter() - t0:.1f} s)", flush=True)
    print(f"{desc} {len(xs)} in {time.perf_counter() - t0:.1f} s", flush=True)
    return out


def preprocess_crop(img_bgr: np.ndarray, input_cfg: dict) -> torch.Tensor:
    """Eval preprocessing shared by training and inference: BGR crop -> resize (size_out) -> RGB -> /255 -> ImageNet norm, [3,H,W]."""
    size = int(input_cfg["size_out"])
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    mean, std = np.array(input_cfg["mean"], np.float32), np.array(input_cfg["std"], np.float32)
    x = (img[:, :, ::-1].astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class CropDataset(Dataset):
    def __init__(self, root, split: str, recipe: str, train: bool, input_cfg: dict, seed: int = 0,
                 limit: int | None = None, cache_ram: bool = False, allow_test: bool = False,
                 io_threads: int = 32):
        if split == "test" and not allow_test:
            raise ValueError("test split is locked: pass allow_test=True for the final evaluation only")
        self.root, self.split, self.recipe, self.train = Path(root), split, recipe, train
        self.seed, self.epoch = seed, 0
        self.input_cfg = input_cfg
        self.flip_idx = list(input_cfg["flip_idx"])
        images = sorted((self.root / "images" / split).glob("*.jpg"), key=lambda p: p.stem)
        self.items = [(p.stem, p, self.root / "labels" / split / f"{p.stem}.txt") for p in images]
        if limit is not None:
            self.items = self.items[:limit]
        # fail at construction, not in a DataLoader worker; threaded because per-file latency dominates on network disks
        self.kpts = _parallel_map(lambda it: self._read_label(it[0], it[2]), self.items, io_threads,
                                  f"CropDataset {split}: labels")
        self.cache = (_parallel_map(lambda it: self._read(it[1]), self.items, io_threads,
                                    f"CropDataset {split}: images cached") if cache_ram else None)

    @staticmethod
    def _read_label(fid: str, lab: Path) -> np.ndarray:
        if not lab.is_file():
            raise FileNotFoundError(f"missing label for {fid}: {lab}")
        return read_keypoints(lab)

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise FileNotFoundError(path)
        return img

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        fid, path, _ = self.items[i]
        img = self.cache[i].copy() if self.cache is not None else self._read(path)
        h, w = img.shape[:2]
        kp = self.kpts[i] * [w, h]
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            img, kp = apply_recipe(img, kp, self.recipe, rng, self.flip_idx)
        kp = (kp / [w, h]).astype(np.float32).reshape(-1)  # normalized to the 256 crop, out-of-crop values kept
        return preprocess_crop(img, self.input_cfg), torch.from_numpy(kp), fid
