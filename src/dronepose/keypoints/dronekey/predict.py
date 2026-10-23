"""DroneKey inference and keypoint export (sub-project A contract): crop -> 8 values -> frame pixels, conf 1.0.

Setting A reads the cached 256 px crops (+ crops_{split}.jsonl box); setting B crops the original frame around the
detected box of C with the existing `crop_from_detection`. Frames without a box give `no_detection()`.
"""
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from dronepose.data.canonical import read_jsonl
from dronepose.data.crops import CROP_SIZE, CropBox, crop_to_frame
from dronepose.keypoints.detections import Detection
from dronepose.keypoints.dronekey.config import load_config, resolve_variant
from dronepose.keypoints.dronekey.data import preprocess_crop
from dronepose.keypoints.dronekey.model import build_model
from dronepose.keypoints.predictors import no_detection
from dronepose.keypoints.two_stage import crop_from_detection
from dronepose.pose.replay import check_no_test

ReadImage = Callable[[Path], np.ndarray | None]


class DroneKeyPredictor:
    def __init__(self, model: torch.nn.Module, input_cfg: dict, device):
        self.model, self.input_cfg, self.device = model.eval(), input_cfg, torch.device(device)

    @torch.no_grad()
    def predict_crops(self, crops_bgr: list[np.ndarray], boxes: list[CropBox], batch: int = 64) -> list[np.ndarray]:
        """256 px BGR crops + their frame boxes -> (4, 3) arrays in frame pixels (x, y, 1.0)."""
        if len(crops_bgr) != len(boxes):
            raise ValueError(f"{len(crops_bgr)} crops but {len(boxes)} boxes")
        out = []
        for i in range(0, len(crops_bgr), batch):
            x = torch.stack([preprocess_crop(c, self.input_cfg) for c in crops_bgr[i:i + batch]]).to(self.device)
            y = self.model(x).float().cpu().numpy().reshape(-1, 4, 2) * CROP_SIZE   # normalized to the crop -> crop px
            for k, box in zip(y, boxes[i:i + batch]):
                out.append(np.hstack([crop_to_frame(k, box, CROP_SIZE), np.ones((4, 1))]))
        return out


def check_variant(ckpt_model: dict, variant_model: dict) -> None:
    """The checkpoint's own model config is authoritative: --variant may only agree with it on wiring and head."""
    for k in ("wiring", "head"):
        if ckpt_model[k] != variant_model[k]:
            raise ValueError(f"checkpoint was trained with {k}={ckpt_model[k]!r}, --variant says {variant_model[k]!r}")


def load_predictor(weights: Path, model_cfg: dict | None, device, input_cfg: dict | None = None,
                   variant: str | None = None, tiny: bool = False) -> DroneKeyPredictor:
    """Predictor from a train.py checkpoint. model_cfg defaults to the checkpoint's run_cfg["model"] (authoritative);
    `variant` (name in configs/dronekey.json) is only cross-checked against it. tiny=True: TinyBackbone (tests)."""
    ck = torch.load(Path(weights), map_location="cpu", weights_only=False)
    run_cfg = ck.get("run_cfg")
    vcfg = resolve_variant(load_config(), variant) if variant else None
    if model_cfg is None:
        model_cfg = run_cfg["model"] if run_cfg else (vcfg or {}).get("model")
        if model_cfg is None:
            raise ValueError(f"{weights}: no run_cfg in the checkpoint, pass model_cfg or variant")
    if vcfg is not None:
        check_variant(model_cfg, vcfg["model"])
    if input_cfg is None:
        input_cfg = run_cfg["input"] if run_cfg else (vcfg or load_config())["input"]
    model = build_model(model_cfg, pretrained=False, tiny=tiny)
    model.load_state_dict(ck["model"], strict=True)
    return DroneKeyPredictor(model.to(torch.device(device)), input_cfg, device)


def _row(frame_id: str, kpts: np.ndarray) -> dict:
    return {"frame_id": frame_id, "kpts_pred": np.asarray(kpts, float).tolist()}


def _imread(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def rows_setting_a(predictor, crops_root: Path, split: str, frame_ids: list[str] | None = None, batch: int = 64,
                   allow_test: bool = False, read_image: ReadImage = _imread) -> list[dict]:
    """Setting A: cached crops of `crops_root` (yolo_crops or dronekey_2dk layout); rows ordered by frame_id."""
    if split == "test" and not allow_test:
        raise ValueError("test split is locked: pass allow_test=True for the final evaluation only")
    root = Path(crops_root)
    boxes = {r["frame_id"]: CropBox(*r["box"]) for r in read_jsonl(root / f"crops_{split}.jsonl")}
    ids = sorted(boxes if frame_ids is None else frame_ids)
    if missing := [f for f in ids if f not in boxes]:
        raise KeyError(f"{len(missing)} frame_ids not in crops_{split}.jsonl, e.g. {missing[0]}")
    rows = []
    for chunk in _chunks(ids, batch):
        crops = []
        for fid in chunk:
            path = root / "images" / split / f"{fid}.jpg"
            img = read_image(path)
            if img is None:
                raise FileNotFoundError(path)
            crops.append(img)
        rows += [_row(f, k) for f, k in zip(chunk, predictor.predict_crops(crops, [boxes[f] for f in chunk], batch))]
    return rows


def rows_setting_b(predictor, records: list[dict], dets: dict[str, Detection], read_image: ReadImage = _imread,
                   batch: int = 64, allow_test: bool = False) -> list[dict]:
    """Setting B: crop the original frame (record["image_path"]) around C's box; no box -> no_detection(). Record order."""
    check_no_test(records, allow_test)
    rows = []
    for chunk in _chunks(records, batch):
        crops, boxes, idx = [], [], []
        for j, rec in enumerate(chunk):
            det = dets.get(rec["frame_id"])
            if det is None or det.box is None:
                continue
            img = read_image(Path(rec["image_path"]))
            if img is None:
                raise FileNotFoundError(rec["image_path"])
            crop, box = crop_from_detection(img, det.box)
            crops.append(crop), boxes.append(box), idx.append(j)
        pred = dict(zip(idx, predictor.predict_crops(crops, boxes, batch)))
        rows += [_row(rec["frame_id"], pred[j] if j in pred else no_detection()) for j, rec in enumerate(chunk)]
    return rows
