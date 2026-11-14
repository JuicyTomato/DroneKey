"""Per-frame perception of the integrated pipeline (spec 2026-11-14 §4): reduced frame -> C -> crop -> DroneKey + K
-> reprojection error of every drone model. Identical in offline and streaming mode, so it can be cached.

The three model calls are injectable callables (tests use fakes); `PerceptionModels.load` builds the real ones.
"""
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dronepose.data.crops import CROP_SIZE, CropBox
from dronepose.keypoints.class_scores import reproj_per_class
from dronepose.keypoints.predictors import no_detection
from dronepose.keypoints.two_stage import crop_from_detection
from dronepose.pipeline.frames import reduce_frame
from dronepose.pipeline.sources import Frame, SceneInfo

MS_KEYS = ("reduce", "detect", "crop", "keypoints", "classify", "geometry")
TOP_CONF = 0.25          # Ultralytics predict default, threshold of C's exported top box


@dataclass
class Perception:
    frame_id: str
    frame_idx: int
    box: list[float] | None
    box_conf: float
    kpts: np.ndarray                      # (4, 3) frame pixels; NaN without a box
    probs: dict[str, float] | None
    reproj: dict[str, float] | None
    ms: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict:
        k = [[None if not math.isfinite(v) else float(v) for v in row] for row in np.asarray(self.kpts, float)]
        return {"frame_id": self.frame_id, "frame_idx": self.frame_idx, "box": self.box, "box_conf": self.box_conf,
                "kpts": k, "probs": self.probs, "reproj": self.reproj, "ms": self.ms}

    @classmethod
    def from_row(cls, row: dict) -> "Perception":
        k = np.array([[np.nan if v is None else v for v in r] for r in row["kpts"]], float)
        return cls(row["frame_id"], int(row["frame_idx"]), row["box"], float(row["box_conf"]), k, row["probs"],
                   row["reproj"], dict(row.get("ms") or {}))


def _sync(device: str | None) -> None:
    if device == "mps":
        import torch
        torch.mps.synchronize()


class PerceptionModels:
    def __init__(self, detect, keypoints, classify, device: str | None = None):
        """detect(img_small) -> (xyxy, conf, cls, kxy, kconf) in small-image pixels; keypoints(crop, CropBox) ->
        (4, 3) frame pixels; classify(crop) -> (probs, names)."""
        self.detect, self.keypoints, self.classify, self.device = detect, keypoints, classify, device

    @classmethod
    def load(cls, cfg: dict, device: str) -> "PerceptionModels":
        from ultralytics import YOLO
        from dronepose.keypoints.dronekey.predict import load_predictor
        det = YOLO(cfg["detector"])
        clf = YOLO(cfg["classifier"])
        dk = load_predictor(Path(cfg["keypoints"]), None, device)

        def detect(img):
            r = det.predict(img, imgsz=640, device=device, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return np.zeros((0, 4)), np.zeros(0), np.zeros(0), None, None
            kc = None if r.keypoints.conf is None else r.keypoints.conf.cpu().numpy()
            return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                    r.keypoints.xy.cpu().numpy(), kc)

        def keypoints(crop, box):
            return dk.predict_crops([crop], [box])[0]

        def classify(crop):
            r = clf.predict(crop, imgsz=CROP_SIZE, device=device, verbose=False)[0]
            return r.probs.data.cpu().numpy(), clf.names

        return cls(detect, keypoints, classify, device)

    def perceive(self, frame: Frame, info: SceneInfo) -> Perception:
        ms = dict.fromkeys(MS_KEYS, 0.0)
        t = time.perf_counter()
        small = reduce_frame(frame.image)
        ms["reduce"] = 1000 * (time.perf_counter() - t)

        t = time.perf_counter()
        xyxy, conf, _cls, _kxy, _kconf = self.detect(small)
        _sync(self.device)
        ms["detect"] = 1000 * (time.perf_counter() - t)
        conf = np.asarray(conf, float)
        out = Perception(frame.frame_id, frame.frame_idx, None, 0.0, no_detection(), None, None, ms)
        if len(conf) == 0 or conf.max() < TOP_CONF:
            return out
        j = int(np.argmax(conf))
        scale = frame.image.shape[1] / small.shape[1]
        out.box = (np.asarray(xyxy[j], float) * scale).tolist()
        out.box_conf = float(conf[j])

        t = time.perf_counter()
        crop, cbox = crop_from_detection(frame.image, out.box)
        ms["crop"] = 1000 * (time.perf_counter() - t)

        t = time.perf_counter()
        out.kpts = np.asarray(self.keypoints(crop, cbox), float)
        _sync(self.device)
        ms["keypoints"] = 1000 * (time.perf_counter() - t)
        if info.fixed_class is not None:
            return out

        t = time.perf_counter()
        probs, names = self.classify(crop)
        _sync(self.device)
        out.probs = {names[i]: float(p) for i, p in enumerate(np.asarray(probs, float))}
        ms["classify"] = 1000 * (time.perf_counter() - t)

        t = time.perf_counter()
        out.reproj = reproj_per_class({"K": frame.K}, out.kpts, info.models)
        ms["geometry"] = 1000 * (time.perf_counter() - t)
        return out
