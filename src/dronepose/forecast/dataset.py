"""Trajectory dataset (plan step 3.1): one trajectory per (scene, drone), gravity frame, metres where the scale is known.

The three background copies of a (scene, drone) share the exact same pose sequence (verified bit-for-bit), so only
one is kept; `copies` remembers them for the pipeline evaluation (E4b), where each copy is a separate noisy view.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dronepose.forecast.config import ForecastConfig
from dronepose.forecast.frames import to_gravity_positions, to_gravity_rotations, up_axis


@dataclass
class Trajectory:
    scene: str
    drone: str
    split: str
    motion_type: str
    copies: list[tuple[str, int]]   # (background, degree)
    s_c: float | None               # dataset units -> metres; None = no verified size (Tello)
    times: np.ndarray               # (T,) seconds, first frame at 0
    p_G: np.ndarray                 # (T,3) dataset units, gravity frame
    R_G: np.ndarray = field(repr=False)  # (T,3,3) body FLU -> G
    freeze_idx: int | None = None        # generated sets: last observed frame before the frozen sticks
    meta: dict = field(default_factory=dict)  # generated sets: JSON-serialisable per-sample metadata

    @property
    def u(self) -> np.ndarray:
        return up_axis(self.R_G)

    @property
    def metric(self) -> bool:
        return self.s_c is not None

    @property
    def positions(self) -> np.ndarray:
        return self.s_c * self.p_G if self.metric else self.p_G

    @property
    def key(self) -> str:
        # generated samples share scene and class: their id keeps the key unique
        return f"{self.scene}_{self.drone}" + (f"_{self.meta['sample_id']}" if "sample_id" in self.meta else "")


def _sequence_arrays(recs: list[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    recs = sorted(recs, key=lambda r: r["frame_idx"])
    idx = [r["frame_idx"] for r in recs]
    if idx != list(range(1, len(recs) + 1)):
        raise ValueError(f"{name}: frames not contiguous from 1 (got {len(idx)} frames, {idx[0]}..{idx[-1]})")
    return np.array([r["t_gt"] for r in recs], float), np.array([r["R_gt"] for r in recs], float)


def build_trajectories(records: list[dict], cfg: ForecastConfig, scales: dict[str, float | None]) -> list[Trajectory]:
    groups: dict[tuple[str, str], dict[tuple[str, int], list[dict]]] = {}
    for r in records:
        groups.setdefault((r["scene"], r["drone"]), {}).setdefault((r["background"], int(r["degree"])), []).append(r)
    trajs = []
    for (scene, drone), copies in sorted(groups.items()):
        names = sorted(copies)
        t0, R0 = _sequence_arrays(copies[names[0]], f"{scene}_{drone}_{names[0][0]}")
        for name in names[1:]:
            t, R = _sequence_arrays(copies[name], f"{scene}_{drone}_{name[0]}")
            if not (np.array_equal(t, t0) and np.array_equal(R, R0)):
                raise ValueError(f"{scene}_{drone}: background copies differ ({names[0][0]} vs {name[0]})")
        splits = {r["split"] for c in copies.values() for r in c}
        if len(splits) != 1:
            raise ValueError(f"{scene}_{drone}: several splits {splits}")
        trajs.append(Trajectory(
            scene=scene, drone=drone, split=splits.pop(), motion_type=cfg.motion_type[scene], copies=names,
            s_c=scales.get(drone), times=np.arange(len(t0)) / cfg.fps,
            p_G=to_gravity_positions(t0), R_G=to_gravity_rotations(R0)))
    return trajs


def save_trajectories(trajs: list[Trajectory], out_dir: Path, split: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, index = {}, []
    for tr in trajs:
        arrays[f"{tr.key}/times"], arrays[f"{tr.key}/p_G"], arrays[f"{tr.key}/R_G"] = tr.times, tr.p_G, tr.R_G
        index.append({"key": tr.key, "scene": tr.scene, "drone": tr.drone, "split": tr.split,
                      "motion_type": tr.motion_type, "copies": [list(c) for c in tr.copies], "s_c": tr.s_c,
                      "length": len(tr.times), "freeze_idx": tr.freeze_idx, "meta": tr.meta})
    np.savez_compressed(out_dir / f"trajectories_{split}.npz", **arrays)
    (out_dir / f"index_{split}.json").write_text(json.dumps(index, indent=1))


def load_trajectories(out_dir: Path, split: str) -> list[Trajectory]:
    out_dir = Path(out_dir)
    index = json.loads((out_dir / f"index_{split}.json").read_text())
    with np.load(out_dir / f"trajectories_{split}.npz") as z:
        return [Trajectory(scene=e["scene"], drone=e["drone"], split=e["split"], motion_type=e["motion_type"],
                           copies=[(c[0], int(c[1])) for c in e["copies"]], s_c=e["s_c"],
                           times=z[f"{e['key']}/times"], p_G=z[f"{e['key']}/p_G"], R_G=z[f"{e['key']}/R_G"],
                           freeze_idx=e.get("freeze_idx"), meta=e.get("meta", {}))
                for e in index]
