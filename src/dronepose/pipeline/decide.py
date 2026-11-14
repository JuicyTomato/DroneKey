"""Decision stage of the integrated pipeline (spec 2026-11-14 §5.2-§5.4): class K+L+g (causal), IPPE + LM with the
class model, choice offline or on a window of W frames; pose files in the E4b format; windows of re-decided poses.
"""
import dataclasses
import importlib.util
from pathlib import Path

import numpy as np

from dronepose.eval.metrics import SYMMETRIC_DRONES, rotation_error_deg, symmetric_rotation_error_deg, translation_error
from dronepose.eval.units import GRAVITY_FROM_CAMERA, METERS_PER_UNIT
from dronepose.forecast.windows import Windows, make_windows
from dronepose.pipeline.choice import WindowChooser, candidates, choose_offline
from dronepose.pipeline.klass import CausalClassFuser, klg_frame_scores
from dronepose.pipeline.perception import Perception
from dronepose.pipeline.sources import SceneInfo
from dronepose.pose.disambiguation import Choice

MODES = ("offline", "streaming")
_EXPORT = Path(__file__).resolve().parents[3] / "scripts" / "export_poses.py"


def _export_poses_module():
    spec = importlib.util.spec_from_file_location("_export_poses", _EXPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pose(ch: Choice | None) -> tuple[list | None, list | None]:
    if ch is None or ch.rejected or ch.solution is None:
        return None, None
    return ch.solution.R.tolist(), ch.solution.t.tolist()


def _result(rec: dict, perc: Perception, ch: Choice, drone_model: str | None) -> dict:
    R, t = _pose(ch)
    out = {"frame_id": perc.frame_id, "frame_idx": perc.frame_idx, "drone_model": drone_model, "R": R, "t": t,
           "rejected": R is None, "reject_reason": "no_class" if drone_model is None else (ch.reason if R is None else None),
           "rot_err_deg": None, "rot_err_sym_deg": None, "t_err": None, "ade": None, "fde": None}
    if R is not None and rec is not None and "R_gt" in rec:
        R_gt = np.asarray(rec["R_gt"], float)
        e = rotation_error_deg(np.asarray(R), R_gt)
        out.update(rot_err_deg=e, t_err=translation_error(np.asarray(t), np.asarray(rec["t_gt"], float)),
                   rot_err_sym_deg=symmetric_rotation_error_deg(np.asarray(R), R_gt)
                   if rec.get("drone") in SYMMETRIC_DRONES else e)
    return out


class StreamingDecider:
    """Decision of one frame at a time, past only: class, IPPE + LM with the class model, choice on the last W frames."""

    def __init__(self, info: SceneInfo, params: dict, W: int):
        self.info, self.params = info, params
        self.fuser = CausalClassFuser()
        self.chooser = WindowChooser(W)

    def step(self, p: Perception, rec: dict | None, K) -> tuple[dict, list[dict]]:
        """(result of this frame, poses re-decided now for the last min(15, W, n) frames, oldest first)."""
        cls = self.info.fixed_class or self.fuser.update(klg_frame_scores(p.probs, p.reproj, self.params))
        c = candidates(p.kpts, K, self.info.models[cls] if cls else None)
        ch, red = self.chooser.step(c, p.frame_idx)
        idx = list(self.chooser.idx)[-len(red):]
        return (_result(rec, p, ch, cls),
                [{"frame_idx": i, "R": _pose(x)[0], "t": _pose(x)[1]} for i, x in zip(idx, red)])


def decide_sequence(perc: list[Perception], records: list[dict], info: SceneInfo, params: dict, mode: str,
                    W: int | None = None) -> tuple[list[dict], dict[int, list[dict]]]:
    """One result per frame (replay format + drone_model) and, in streaming, the poses re-decided at every frame
    for the last 15 frames ({frame_idx: [{"frame_idx", "R", "t"}, ...]}, oldest first; empty dict offline)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}: expected one of {MODES}")
    if mode == "streaming" and (W is None or W < 1):
        raise ValueError("streaming needs W >= 1")
    if [p.frame_id for p in perc] != [r["frame_id"] for r in records]:
        raise ValueError("perceptions and records are not the same frames in the same order")
    if mode == "streaming":
        dec = StreamingDecider(info, params, W)
        out, redecided = [], {}
        for p, r in zip(perc, records):
            res, red = dec.step(p, r, r["K"])
            out.append(res)
            redecided[p.frame_idx] = red
        return out, redecided
    fuser = CausalClassFuser()
    classes, cands = [], []
    for p, r in zip(perc, records):
        cls = info.fixed_class or fuser.update(klg_frame_scores(p.probs, p.reproj, params))
        classes.append(cls)
        cands.append(candidates(p.kpts, r["K"], info.models[cls] if cls else None))
    emitted = choose_offline(cands, [p.frame_idx for p in perc])
    return [_result(r, p, ch, cls) for p, r, ch, cls in zip(perc, records, emitted, classes)], {}


def write_pose_file(results_by_seq: dict[str, list[dict]], records_by_id: dict[str, dict], split: str, seed: int,
                    combo: str, path: Path) -> None:
    """Rows of `export_poses.pose_rows` (the E4b format read by `load_pose_windows`)."""
    mod = _export_poses_module()
    res = [r for rs in results_by_seq.values() for r in rs]
    model_drone = {r["frame_id"]: r["drone_model"] for r in res}
    mod.write_jsonl(Path(path), mod.pose_rows(res, records_by_id, model_drone, split, seed, combo))


def _g_pose(R, t, drone: str, model: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    """R_g, t_g as E4b sees them (pose_rows + rescale to the predicted class); None = lost frame."""
    if R is None or model is None:
        return None
    s_true = METERS_PER_UNIT.get(drone)
    s = 1.0 if s_true is None else (s_true if model == drone else METERS_PER_UNIT.get(model))
    if s is None:
        return None
    G = GRAVITY_FROM_CAMERA
    return G @ np.asarray(R, float), G @ np.asarray(t, float) * s


def redecided_windows(redecided_by_seq: dict[str, dict[int, list[dict]]], emitted_by_seq: dict[str, list[dict]],
                      records_by_id: dict[str, dict], trajs, cfg, seed: int = 0) -> Windows:
    """As `load_pose_windows`, but the window ending at frame t observes the poses re-decided at t (frames outside
    the re-decided list keep the emitted pose; the class of each frame is the one emitted for it)."""
    gt = make_windows(trajs, cfg.obs_len, cfg.pred_len, cfg.fps)
    N = cfg.obs_len
    rows, copies, seeds, obs_p, obs_R, valid = [], [], [], [], [], []
    for j, tr in enumerate(trajs):
        win = np.flatnonzero(gt.traj == j)
        for bg, deg in tr.copies:
            seq = f"{tr.scene}_{tr.drone}_{bg}_{int(deg):03d}"
            em = {r["frame_idx"]: r for r in emitted_by_seq[seq]}
            red = redecided_by_seq[seq]
            for w in win:
                frames = gt.start[w] + np.arange(N) + 1                     # frame_idx are 1-based
                over = {x["frame_idx"]: x for x in red.get(int(frames[-1]), [])}
                P, Rw, V = np.zeros((N, 3)), np.tile(np.eye(3), (N, 1, 1)), np.zeros(N, bool)
                for k, f in enumerate(frames):
                    e = em[int(f)]
                    x = over.get(int(f), e)
                    drone = records_by_id[e["frame_id"]]["drone"]
                    g = _g_pose(x["R"], x["t"], drone, e["drone_model"])
                    if g is not None:
                        Rw[k], P[k], V[k] = g[0], g[1], True
                rows.append(w)
                obs_p.append(P - gt.last_p[w])
                obs_R.append(Rw)
                valid.append(V)
            copies += [f"{bg}_{int(deg):03d}"] * len(win)
            seeds += [seed] * len(win)
    idx = np.array(rows, int)
    w = gt.subset(idx)
    obs_R = np.stack(obs_R)
    return dataclasses.replace(w, obs_p=np.stack(obs_p), obs_R=obs_R, obs_u=obs_R[..., :, 2], obs_valid=np.stack(valid),
                               meta={**w.meta, "copy": np.array(copies), "pose_seed": np.array(seeds), "gt_index": idx})
