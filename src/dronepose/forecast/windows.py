"""Observation / prediction windows cut from trajectories, stride 1, never across trajectories.

Positions are relative to the last observed position (translation only) and in the trajectory's own units
(metres, or dataset units for drones without a verified size: see `metric`). All windows share the same times.
"""
from dataclasses import dataclass, field, replace

import numpy as np

from dronepose.forecast.dataset import Trajectory


@dataclass
class Windows:
    obs_t: np.ndarray    # (N,) seconds, last observation at 0
    fut_t: np.ndarray    # (H,) seconds after the last observation
    obs_p: np.ndarray    # (W,N,3) relative to last_p
    fut_p: np.ndarray    # (W,H,3) relative to last_p
    obs_R: np.ndarray    # (W,N,3,3)
    fut_R: np.ndarray    # (W,H,3,3)
    obs_u: np.ndarray    # (W,N,3)
    fut_u: np.ndarray    # (W,H,3)
    last_p: np.ndarray   # (W,3) absolute position of the last observation
    traj: np.ndarray     # (W,) index into traj_keys
    start: np.ndarray    # (W,) first observed frame
    metric: np.ndarray   # (W,) bool: positions in metres
    scene: np.ndarray    # (W,) str
    drone: np.ndarray    # (W,) str
    motion: np.ndarray   # (W,) str
    traj_keys: list[str]
    meta: dict[str, np.ndarray] = field(default_factory=dict)  # one column per meta key shared by all trajectories
    obs_valid: np.ndarray | None = None  # (W,N) bool: observed frames with a pose (None = all; lost frames ignored)

    def __len__(self) -> int:
        return len(self.traj)

    def subset(self, idx) -> "Windows":
        """The windows `idx` (an index array or a boolean mask), every per-window field cut alike."""
        cut = lambda a: None if a is None else np.asarray(a)[idx]
        return replace(self, obs_p=cut(self.obs_p), fut_p=cut(self.fut_p), obs_R=cut(self.obs_R),
                       fut_R=cut(self.fut_R), obs_u=cut(self.obs_u), fut_u=cut(self.fut_u), last_p=cut(self.last_p),
                       traj=cut(self.traj), start=cut(self.start), metric=cut(self.metric), scene=cut(self.scene),
                       drone=cut(self.drone), motion=cut(self.motion), meta={k: cut(v) for k, v in self.meta.items()},
                       obs_valid=cut(self.obs_valid))


def _cut(trajs: list[Trajectory], starts: list[np.ndarray], N: int, H: int, fps: int) -> Windows:
    pos, rot, meta = [], [], []
    for i, (tr, st) in enumerate(zip(trajs, starts)):
        steps = np.diff(tr.times)
        if len(steps) and not np.allclose(steps, 1 / fps, rtol=0, atol=1e-9):
            raise ValueError(f"{tr.key}: time step is not 1/{fps}")
        n = len(st)
        if n <= 0:
            continue
        idx = st[:, None] + np.arange(N + H)[None]
        pos.append(tr.positions[idx])
        rot.append(tr.R_G[idx])
        meta.append((np.full(n, i), st, np.full(n, tr.metric), np.full(n, tr.scene),
                     np.full(n, tr.drone), np.full(n, tr.motion_type)))
    if pos:
        P, R = np.concatenate(pos), np.concatenate(rot)
        traj, start, metric, scene, drone, motion = (np.concatenate(m) for m in zip(*meta))
    else:
        P, R = np.zeros((0, N + H, 3)), np.zeros((0, N + H, 3, 3))
        traj, start = np.zeros(0, int), np.zeros(0, int)
        metric, scene, drone, motion = np.zeros(0, bool), np.zeros(0, str), np.zeros(0, str), np.zeros(0, str)
    last = P[:, N - 1].copy()
    rel = P - last[:, None]
    U = R[..., :, 2]
    traj = traj.astype(int)
    shared = set.intersection(*(set(t.meta) for t in trajs)) if trajs else set()
    if "weight" not in shared and any("weight" in t.meta for t in trajs):
        raise ValueError("some trajectories carry an importance 'weight' and others do not (stratified and "
                         "unstratified sets mixed): the weights would be dropped silently")
    columns = {k: np.array([trajs[j].meta[k] for j in traj]) for k in sorted(shared)} if len(traj) else {}
    return Windows(obs_t=(np.arange(N) - (N - 1)) / fps, fut_t=np.arange(1, H + 1) / fps,
                   obs_p=rel[:, :N], fut_p=rel[:, N:], obs_R=R[:, :N], fut_R=R[:, N:], obs_u=U[:, :N], fut_u=U[:, N:],
                   last_p=last, traj=traj, start=start.astype(int), metric=metric.astype(bool),
                   scene=scene, drone=drone, motion=motion, traj_keys=[t.key for t in trajs], meta=columns)


def make_windows(trajs: list[Trajectory], obs_len: int, pred_len: int, fps: int) -> Windows:
    N, H = obs_len, pred_len
    return _cut(trajs, [np.arange(max(len(tr.times) - N - H + 1, 0)) for tr in trajs], N, H, fps)


def freeze_windows(trajs: list[Trajectory], obs_len: int, pred_len: int, fps: int) -> Windows:
    """One window per trajectory whose last observed frame is `freeze_idx` (generated sets: frozen sticks after it)."""
    N, H = obs_len, pred_len
    starts = []
    for tr in trajs:
        f = tr.freeze_idx
        if f is None:
            raise ValueError(f"{tr.key}: no freeze_idx (not a generated trajectory)")
        if f + 1 < N or f + H >= len(tr.times):
            raise ValueError(f"{tr.key}: freeze_idx {f} leaves fewer than {N} observed or {H} future frames")
        starts.append(np.array([f - N + 1]))
    return _cut(trajs, starts, N, H, fps)
