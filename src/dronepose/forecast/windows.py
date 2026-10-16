"""Observation / prediction windows cut from trajectories, stride 1, never across trajectories.

Positions are relative to the last observed position (translation only) and in the trajectory's own units
(metres, or dataset units for drones without a verified size: see `metric`). All windows share the same times.
"""
from dataclasses import dataclass

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

    def __len__(self) -> int:
        return len(self.traj)


def make_windows(trajs: list[Trajectory], obs_len: int, pred_len: int, fps: int) -> Windows:
    N, H = obs_len, pred_len
    pos, rot, meta = [], [], []
    for i, tr in enumerate(trajs):
        steps = np.diff(tr.times)
        if len(steps) and not np.allclose(steps, 1 / fps, rtol=0, atol=1e-9):
            raise ValueError(f"{tr.key}: time step is not 1/{fps}")
        n = len(tr.times) - N - H + 1
        if n <= 0:
            continue
        idx = np.arange(n)[:, None] + np.arange(N + H)[None]
        pos.append(tr.positions[idx])
        rot.append(tr.R_G[idx])
        meta.append((np.full(n, i), np.arange(n), np.full(n, tr.metric), np.full(n, tr.scene),
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
    return Windows(obs_t=(np.arange(N) - (N - 1)) / fps, fut_t=np.arange(1, H + 1) / fps,
                   obs_p=rel[:, :N], fut_p=rel[:, N:], obs_R=R[:, :N], fut_R=R[:, N:], obs_u=U[:, :N], fut_u=U[:, N:],
                   last_p=last, traj=traj.astype(int), start=start.astype(int), metric=metric.astype(bool),
                   scene=scene, drone=drone, motion=motion, traj_keys=[t.key for t in trajs])
