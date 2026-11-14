"""Choice between the IPPE solutions: offline (`global` on the whole sequence) or streaming (`global` on the last W
frames, the choice for the newest frame is emitted at once; spec 2026-11-14 §5.2)."""
from collections import deque

import numpy as np

from dronepose.data.drone_models import DroneModel
from dronepose.pose.disambiguation import Choice
from dronepose.pose.global_disambiguation import GlobalParams, disambiguate_global
from dronepose.pose.pnp import PnPSolution
from dronepose.pose.solvers import solve

N_REDECIDED = 15     # observation window of the forecast


def candidates(kpts: np.ndarray, K: np.ndarray, model: DroneModel | None) -> list[PnPSolution]:
    """IPPE + LM candidates with `model`; [] without a model or with non-finite keypoints."""
    if model is None:
        return []
    img = np.asarray(kpts, float)[:, :2]
    if not np.isfinite(img).all():
        return []
    return solve("ippe", model.kpts_flu, img, np.asarray(K, float), refine=True)


def choose_offline(cands: list[list[PnPSolution]], frame_idx: list[int], up: np.ndarray | None = None) -> list[Choice]:
    """`up` (camera-frame vertical) is the optional gravity prior of `disambiguate_global`."""
    return disambiguate_global(cands, GlobalParams(), frame_idx, up=up)


class WindowChooser:
    def __init__(self, W: int, params: GlobalParams = GlobalParams(), up: np.ndarray | None = None):
        if W < 1:
            raise ValueError("W must be >= 1")
        self.W, self.params, self.up = W, params, up
        self.cands: deque = deque(maxlen=W)
        self.idx: deque = deque(maxlen=W)

    def step(self, cands: list[PnPSolution], frame_idx: int) -> tuple[Choice, list[Choice]]:
        """(choice emitted for this frame, choices re-decided now for the last min(15, n) frames, oldest first)."""
        self.cands.append(cands)
        self.idx.append(frame_idx)
        ch = disambiguate_global(list(self.cands), self.params, list(self.idx), up=self.up)
        return ch[-1], ch[-N_REDECIDED:]
