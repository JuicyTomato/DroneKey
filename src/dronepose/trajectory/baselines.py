import numpy as np


def constant_velocity(times: np.ndarray, positions: np.ndarray, future_times: np.ndarray) -> np.ndarray:
    times, positions, future_times = np.asarray(times, float), np.asarray(positions, float), np.asarray(future_times, float)
    if len(times) == 0:
        raise ValueError("no history")
    if len(times) == 1:
        return np.repeat(positions[-1:], len(future_times), axis=0)
    A = np.stack([np.ones_like(times), times - times[-1]], axis=1)
    coef, *_ = np.linalg.lstsq(A, positions, rcond=None)          # rows: position at last time, velocity
    dt = (future_times - times[-1])[:, None]
    return coef[0] + dt * coef[1]
