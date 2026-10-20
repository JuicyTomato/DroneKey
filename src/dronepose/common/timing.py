"""Latency summaries for the FPS benchmark."""
import numpy as np


def latency_summary(ms: list[float], warmup: int = 50) -> dict:
    """Median and p90 of per-frame latencies (ms) after dropping the first `warmup` measurements."""
    x = np.asarray(ms, float)[warmup:]
    if len(x) == 0:
        raise ValueError(f"no measurements left after dropping {warmup} warmup frames")
    return {"n": int(len(x)), "median_ms": float(np.median(x)), "p90_ms": float(np.percentile(x, 90))}


def end_to_end_fps(stage_medians_ms: list[float]) -> float:
    """Frames per second of a sequential pipeline whose stages take the given median times."""
    return 1000.0 / sum(stage_medians_ms)
