"""Integrated pipeline demo (spec 2026-11-14): pixels -> C -> crop -> DroneKey + K -> class K+L+g -> IPPE + LM ->
choice (offline: `global` on the whole sequence; streaming: `global` on the last W frames, emitted at once) ->
R-CV-6D forecast of the next second -> video.

Usage (venv ~/.venvs/dronepose-yolo, from the repo root):
  python scripts/demo.py --sequence 14_02_07_000 --mode offline|streaming [--W N] [--obs e|r]
      --out OUT.mp4 [--rows OUT.jsonl] [--config configs/pipeline.json] [--device mps]
Streaming processes every frame in order, using only frames up to the current one, and records the time of every
stage (no frame is skipped when slower than real time). Default W / obs: the choices of
experiments/pipeline/streaming_val_s0/results.json. Source: 6DroneSyn val only (never the test).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from dronepose.eval.units import METERS_PER_UNIT  # noqa: E402
from dronepose.pipeline.decide import StreamingDecider, decide_sequence  # noqa: E402
from dronepose.pipeline.draw import OUT_H, OUT_W, draw_frame  # noqa: E402
from dronepose.pipeline.forecast_now import Forecaster, live_window  # noqa: E402
from dronepose.pipeline.klass import load_klg_params  # noqa: E402
from dronepose.pipeline.perception import PerceptionModels  # noqa: E402
from dronepose.pipeline.sources import SixDroneSynSource  # noqa: E402

CHOICES = Path("experiments/pipeline/streaming_val_s0/results.json")
N_OBS = 15


def defaults() -> tuple[int | None, str]:
    if not CHOICES.exists():
        return None, "e"
    r = json.loads(CHOICES.read_text())
    return int(r["window_selection"]["chosen"]), r["obs_mode_selection"]["chosen"]


def forecast_cam(fc: Forecaster, window: list[dict], G: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """window: the last 15 frames {R, t, cls} (oldest first) -> (future in gravity metres, future in camera units)."""
    if len(window) < N_OBS:
        return None, None
    P, Rg, V, ref = live_window(window)
    if ref is None:
        return None, None
    fut = fc.predict(P, Rg, V, ref)
    if fut is None:
        return None, None
    return fut, (G.T @ fut.T).T / METERS_PER_UNIT[ref]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="6dronesyn", choices=["6dronesyn"])
    ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--mode", required=True, choices=["offline", "streaming"])
    ap.add_argument("--W", type=int, default=None)
    ap.add_argument("--obs", choices=["e", "r"], default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    dW, dobs = defaults()
    W, obs = args.W or dW, args.obs or dobs
    if args.mode == "streaming" and W is None:
        sys.exit("streaming needs --W (no choice found in " + str(CHOICES) + ")")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg not found")
    cfg = json.loads(Path(args.config).read_text())
    device = args.device or cfg["device"]
    src = SixDroneSynSource(Path(args.canonical))
    seq = args.sequence if args.sequence in src.sequences() else next(
        (s for s in src.sequences() if s.startswith(args.sequence)), None)
    if seq is None:
        sys.exit(f"sequence {args.sequence} not in {args.canonical}")
    info, recs = src.info(seq), src.records(seq)
    params = load_klg_params(Path(cfg["class_params"]))
    models = PerceptionModels.load(cfg, device)
    fc = Forecaster(seq[:2], Path(args.config))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
                           "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                          stdin=subprocess.PIPE)
    rows = []
    try:
        if args.mode == "streaming":
            dec = StreamingDecider(info, params, W)
            hist: deque = deque(maxlen=N_OBS)          # emitted poses {frame_idx, R, t, cls}
            for frame in src.frames(seq):
                t0 = time.perf_counter()
                p = models.perceive(frame, info)
                t1 = time.perf_counter()
                res, red = dec.step(p, frame.record, frame.K)
                t2 = time.perf_counter()
                hist.append({"frame_idx": frame.frame_idx, "R": res["R"], "t": res["t"], "cls": res["drone_model"]})
                window = list(hist)
                if obs == "r":
                    over = {x["frame_idx"]: x for x in red}
                    window = [{**h, **({"R": over[h["frame_idx"]]["R"], "t": over[h["frame_idx"]]["t"]}
                                       if h["frame_idx"] in over else {})} for h in window]
                fut_g, fut_cam = forecast_cam(fc, window, info.G)
                t3 = time.perf_counter()
                ms = {**p.ms, "decide": 1000 * (t2 - t1), "forecast": 1000 * (t3 - t2)}
                total = 1000 * (t3 - t0)
                ff.stdin.write(draw_frame(frame, p, res, fut_cam, f"streaming W={W} obs={obs}", total).tobytes())
                rows.append({**p.to_row(), "ms": ms, "ms_total": total, "R": res["R"], "t": res["t"],
                             "drone_model": res["drone_model"], "rot_err_deg": res["rot_err_deg"],
                             "future_g": None if fut_g is None else fut_g.tolist()})
        else:
            percs = [models.perceive(f, info) for f in src.frames(seq)]
            results, _ = decide_sequence(percs, recs, info, params, "offline")
            hist = deque(maxlen=N_OBS)
            for frame, p, res in zip(src.frames(seq), percs, results):
                hist.append({"frame_idx": frame.frame_idx, "R": res["R"], "t": res["t"], "cls": res["drone_model"]})
                fut_g, fut_cam = forecast_cam(fc, list(hist), info.G)
                ff.stdin.write(draw_frame(frame, p, res, fut_cam, "offline", None).tobytes())
                rows.append({**p.to_row(), "R": res["R"], "t": res["t"], "drone_model": res["drone_model"],
                             "rot_err_deg": res["rot_err_deg"], "future_g": None if fut_g is None else fut_g.tolist()})
    finally:
        ff.stdin.close()
        ff.wait()
    if args.rows:
        Path(args.rows).parent.mkdir(parents=True, exist_ok=True)
        Path(args.rows).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{out}: {len(rows)} frames, {sum(r['R'] is not None for r in rows)} with pose, "
          f"{sum(r['future_g'] is not None for r in rows)} with forecast")


if __name__ == "__main__":
    main()
