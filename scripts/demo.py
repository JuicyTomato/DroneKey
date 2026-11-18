"""Integrated pipeline demo (spec 2026-11-14): pixels -> C -> crop -> DroneKey + K -> class K+L+g -> IPPE + LM ->
choice (offline: `global` on the whole sequence; streaming: `global` on the last W frames, emitted at once) ->
R-CV-6D forecast of the next second -> video.

Usage (venv ~/.venvs/dronepose-yolo, from the repo root):
  python scripts/demo.py --sequence 14_02_07_000 --mode offline|streaming [--W N] [--obs e|r]
      --out OUT.mp4 [--rows OUT.jsonl] [--config configs/pipeline.json] [--device mps] [--gravity]
Streaming processes every frame in order, using only frames up to the current one, and records the time of every
stage (no frame is skipped when slower than real time). Default W / obs: the choices of
experiments/pipeline/streaming_val_s0/results.json. Source: 6DroneSyn val (never the test) or --source mav6d (authors' train only, --config configs/pipeline_mav6d.json).
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
from dronepose.pipeline.perception import PerceptionModels, undistort_perception  # noqa: E402
from dronepose.pipeline.sources import MAV6DSource, SixDroneSynSource  # noqa: E402

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
    P, Rg, V, ref = live_window(window, G)
    if ref is None:
        return None, None
    fut = fc.predict(P, Rg, V, ref)
    if fut is None:
        return None, None
    return fut, (G.T @ fut.T).T / METERS_PER_UNIT[ref]


def shown_forecast(res: dict, fut_cam, only_with_pose: bool):
    """What is drawn: None when there is no forecast or it is not finite, or (only_with_pose) when this frame has no pose."""
    if fut_cam is None or not np.isfinite(np.asarray(fut_cam, float)).all():
        return None
    if only_with_pose and res.get("R") is None:
        return None
    return fut_cam


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="6dronesyn", choices=["6dronesyn", "mav6d"])
    ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
    ap.add_argument("--gt", default="data/processed/mav6d_gt.jsonl", help="mav6d only")
    ap.add_argument("--hub-model", default="experiments/real/mav6d/hub_model.json", help="mav6d only")
    ap.add_argument("--fps", type=float, default=None, help="video fps (default 30; mav6d: from the source)")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames (trial runs only)")
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--mode", required=True, choices=["offline", "streaming"])
    ap.add_argument("--W", type=int, default=None)
    ap.add_argument("--obs", choices=["e", "r"], default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-forecast", action="store_true",
                    help="no forecast: Forecaster not loaded, nothing computed or drawn, header without the forecast")
    ap.add_argument("--gravity", action="store_true",
                    help="gravity prior in the choice (candidates tilted > 70 deg from the vertical are excluded); "
                         "post hoc analysis, off by default")
    return ap


def resolve_source(args) -> tuple:
    """(source, sequence, video fps); exits with a message on a missing or test sequence, before any model loads."""
    if args.source == "mav6d":
        src = MAV6DSource(Path(args.gt), Path(args.hub_model))
        seq = args.sequence
        try:
            src.records(seq)
        except ValueError as e:
            sys.exit(f"sequence {seq}: {e} (authors' test split is never used)")
        except KeyError:
            sys.exit(f"sequence {seq} not found in {args.gt} (full name needed, e.g. {src.sequences()[:1]})")
        return src, seq, args.fps if args.fps else round(src.fps(seq), 2)
    src = SixDroneSynSource(Path(args.canonical))
    seq = args.sequence if args.sequence in src.sequences() else next(
        (s for s in src.sequences() if s.startswith(args.sequence)), None)
    if seq is None:
        sys.exit(f"sequence {args.sequence} not found in {args.canonical}")
    return src, seq, args.fps if args.fps else 30


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    dW, dobs = defaults()
    W, obs = args.W or dW, args.obs or dobs
    if args.mode == "streaming" and W is None:
        sys.exit("streaming needs --W (no choice found in " + str(CHOICES) + ")")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg not found")
    cfg = json.loads(Path(args.config).read_text())
    device = args.device or cfg["device"]
    src, seq, fps = resolve_source(args)
    mav = args.source == "mav6d"
    info, recs = src.info(seq), src.records(seq)
    params = load_klg_params(Path(cfg["class_params"]))
    models = PerceptionModels.load(cfg, device)
    fc = None if args.no_forecast else Forecaster(cfg.get("forecast_scene", seq[:2]), Path(args.config))
    show_fc = not args.no_forecast
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                          stdin=subprocess.PIPE)
    rows = []
    try:
        if args.mode == "streaming":
            dec = StreamingDecider(info, params, W, args.gravity)
            hist: deque = deque(maxlen=N_OBS)          # emitted poses {frame_idx, R, t, cls}
            for n, frame in enumerate(src.frames(seq)):
                if args.max_frames is not None and n >= args.max_frames:
                    break
                t0 = time.perf_counter()
                p = models.perceive(frame, info)
                t1 = time.perf_counter()
                pd = undistort_perception(p, frame.K, frame.dist) if mav else p
                res, red = dec.step(pd, frame.record, frame.K)
                t2 = time.perf_counter()
                hist.append({"frame_idx": frame.frame_idx, "R": res["R"], "t": res["t"], "cls": res["drone_model"]})
                window = list(hist)
                if obs == "r":
                    over = {x["frame_idx"]: x for x in red}
                    window = [{**h, **({"R": over[h["frame_idx"]]["R"], "t": over[h["frame_idx"]]["t"]}
                                       if h["frame_idx"] in over else {})} for h in window]
                fut_g, fut_cam = (None, None) if fc is None else forecast_cam(fc, window, info.G)
                t3 = time.perf_counter()
                ms = {**p.ms, "decide": 1000 * (t2 - t1), "forecast": 1000 * (t3 - t2)}
                total = 1000 * (t3 - t0)
                shown = shown_forecast(res, fut_cam, mav)
                ff.stdin.write(draw_frame(frame, p, res, shown, f"streaming W={W} obs={obs}", total, show_fc).tobytes())
                rows.append({**p.to_row(), "ms": ms, "ms_total": total, "R": res["R"], "t": res["t"],
                             "drone_model": res["drone_model"], "rot_err_deg": res["rot_err_deg"],
                             "future_g": None if fut_g is None else fut_g.tolist(),
                             "forecast_shown": shown is not None})
        else:
            frames = list(src.frames(seq))[:args.max_frames]
            percs = [models.perceive(f, info) for f in frames]
            pds = [undistort_perception(p, f.K, f.dist) if mav else p for p, f in zip(percs, frames)]
            results, _ = decide_sequence(pds, recs[:len(frames)], info, params, "offline", gravity=args.gravity)
            hist = deque(maxlen=N_OBS)
            for frame, p, res in zip(frames, percs, results):
                hist.append({"frame_idx": frame.frame_idx, "R": res["R"], "t": res["t"], "cls": res["drone_model"]})
                fut_g, fut_cam = (None, None) if fc is None else forecast_cam(fc, list(hist), info.G)
                shown = shown_forecast(res, fut_cam, mav)
                ff.stdin.write(draw_frame(frame, p, res, shown, "offline", None, show_fc).tobytes())
                rows.append({**p.to_row(), "R": res["R"], "t": res["t"], "drone_model": res["drone_model"],
                             "rot_err_deg": res["rot_err_deg"], "future_g": None if fut_g is None else fut_g.tolist(),
                             "forecast_shown": shown is not None})
    finally:
        ff.stdin.close()
        ff.wait()
    if args.rows:
        Path(args.rows).parent.mkdir(parents=True, exist_ok=True)
        Path(args.rows).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{out}: {len(rows)} frames, {sum(r['R'] is not None for r in rows)} with pose, "
          f"{sum(r['forecast_shown'] for r in rows)} with forecast shown")


if __name__ == "__main__":
    main()
