"""Run the live perception (C -> crop -> DroneKey + K -> reprojection per class) on 6DroneSyn val and cache it.

Usage (venv ~/.venvs/dronepose-yolo, from the repo root):
  python scripts/pipeline/run_perception.py [--config configs/pipeline.json] --out OUT.jsonl
      [--sequences SEQ ...] [--limit N] [--device mps]
One Perception row per frame. Resumable: frames already in OUT are skipped. Never touches the test split.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install

from dronepose.pipeline.perception import PerceptionModels  # noqa: E402
from dronepose.pipeline.sources import SixDroneSynSource  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--canonical", default="data/processed/canonical_val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="frames per sequence")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    device = args.device or cfg["device"]
    src = SixDroneSynSource(Path(args.canonical))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {json.loads(l)["frame_id"] for l in f if l.strip()}
    models = PerceptionModels.load(cfg, device)
    seqs = args.sequences or src.sequences()
    t0, n = time.time(), 0
    with open(out, "a") as f:
        for k, seq in enumerate(seqs):
            info = src.info(seq)
            recs = src.records(seq)[:args.limit]
            if all(r["frame_id"] in done for r in recs):
                continue
            for j, frame in enumerate(src.frames(seq)):
                if args.limit is not None and j >= args.limit:
                    break
                if frame.frame_id in done:
                    continue
                f.write(json.dumps(models.perceive(frame, info).to_row()) + "\n")
                n += 1
            f.flush()
            print(f"{k + 1}/{len(seqs)} {seq}: {n} frames, {time.time() - t0:.0f}s "
                  f"({1000 * (time.time() - t0) / max(n, 1):.0f} ms/frame)", flush=True)
    print(f"done {n} new frames in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
