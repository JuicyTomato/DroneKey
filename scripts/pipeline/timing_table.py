"""Per-stage time table of streaming demo rows (spec 2026-11-14 §8): p50 / p95 in ms, first 10 frames excluded.

Usage: python scripts/pipeline/timing_table.py ROWS.jsonl ... [--out experiments/pipeline/timing_mac.md] [--title T]
"""
import argparse
import json
from pathlib import Path

import numpy as np

STAGES = ("reduce", "detect", "crop", "keypoints", "classify", "geometry", "decide", "forecast")
WARMUP = 10


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="Tempi per stadio, streaming (Mac)")
    args = ap.parse_args(argv)
    rows = []
    for path in args.rows:
        with open(path) as f:
            rows += [json.loads(l) for l in f if l.strip()][WARMUP:]
    rows = [r for r in rows if "ms_total" in r]
    det = [r for r in rows if r["box"] is not None]
    L = [f"# {args.title}", "", f"{len(rows)} frame ({len(det)} con box) da {len(args.rows)} sequenze; "
         f"primi {WARMUP} frame di ogni sequenza esclusi. Stadi dopo il detector: solo frame con box.", "",
         "| stadio | p50 (ms) | p95 (ms) |", "|---|---|---|"]
    for s in STAGES:
        v = [r["ms"][s] for r in (rows if s in ("reduce", "detect", "decide", "forecast") else det)]
        L.append(f"| {s} | {np.percentile(v, 50):.1f} | {np.percentile(v, 95):.1f} |")
    tot = [r["ms_total"] for r in rows]
    L.append(f"| **totale** | **{np.percentile(tot, 50):.1f}** | **{np.percentile(tot, 95):.1f}** |")
    L += ["", f"Tempo reale a 30 fps: 33.3 ms per frame; p50 = {1000 / np.percentile(tot, 50):.1f} fps."]
    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
