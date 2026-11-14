"""Offline vs streaming over the seeds: per-seed rows (with their 95% intervals) and mean ± sd across seeds.

Usage: python scripts/pipeline/aggregate_seeds.py [--seeds 0 1 2] [--out experiments/pipeline/streaming_val_seeds.md]
Reads experiments/pipeline/streaming_val_sN/results.json (W and observation mode of seeds 1, 2 frozen to seed 0's).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dronepose.pipeline.streaming_eval import seed_summary  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="experiments/pipeline/streaming_val_seeds.md")
    args = ap.parse_args(argv)
    res = {s: json.loads(Path(f"experiments/pipeline/streaming_val_s{s}/results.json").read_text()) for s in args.seeds}
    W = {s: r["window_selection"]["chosen"] for s, r in res.items()}
    obs = {s: r["obs_mode_selection"]["chosen"] for s, r in res.items()}
    if len(set(W.values())) != 1 or len(set(obs.values())) != 1:
        sys.exit(f"seeds use different choices: W {W}, obs {obs}")
    a = seed_summary(res)
    P, C = a["per_seed"], a["ci"]
    f = lambda lo_hi, k=1.0, d=3: f"[{k * lo_hi[0]:+.{d}f}, {k * lo_hi[1]:+.{d}f}]"
    L = [f"# Offline vs streaming W = {W[args.seeds[0]]} ({obs[args.seeds[0]]}) — val 6DroneSyn, semi {args.seeds}", "",
         "Seme = DroneKey + classificatore K + R-CV-6D dello stesso seme; detector C unico. W e osservazioni scelti sul seme 0 "
         "e fissati per gli altri. IC 95%: bootstrap appaiato per traiettoria, dentro ogni seme.", "",
         "| seme | rot off → stream (°) | Δ rot [IC] | flip off → stream (senza 08) | Δ flip punti [IC] | ADE 1 s off → stream (m) | Δ ADE [IC] |",
         "|---|---|---|---|---|---|---|"]
    for s, p in P.items():
        c = C[s]
        L.append(f"| {s} | {p['rot_off']:.3f} → {p['rot_stream']:.3f} | {p['rot_delta']:+.3f} {f(c['rot_delta'])} "
                 f"| {100 * p['flip_off']:.2f}% → {100 * p['flip_stream']:.2f}% | {100 * p['flip_delta']:+.2f} {f(c['flip_delta'], 100, 2)} "
                 f"| {p['ade_off']:.4f} → {p['ade_stream']:.4f} | {p['ade_delta']:+.4f} {f(c['ade_delta'], 1, 4)} |")
    m, sd, n = a["mean"], a["sd"], a["n_sig"]
    L.append(f"| **media ± sd** | {m['rot_off']:.3f} → {m['rot_stream']:.3f} | {m['rot_delta']:+.3f} ± {sd['rot_delta']:.3f} "
             f"| {100 * m['flip_off']:.2f}% → {100 * m['flip_stream']:.2f}% | {100 * m['flip_delta']:+.2f} ± {100 * sd['flip_delta']:.2f} "
             f"| {m['ade_off']:.4f} → {m['ade_stream']:.4f} | {m['ade_delta']:+.4f} ± {sd['ade_delta']:.4f} |")
    L += ["", f"Semi con IC che esclude lo zero: rotazione {n['rot_delta']}/{len(P)}, flip {n['flip_delta']}/{len(P)}, "
          f"ADE {n['ade_delta']}/{len(P)}."]
    text = "\n".join(L) + "\n"
    Path(args.out).write_text(text)
    Path(args.out).with_suffix(".json").write_text(json.dumps(a, indent=1))
    print(text)


if __name__ == "__main__":
    main()
