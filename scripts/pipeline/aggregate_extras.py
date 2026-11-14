"""Post hoc extras over the seeds (gravity prior, flip-aware W): per-seed rows and mean ± sd across seeds.

Usage: python scripts/pipeline/aggregate_extras.py [--seeds 0 1 2] [--out experiments/pipeline/streaming_extras_seeds.md]
Reads experiments/pipeline/streaming_extras_sN/extras.json (made by `eval_streaming.py --extras`). These rows are
declared in the thesis as added after the results; they sit next to the official ones of aggregate_seeds.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dronepose.pipeline.streaming_eval import NET  # noqa: E402


def _ms(vals, k=1.0, d=3):
    a = np.array(vals, float) * k
    return f"{a.mean():.{d}f} ± {a.std(ddof=1) if len(a) > 1 else 0.0:.{d}f}"


def _row(ex, name):
    return ex["summary"][name]["median_rot_err_deg"], ex["summary"][name]["flip_rate_no08"], \
        ex["forecast"][name][NET]["ade_1s"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="experiments/pipeline/streaming_extras_seeds.md")
    args = ap.parse_args(argv)
    ex = {s: json.loads(Path(f"experiments/pipeline/streaming_extras_s{s}/extras.json").read_text()) for s in args.seeds}
    Ws = {e["W_official"] for e in ex.values()}
    if len(Ws) != 1:
        sys.exit(f"seeds use different official W: {Ws}")
    Wc = Ws.pop()
    ci = lambda d, k=1.0, n=3: f"[{k * d[0]:+.{n}f}, {k * d[1]:+.{n}f}]"
    out = {"W_official": Wc, "per_seed": {}, "mean": {}, "sd": {}}
    L = [f"# Extra post hoc (gravita', W per flip) — val 6DroneSyn, semi {args.seeds}", "",
         "Analisi aggiunte DOPO i risultati, accanto alle righe ufficiali (`streaming_val_seeds.md`), mai al loro posto. "
         "Gravita' = `up` passato a `global` (candidati inclinati > 70° dalla verticale esclusi). "
         "IC 95%: bootstrap appaiato per traiettoria dentro ogni seme.", ""]

    # 1. offline vs offline_g, and gravity streaming vs references
    def comparison(title, a, ref):
        """rot / flip / ADE of `a` vs `ref` per seed (names are rows of extras.json)."""
        L.extend([f"## {title}", "",
                  "| seme | rot ref → a (°) | Δ rot [IC] | flip ref → a (senza 08) | Δ flip punti [IC] | ADE 1 s ref → a (m) | Δ ADE [IC] |",
                  "|---|---|---|---|---|---|---|"])
        acc = {k: [] for k in ("rot_ref", "rot_a", "rot_d", "flip_ref", "flip_a", "flip_d", "ade_ref", "ade_a", "ade_d")}
        per = {}
        for s, e in ex.items():
            rr, fr, ar = _row(e, ref)
            ra, fa, aa = _row(e, a)
            dr, df = e["rot_vs"][ref][a], e["flip_vs"][ref][a]
            da = e["forecast_minus"][f"{a} - {ref}"][NET]
            vals = dict(rot_ref=rr, rot_a=ra, rot_d=dr[0], flip_ref=fr, flip_a=fa, flip_d=df[0], ade_ref=ar, ade_a=aa,
                        ade_d=da["mean"])
            per[s] = {**vals, "ci": {"rot_d": dr[1:], "flip_d": df[1:], "ade_d": da["ci"]}}
            for k, v in vals.items():
                acc[k].append(v)
            L.append(f"| {s} | {rr:.3f} → {ra:.3f} | {dr[0]:+.3f} {ci(dr[1:])} | {100 * fr:.2f}% → {100 * fa:.2f}% "
                     f"| {100 * df[0]:+.2f} {ci(df[1:], 100, 2)} | {ar:.4f} → {aa:.4f} | {da['mean']:+.4f} {ci(da['ci'], 1, 4)} |")
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        sd = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in acc.items()}
        L.append(f"| **media ± sd** | {m['rot_ref']:.3f} → {m['rot_a']:.3f} | {m['rot_d']:+.3f} ± {sd['rot_d']:.3f} "
                 f"| {100 * m['flip_ref']:.2f}% → {100 * m['flip_a']:.2f}% | {100 * m['flip_d']:+.2f} ± {100 * sd['flip_d']:.2f} "
                 f"| {m['ade_ref']:.4f} → {m['ade_a']:.4f} | {m['ade_d']:+.4f} ± {sd['ade_d']:.4f} |")
        n = {k: int(sum(not (c[0] <= 0 <= c[1]) for c in (p["ci"][k] for p in per.values())))
             for k in ("rot_d", "flip_d", "ade_d")}
        L.extend(["", f"Semi con IC che esclude lo zero: rotazione {n['rot_d']}/{len(per)}, flip {n['flip_d']}/{len(per)}, "
                  f"ADE {n['ade_d']}/{len(per)}.", ""])
        out["per_seed"][title], out["mean"][title], out["sd"][title] = per, m, sd
        out.setdefault("n_sig", {})[title] = n

    comparison("Gravita' offline: offline_g vs offline", "offline_g", "offline")
    comparison(f"Streaming W{Wc} con gravita' vs offline con gravita' (W{Wc}_g vs offline_g)", f"W{Wc}_g", "offline_g")
    comparison(f"Streaming W{Wc} con gravita' vs offline ufficiale (W{Wc}_g vs offline)", f"W{Wc}_g", "offline")

    # 2. flip-aware W
    L += ["## Scelta di W per flip", "",
          "Copertura >= 95%, meno flip senza 08; pareggio = IC della differenza di flip include 0; poi rot mediana minore, "
          "poi W minore. Previsione con pose emesse. Δ ADE contro offline (o offline_g per la griglia con gravita').", "",
          "| griglia | seme | W scelto (in pareggio) | rot (°) | flip senza 08 | ADE 1 s (m) | Δ ADE vs riferimento [IC] |",
          "|---|---|---|---|---|---|---|"]
    out["flip_aware"] = {}
    for tag, suf, ref in (("senza gravita'", "", "offline"), ("con gravita'", "_g", "offline_g")):
        acc = {"rot": [], "flip": [], "ade": [], "dade": []}
        out["flip_aware"][tag] = {}
        for s, e in ex.items():
            fa = e["flip_aware"]["nogravity" if not suf else "gravity"]
            name = f"W{fa['chosen']}{suf}"
            r, f_, a = _row(e, name)
            key = f"{name} - {ref}"
            d = e["forecast_minus"].get(key, {}).get(NET)
            dtxt = f"{d['mean']:+.4f} {ci(d['ci'], 1, 4)}" if d else "(uguale al ref.)"
            L.append(f"| {tag} | {s} | {fa['chosen']} ({fa['tied']}) | {r:.3f} | {100 * f_:.2f}% | {a:.4f} | {dtxt} |")
            acc["rot"].append(r); acc["flip"].append(f_); acc["ade"].append(a)
            acc["dade"].append(d["mean"] if d else float("nan"))
            out["flip_aware"][tag][s] = {"chosen": fa["chosen"], "tied": fa["tied"], "rot": r, "flip": f_, "ade": a,
                                         "ade_minus_ref": d}
        L.append(f"| {tag} | **media ± sd** | | {_ms(acc['rot'])} | {_ms(acc['flip'], 100, 2)}% | {_ms(acc['ade'], 1, 4)} "
                 f"| {_ms(acc['dade'], 1, 4)} |")
    text = "\n".join(L) + "\n"
    Path(args.out).write_text(text)
    Path(args.out).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
    print(text)


if __name__ == "__main__":
    main()
