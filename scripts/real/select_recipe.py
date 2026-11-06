"""Rule of spec §7 on the results of eval_e2.py (authors' train split only).

  .venv/bin/python scripts/real/select_recipe.py --e2 sel/e2.json --dumps sel/frames --ref final --cand cp50 --cand cp90 --out sel/step1.json
  .venv/bin/python scripts/real/select_recipe.py --decide sel/step1.json sel/step2.json [sel/combo.json] --out sel/decision.json

Rotation with the real 3D model, paired on the frames with a pose in both models, bootstrap per sequence.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # dronepose without pip install
from dronepose.eval.bootstrap import paired_bootstrap  # noqa: E402
from dronepose.eval.recipe_select import decide, paired_groups, paired_median, wins  # noqa: E402

N_BOOT = 2000


def load_rot(dumps: Path, name: str) -> dict:
    return json.loads((dumps / f"{name}.json").read_text())


def head_to_head(names: list[str], dumps: Path) -> str:
    """The model with the lowest rotation against each other one, each time on the frames with a pose in both."""
    best = names[0]
    for other in names[1:]:
        a, b, _ = paired_median(load_rot(dumps, best), load_rot(dumps, other))
        if b < a:
            best = other
    return best


def compare(e2: dict, dumps: Path, ref: str, cands: list[str]) -> dict:
    if any("authors_test" in m["real"] for m in e2["models"].values()):
        raise ValueError("these results include the authors' test split: run eval_e2.py on authors_train/gt.jsonl")
    rows = load_rot(dumps, ref)
    r = e2["models"][ref]["real"]["all"]
    out: dict = {"ref": ref, "ref_median": r["median_rot_sym"], "candidates": {}}
    for c in cands:
        m = e2["models"][c]
        p, px = m["real"]["all"], m["kpts2d"]["all"]
        groups = paired_groups(rows, load_rot(dumps, c))
        boot = paired_bootstrap(groups, "ref", N_BOOT)["cand"]
        out["candidates"][c] = {"boot": boot, "wins": wins(boot, r["coverage"], p["coverage"], r["flip_rate"], p["flip_rate"]),
                                "median": p["median_rot_sym"], "coverage": p["coverage"], "flip_rate": p["flip_rate"],
                                "px_median": px["px_median"], "paired_sequences": len(groups["ref"]),
                                "paired_frames": sum(map(len, groups["ref"].values()))}
        print(f"{c:10s} rot {p['median_rot_sym']:.2f} diff {boot[0]:+.2f} [{boot[1]:+.2f}, {boot[2]:+.2f}]  px "
              f"{px['px_median']:.1f}  cov {p['coverage']:.3f}  flip {p['flip_rate']:.3f}  "
              f"{'WINS' if out['candidates'][c]['wins'] else '-'}")
    winners = [c for c, v in out["candidates"].items() if v["wins"]]
    out["best"] = head_to_head(cands, dumps) if cands else None     # also without winners: the detector needs a level
    out["winner"] = head_to_head(winners, dumps) if winners else None
    out["median"] = out["candidates"][out["winner"]]["median"] if winners else None
    print(f"reference {ref}: rot {r['median_rot_sym']:.2f}  cov {r['coverage']:.3f}  flip {r['flip_rate']:.3f}; "
          f"winner: {out['winner']}, best: {out['best']}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e2")
    ap.add_argument("--dumps")
    ap.add_argument("--ref")
    ap.add_argument("--cand", action="append", default=[])
    ap.add_argument("--decide", nargs="+", metavar="STEP", help="step1.json step2.json [combo.json]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.decide:
        steps = [json.loads(Path(p).read_text()) for p in args.decide]
        a, b = steps[0].get("winner"), steps[1].get("winner")
        if a and b:                                    # two winners are compared head to head, not on their own frames
            if not args.dumps:
                ap.error("--decide with two winners needs --dumps")
            dumps = Path(args.dumps)
            steps[0]["median"], steps[1]["median"], _ = paired_median(load_rot(dumps, a), load_rot(dumps, b))
        combo = None
        if len(steps) > 2:
            name = next(iter(steps[2]["candidates"]))
            combo = {"name": name, "wins": steps[2]["candidates"][name]["wins"]}
        res = decide(steps[0], steps[1], combo)
        if res["combo_needed"]:
            res["reference"] = head_to_head([a, b], dumps)
        print(res)
    else:
        res = compare(json.loads(Path(args.e2).read_text()), Path(args.dumps), args.ref, args.cand)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
