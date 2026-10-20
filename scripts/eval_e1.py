"""E1 evaluation: replay `global` on saved keypoints and print/save the metrics table per model.

Usage:
  python scripts/eval_e1.py [--split val|test] [--allow-test] [--breakdown] [--ref NAME]
      [--group NAME=N1,N2,N3] --out OUT.json SPEC...
SPEC = NAME=KPTS.jsonl[,det=DETS.jsonl][,cls=gt|frame|causal]
  det: JSONL with boxes (box_conf becomes the AP score); cls: drone class given to the 3D model
  (gt = true drone, frame = detected class, causal = running majority vote; frame/causal need det).
"""
import argparse
import json
from pathlib import Path

from dronepose.data.canonical import read_jsonl
from dronepose.data.drone_models import load_drone_models
from dronepose.eval.bootstrap import by_trajectory, paired_bootstrap
from dronepose.eval.e1 import e1_table, mean_std
from dronepose.keypoints.detections import class_causal_vote, class_per_frame, load_detections
from dronepose.pose.replay import check_no_test, load_keypoints, replay

ROWS = ("all", "no08", "08")
SHOWN = ("n_frames", "coverage", "median_rot_sym", "p90_rot_sym", "median_t_camera", "median_t_meters",
         "flip_rate", "px_median", "px_p90", "pck_5px", "identity", "oks_ap", "sr90")


def parse_spec(spec: str) -> dict:
    """Parse `NAME=KPTS.jsonl[,det=...][,cls=...]` into {name, kpts, det, cls}."""
    head, *opts = spec.split(",")
    name, sep, kpts = head.partition("=")
    if not sep or not name or not kpts:
        raise ValueError(f"bad spec {spec!r}: expected NAME=KPTS.jsonl[,det=DETS.jsonl][,cls=gt|frame|causal]")
    out = {"name": name, "kpts": kpts, "det": None, "cls": "gt"}
    for o in opts:
        k, _, v = o.partition("=")
        if k not in ("det", "cls") or not v:
            raise ValueError(f"bad option {o!r} in spec {spec!r}")
        out[k] = v
    if out["cls"] not in ("gt", "frame", "causal"):
        raise ValueError(f"cls must be gt, frame or causal, got {out['cls']!r}")
    if out["cls"] != "gt" and out["det"] is None:
        raise ValueError(f"spec {name}: cls={out['cls']} needs det=")
    return out


def _fmt(v) -> str:
    return "  -  " if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def run(specs: list[str], split: str = "val", allow_test: bool = False, breakdown: bool = False,
        ref: str | None = None, groups: list[str] | None = None, records: list[dict] | None = None,
        canonical: Path | None = None, models_path: Path = Path("configs/drone_models.json")) -> dict:
    """Evaluate every spec on `split` and return {models, bootstrap, groups}; rejects test records without allow_test."""
    parsed = [parse_spec(s) for s in specs]
    names = [p["name"] for p in parsed]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate model names: {names}")
    if records is None:
        records = read_jsonl(canonical or Path(f"data/processed/canonical_{split}.jsonl"))
    check_no_test(records, allow_test)
    models = load_drone_models(models_path)
    out: dict = {"split": split, "models": {}, "bootstrap": {}, "groups": {}}
    flat: dict[str, list[dict]] = {}
    for p in parsed:
        kpts = load_keypoints(Path(p["kpts"]))
        dets = load_detections(Path(p["det"])) if p["det"] else None
        model_drone = None
        if p["cls"] != "gt":
            model_drone = (class_per_frame if p["cls"] == "frame" else class_causal_vote)(records, dets)
        res = replay(records, kpts, models, "global", model_drone=model_drone)
        flat[p["name"]] = [r for rs in res.values() for r in rs]
        scores = {fid: d.conf for fid, d in dets.items()} if dets else None
        out["models"][p["name"]] = {"spec": p, "table": e1_table(flat[p["name"]], records, kpts, scores, breakdown)}

    for name, m in out["models"].items():
        for row in ROWS:
            print(f"{name:8s} {row:5s} " + "  ".join(f"{k}={_fmt(m['table'][row][k])}" for k in SHOWN))

    ref = ref or names[0]
    if ref not in flat:
        raise ValueError(f"--ref {ref!r} is not one of {names}")
    if len(names) > 1:
        print(f"\npaired bootstrap vs {ref} (all frames, trajectories as units, 95%)")
        for key, label in (("rot_err_sym_deg", "rot"), ("t_err", "t")):
            bs = paired_bootstrap({n: by_trajectory(flat[n], key) for n in names}, ref)
            out["bootstrap"][label] = {n: {"diff": d, "lo": lo, "hi": hi} for n, (d, lo, hi) in bs.items()}
            for n, (d, lo, hi) in bs.items():
                if n != ref:
                    print(f"{n:8s} {label:3s} diff {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    for g in groups or []:
        gname, _, members = g.partition("=")
        members = members.split(",")
        missing = [m for m in members if m not in out["models"]]
        if not gname or missing:
            raise ValueError(f"bad --group {g!r}: expected NAME=N1,N2,N3 with known models (unknown: {missing})")
        ms = mean_std([out["models"][m]["table"] for m in members])
        out["groups"][gname] = {"members": members, "mean_std": ms}
        print(f"\ngroup {gname} ({len(members)} models), mean ± std")
        for row in ROWS:
            print(f"{gname:8s} {row:5s} " + "  ".join(
                f"{k}={_fmt(ms[row][k]['mean'])}±{_fmt(ms[row][k]['std'])}" for k in SHOWN if k != "n_frames"))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", metavar="SPEC")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--ref", default=None, help="reference model for the bootstrap (default: first SPEC)")
    ap.add_argument("--group", action="append", default=[], metavar="NAME=N1,N2,N3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = run(args.specs, args.split, args.allow_test, args.breakdown, args.ref, args.group)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
