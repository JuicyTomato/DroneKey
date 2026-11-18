"""Last measure of the sim-to-real recipe: the authors' test split of MAV6D (9 sequences, 4962 frames), once.

  .venv/bin/python scripts/real/final_test.py --frozen experiments/real/sim2real/frozen.json --smoke        # train frames only
  .venv/bin/python scripts/real/final_test.py --frozen experiments/real/sim2real/frozen.json --allow-test   # once

frozen.json fixes what is measured, before the first look: {"ref": yolo reference of the bootstrap, "support": [references
outside the seed groups], "detector": W, "yolo": {name: W}, "dk": {name: W}, "refs_yolo": {name: W}, "refs_dk": {name: W},
"refs_a_kpts": {name: KPTS of E2}}. Setting A: new models on the true box, references from their E2 export. Setting B:
everything through the new detector. Output: <out>/e2_{a,b}.json. The file is checked before the test is touched.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_authors import write_split  # noqa: E402

TEST_OUT = Path("experiments/real/sim2real/test")
SMOKE_OUT = Path("experiments/real/sim2real/smoke_test")
PY = ".venv/bin/python"
YOLO_PY = str(Path.home() / ".venvs/dronepose-yolo/bin/python")
FRAMES = "/Volumes/MakeMePasta/MAV6D/frames640"
SMOKE_FRAMES = 200


def guard(allow_test: bool, out: Path, smoke: bool) -> None:
    """--allow-test for the real run, which happens once; a smoke run stays away from its folder."""
    if smoke:
        if Path(out).resolve() == TEST_OUT.resolve():
            raise ValueError("--smoke must not write in the test folder")
        return
    if not allow_test:
        raise ValueError("the authors' test split is measured once: pass --allow-test")
    if Path(out).exists():
        raise ValueError(f"{out} exists: il test si misura una volta sola")


def check_frozen(frozen: dict) -> None:
    """A reference missing from the setting-A exports would drop out of a table that cannot be redone."""
    if "det_conf" not in frozen:
        raise ValueError("det_conf missing: the detector threshold is chosen on the authors' train, never on the test")
    refs = {**frozen["refs_yolo"], **frozen["refs_dk"]}
    if frozen["ref"] not in frozen["refs_yolo"]:
        raise ValueError(f"ref {frozen['ref']!r} is not in refs_yolo")
    missing = sorted(set(refs) - set(frozen["refs_a_kpts"]))
    if missing:
        raise ValueError(f"references without a setting-A export: {missing}")
    if set(frozen["support"]) - set(frozen["refs_yolo"]):
        raise ValueError(f"support models are not references: {frozen['support']}")


def missing_files(frozen: dict) -> list[str]:
    paths = [frozen["detector"]]
    for key in ("yolo", "dk", "refs_yolo", "refs_dk", "refs_a_kpts"):
        paths += frozen[key].values()
    return [p for p in paths if not Path(p).exists()]


def _yolo(name: str, weights: str, det: Path, setting: str, out: Path, gt: Path) -> list[str]:
    return [YOLO_PY, "scripts/export_two_stage.py", "--det", str(det), "--weights", weights, "--split", "mav6d",
            "--canonical", str(gt), "--out", str(out / "kpts" / f"kpts_{setting}_{name}.jsonl")]


def _dk(name: str, weights: str, det: Path, setting: str, out: Path, gt: Path) -> list[str]:
    return [PY, "scripts/dronekey/export_dronekey.py", "--weights", weights, "--variant", "paper", "--split", "mav6d",
            "--setting", "B", "--det", str(det), "--canonical", str(gt),
            "--out", str(out / "kpts" / f"kpts_{setting}_{name}.jsonl")]


def build_commands(frozen: dict, out: Path, split_dir: Path) -> list[list[str]]:
    """Detector, exports, then the two evaluations. Exports of setting A use the true box, as in E2."""
    gt, gtbox, det = split_dir / "gt.jsonl", split_dir / "gtbox.jsonl", out / "kpts" / "det_new.jsonl"
    new_y, new_d, ref_y, ref_d = frozen["yolo"], frozen["dk"], frozen["refs_yolo"], frozen["refs_dk"]
    cmds = [[YOLO_PY, "scripts/export_frames_c.py", "--weights", frozen["detector"], "--split", "mav6d", "--frames-dir", FRAMES,
             "--ids-from", str(gt), "--conf", str(frozen["det_conf"]), "--out", str(det)]]
    cmds += [_yolo(n, w, s, x, out, gt) for n, w in new_y.items() for s, x in ((gtbox, "a"), (det, "b"))]
    cmds += [_dk(n, w, s, x, out, gt) for n, w in new_d.items() for s, x in ((gtbox, "a"), (det, "b"))]
    cmds += [_yolo(n, w, det, "b", out, gt) for n, w in ref_y.items()]
    cmds += [_dk(n, w, det, "b", out, gt) for n, w in ref_d.items()]
    new, ref = {**new_y, **new_d}, {**ref_y, **ref_d}
    groups = []
    for name, members in (("yolo_new", new_y), ("dk_new", new_d), ("yolo_e2", [n for n in ref_y if n not in frozen["support"]]),
                          ("dk_e2", ref_d)):
        if members:
            groups += ["--group", f"{name}={','.join(members)}"]
    common = ["--canonical", str(gt), "--ref", frozen["ref"], "--det-metrics", f"New={det}", *groups]
    specs_a = [f"{n}={out / 'kpts' / f'kpts_a_{n}.jsonl'}" for n in new] + [f"{n}={p}" for n, p in frozen["refs_a_kpts"].items()]
    specs_b = [f"{n}={out / 'kpts' / f'kpts_b_{n}.jsonl'}" for n in [*new, *ref]]
    cmds.append([PY, "scripts/real/eval_e2.py", *common, "--out", str(out / "e2_a.json"), *specs_a])
    cmds.append([PY, "scripts/real/eval_e2.py", *common, "--class-from", f"New={det}", "--out", str(out / "e2_b.json"), *specs_b])
    return cmds


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_FRAMES} frames of the authors' train split, the test stays closed")
    args = ap.parse_args(argv)
    out = SMOKE_OUT if args.smoke else TEST_OUT
    guard(args.allow_test, out, args.smoke)
    frozen = json.loads(Path(args.frozen).read_text())
    check_frozen(frozen)
    if missing_files(frozen):
        raise FileNotFoundError(f"frozen.json: missing files {missing_files(frozen)}")
    split_dir = out / "split"
    (out / "kpts").mkdir(parents=True, exist_ok=True)
    if args.smoke:
        write_split("train", split_dir, limit=SMOKE_FRAMES)
    else:
        write_split("test", split_dir, allow_test=True)
    for cmd in build_commands(frozen, out, split_dir):
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
