"""MAV6D records of one authors' split, so that later steps cannot read the other one.

  .venv/bin/python scripts/real/split_authors.py --which train --out-dir experiments/real/sim2real/authors_train

The test split is measured once, at the end: it needs --allow-test, and scripts/real/final_test.py is the only caller.
"""
import argparse
import json
from pathlib import Path

GT = Path("data/processed/mav6d_gt.jsonl")
GTBOX = Path("data/processed/mav6d_gtbox.jsonl")


def filter_rows(rows: list[str], ids: set[str]) -> list[str]:
    return [line for line in rows if json.loads(line)["frame_id"] in ids]


def write_split(which: str, out_dir: Path, allow_test: bool = False, gt: Path = GT, gtbox: Path = GTBOX,
                limit: int | None = None) -> None:
    if which == "test" and not allow_test:
        raise ValueError("the authors' test split is measured once: pass --allow-test (final_test.py only)")
    rows = [line for line in Path(gt).read_text().splitlines(keepends=True) if line.strip()]
    keep = [line for line in rows if json.loads(line)["authors_split"] == which][:limit]
    ids = {json.loads(line)["frame_id"] for line in keep}
    boxes = [line for line in Path(gtbox).read_text().splitlines(keepends=True) if line.strip()]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gt.jsonl").write_text("".join(keep))
    (out_dir / "gtbox.jsonl").write_text("".join(filter_rows(boxes, ids)))
    print(f"{len(keep)} frames, {which} -> {out_dir}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", required=True, choices=["train", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    write_split(args.which, Path(args.out_dir), args.allow_test, limit=args.limit)


if __name__ == "__main__":
    main()
