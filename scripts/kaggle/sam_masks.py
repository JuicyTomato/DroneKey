"""SAM on the crops of build_sam_inputs.py: box around the propellers and the 4 hubs as positive points. Resumable.

Kaggle (GPU, Internet On):  pip install -q ultralytics==8.4.178
    python -u $CODE/scripts/kaggle/sam_masks.py --inputs $INPUTS --out /kaggle/working/sam_out --model sam2.1_s.pt
An empty mask writes no PNG and adds a line to failed.txt.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

BOX_PAD = 0.25


def prompts(kpts: list, size: int) -> tuple[list[float], list[list[float]], list[int]]:
    """(box xyxy, points, labels): hub extent plus BOX_PAD * larger extent, clipped to the crop."""
    k = np.array(kpts, float)
    lo, hi = k.min(0), k.max(0)
    pad = BOX_PAD * float(max(hi - lo))
    box = np.r_[np.clip(lo - pad, 0, size), np.clip(hi + pad, 0, size)]
    return box.tolist(), k.tolist(), [1] * len(k)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sam2.1_s.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--box-only", action="store_true", help="no point prompts")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    from ultralytics import SAM

    inputs, out = Path(args.inputs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (inputs / "index.jsonl").read_text().splitlines() if l.strip()][:args.limit]
    model = SAM(args.model)
    t0, done, failed = time.time(), 0, []
    for i, r in enumerate(rows, 1):
        path = out / f"{r['geom']}.png"
        if not path.exists():
            box, pts, labels = prompts(r["kpts"], r["size"])
            kw = {} if args.box_only else {"points": [pts], "labels": [labels]}
            res = model.predict(str(inputs / "images" / f"{r['geom']}.jpg"), bboxes=[box], device=args.device,
                                verbose=False, **kw)[0]
            m = None if res.masks is None else res.masks.data.cpu().numpy().max(0)
            if m is None or not m.any():
                failed.append(r["geom"])
            else:
                cv2.imwrite(str(path), (m > 0.5).astype(np.uint8) * 255)
                done += 1
        if i % 500 == 0:
            print(f"{i}/{len(rows)} {time.time() - t0:.0f} s", flush=True)
    (out / "failed.txt").write_text("".join(g + "\n" for g in failed))
    print(f"{done} masks, {len(failed)} failed, {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
