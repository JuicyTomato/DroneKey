"""Overfit check (spec 6.2): train = val = the first n train crops (sorted by frame_id), no augmentation.
Passes when the final median keypoint error is < overfit.target_px (px of the 256 crop). Writes <out> (JSON with the
curve, final_px_median, dead_relu_frac = fraction of outputs exactly 0) and the training run in <out stem>_run/.

  python scripts/dronekey/overfit.py --crops-root data/processed/yolo_crops --variant paper --out experiments/dronekey/overfit_paper.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dronepose.keypoints.dronekey import train as T
from dronepose.keypoints.dronekey.config import load_config, resolve_variant
from dronepose.keypoints.dronekey.data import CropDataset
from dronepose.keypoints.dronekey.model import build_model

TINY = dict(d=16, n_layers=2, nhead=2, ffn=32)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops-root", default="data/processed/yolo_crops")
    ap.add_argument("--variant", default="paper")
    ap.add_argument("--n", type=int, default=None, help="default: overfit.n")
    ap.add_argument("--epochs", type=int, default=None, help="default: overfit.epochs")
    ap.add_argument("--batch", type=int, default=None, help="default: overfit.batch")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cr-softplus", action="store_true", help="softplus on the CR outputs (config default is now true; flag kept for compatibility)")
    ap.add_argument("--no-pretrained", action="store_true", help="random ResNet50 init (tests)")
    ap.add_argument("--tiny", action="store_true", help="TinyBackbone, d 16, 2 layers (tests only)")
    ap.add_argument("--out", required=True, help="result JSON, e.g. experiments/dronekey/overfit_paper.json")
    return ap.parse_args(argv)


@torch.no_grad()
def dead_relu_frac(model, ds, device, batch: int) -> float:
    """Fraction of output values that are exactly 0.0 over the dataset (model in eval)."""
    model.eval()
    zero, total = 0, 0
    for x, _, _ in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0):
        out = model(x.to(device))
        zero += int((out == 0).sum())
        total += out.numel()
    return zero / total


def main(argv=None) -> dict:
    args = parse_args(argv)
    t0 = time.time()
    cfg = load_config()
    oc = cfg["overfit"]
    n, epochs, batch = args.n or oc["n"], args.epochs or oc["epochs"], args.batch or oc["batch"]
    run_cfg = resolve_variant(cfg, args.variant)
    run_cfg["train"].update(lr=oc["lr"], batch=batch, epochs=epochs, workers=0)
    if args.cr_softplus:
        run_cfg["model"]["cr_softplus"] = True
    if args.tiny:
        run_cfg["model"].update(TINY)
        run_cfg["train"]["amp"] = False
    run_cfg.update(seed=0, name=f"overfit_{args.variant}", variant=args.variant, dataset="6dronesyn", recipe="none")

    device = T.pick_device(args.device)
    inp = run_cfg["input"]
    train_ds = CropDataset(args.crops_root, "train", "none", True, inp, seed=0, limit=n)
    val_ds = CropDataset(args.crops_root, "train", "none", False, inp, limit=n)  # the same n crops
    print(f"overfit {args.variant}: {len(train_ds)} crops, {epochs} epochs, lr {oc['lr']}, batch {batch}, "
          f"device {device}, cr_softplus {run_cfg['model'].get('cr_softplus', False)}", flush=True)
    T.seed_everything(0)  # head init reproducible; fit re-seeds fresh runs
    model = build_model(run_cfg["model"], pretrained=not args.no_pretrained, tiny=args.tiny)
    out = Path(args.out)
    hist = T.fit(run_cfg, model, train_ds, val_ds, out.with_name(out.stem + "_run"), device,
                 save_checkpoints=False)  # the check never resumes; last.pt would be ~0.9 GB
    final = hist[-1]["val_px_median"]
    res = {"variant": args.variant, "n": len(train_ds), "epochs": epochs, "lr": oc["lr"], "batch": batch,
           "cr_softplus": bool(run_cfg["model"].get("cr_softplus", False)),
           "curve": [r["val_px_median"] for r in hist], "final_px_median": final,
           "dead_relu_frac": dead_relu_frac(model, val_ds, device, batch),
           "passed": final < oc["target_px"], "seconds": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"final px median {final:.3f} (target < {oc['target_px']}), dead_relu_frac {res['dead_relu_frac']:.3f}, "
          f"passed {res['passed']} -> {out}", flush=True)
    return res


if __name__ == "__main__":
    main()
