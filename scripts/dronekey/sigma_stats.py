"""Percentiles of the min/max eigenvalue of the keypoint covariance Sigma (crop-normalized coordinates) and the
fraction of min eigenvalue below eps -> experiments/dronekey/sigma_stats_<crops-root name>.json."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dronepose.keypoints.dronekey.data import read_keypoints
from dronepose.keypoints.dronekey.loss import keypoint_covariance

PCTS = (1, 5, 50, 95, 99)


def sigma_stats(kpts: np.ndarray, eps: float) -> dict:
    """kpts: [N, 4, 2] normalized to the crop."""
    ev = torch.linalg.eigvalsh(keypoint_covariance(torch.as_tensor(kpts, dtype=torch.float64))).numpy()  # ascending
    lo, hi = ev[:, 0], ev[:, 1]
    return {"n": int(len(ev)), "eps": eps,
            "min_eig": {f"p{p}": float(np.percentile(lo, p)) for p in PCTS},
            "max_eig": {f"p{p}": float(np.percentile(hi, p)) for p in PCTS},
            "frac_min_below_eps": float((lo < eps).mean())}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--config", default="configs/dronekey.json")
    ap.add_argument("--out-dir", default="experiments/dronekey")
    a = ap.parse_args(argv)
    if a.split == "test" and not a.allow_test:
        raise ValueError("test split is locked: pass --allow-test for the final evaluation only")
    root = Path(a.crops_root)
    eps = json.loads(Path(a.config).read_text())["loss"]["eps"]
    labels = sorted((root / "labels" / a.split).glob("*.txt"))
    res = {"crops_root": str(root), "split": a.split,
           **sigma_stats(np.stack([read_keypoints(p) for p in labels]), eps)}
    out = Path(a.out_dir) / f"sigma_stats_{root.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
