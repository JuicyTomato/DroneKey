"""E1 table: pose, translation and keypoint metrics per subset of frames (all / no08 / 08 / breakdowns)."""
import numpy as np

from dronepose.eval.keypoint_metrics import ap_paper, frame_px_error, identity_ok, oks, oks_ap, pck, success_rate
from dronepose.eval.units import METERS_PER_UNIT, WORLD_PER_CAMERA, tilt_deg

_MISSING = np.full((4, 3), np.nan)
_TILT_BINS = (("tilt_0-15", 0.0, 15.0), ("tilt_15-30", 15.0, 30.0), ("tilt_30+", 30.0, np.inf))


def subsets(records: list[dict], breakdown: bool) -> dict[str, set[str]]:
    """Frame ids per subset: all, no08, 08; with `breakdown` also per drone, scene and GT tilt bin."""
    out: dict[str, set[str]] = {"all": {r["frame_id"] for r in records},
                                "no08": {r["frame_id"] for r in records if r["drone"] != "08"},
                                "08": {r["frame_id"] for r in records if r["drone"] == "08"}}
    if breakdown:
        for r in records:
            tilt = tilt_deg(np.array(r["R_gt"]))
            names = [f"drone_{r['drone']}", f"scene_{r['scene']}"]
            names += [n for n, lo, hi in _TILT_BINS if lo <= tilt < hi]
            for n in names:
                out.setdefault(n, set()).add(r["frame_id"])
    return out


def _med(xs: list[float]) -> float | None:
    return float(np.median(xs)) if len(xs) else None


def e1_metrics(results: list[dict], records_by_id: dict[str, dict], kpts_by_id: dict[str, np.ndarray],
               scores_by_id: dict[str, float] | None = None) -> dict:
    """Metrics of one subset: `results` are replay results; frames without keypoints fail PCK/SR/AP."""
    n = len(results)
    posed = [r for r in results if r["rot_err_deg"] is not None]
    kpts, gts, areas, diags = [], [], [], []
    for r in results:
        rec = records_by_id[r["frame_id"]]
        x0, y0, x1, y1 = rec["bbox_xyxy"]
        kpts.append(np.asarray(kpts_by_id.get(r["frame_id"], _MISSING), float))
        gts.append(np.array(rec["kpts2d"], float))
        areas.append((x1 - x0) * (y1 - y0))
        diags.append(float(np.hypot(x1 - x0, y1 - y0)))
    px = [e for e in (frame_px_error(p, g) for p, g in zip(kpts, gts)) if e is not None]
    ident = [i for i in (identity_ok(p, g) for p, g in zip(kpts, gts)) if i is not None]
    fixed = pck(kpts, gts, [2.0, 5.0, 10.0])
    rel = pck(kpts, gts, [0.02, 0.05], diags)
    oks_v, scores = [], []
    for r, p, g, a in zip(results, kpts, gts, areas):
        finite = bool(np.isfinite(p[:, :2]).all())
        oks_v.append(oks(p, g, a))
        if not finite:
            scores.append(-np.inf)
        elif scores_by_id is not None:
            scores.append(float(scores_by_id.get(r["frame_id"], 0.0)))
        else:
            scores.append(float(np.mean(p[:, 2])) if p.shape[1] > 2 else 1.0)
    t_err = [r["t_err"] for r in posed]
    meters = [r["t_err"] * METERS_PER_UNIT[r["frame_id"].split("_")[1]] for r in posed
              if r["frame_id"].split("_")[1] in METERS_PER_UNIT]
    rot = [r["rot_err_sym_deg"] for r in posed]
    t_cam = _med(t_err)
    return {
        "n_frames": n,
        "coverage": len(posed) / n if n else None,
        "median_rot_sym": _med(rot),
        "p90_rot_sym": float(np.percentile(rot, 90)) if rot else None,
        "median_t_camera": t_cam,
        "median_t_world": None if t_cam is None else t_cam * WORLD_PER_CAMERA,
        "median_t_meters": _med(meters),
        "flip_rate": float(np.mean([r["rot_err_deg"] > 90 for r in posed])) if posed else None,
        "px_median": _med(px),
        "px_p90": float(np.percentile(px, 90)) if px else None,
        "pck_2px": fixed[2.0], "pck_5px": fixed[5.0], "pck_10px": fixed[10.0],
        "pck_0.02diag": rel[0.02], "pck_0.05diag": rel[0.05],
        "identity": float(np.mean(ident)) if ident else None,
        "oks_ap": oks_ap(oks_v, scores),
        "ap_paper": ap_paper(oks_v),
        "sr90": success_rate(oks_v, 0.90), "sr95": success_rate(oks_v, 0.95),
    }


def e1_table(results: list[dict], records: list[dict], kpts_by_id: dict[str, np.ndarray],
             scores_by_id: dict[str, float] | None = None, breakdown: bool = False) -> dict[str, dict]:
    """`e1_metrics` for every subset of `subsets(records, breakdown)`."""
    by_id = {r["frame_id"]: r for r in records}
    return {name: e1_metrics([r for r in results if r["frame_id"] in ids], by_id, kpts_by_id, scores_by_id)
            for name, ids in subsets(records, breakdown).items()}


def mean_std(tables: list[dict[str, dict]]) -> dict[str, dict]:
    """Per row and numeric metric: mean and sample std (ddof=1) across tables; None if a value is missing."""
    out: dict[str, dict] = {}
    for row in tables[0]:
        out[row] = {}
        for metric, first in tables[0][row].items():
            if not isinstance(first, (int, float)) and first is not None:
                continue
            vals = [t[row][metric] for t in tables]
            if any(v is None for v in vals):
                out[row][metric] = {"mean": None, "std": None}
            else:
                out[row][metric] = {"mean": float(np.mean(vals)),
                                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
    return out
