"""Grid of drone-class pipelines (appearance x time x geometry), leave-one-scene-out tuning and the selection rule
of spec 2026-10-27-classe-drone-design.md (section 6)."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Callable

from dronepose.eval.class_eval import scene_of

ASPECTS = ("C1", "CM", "K", "K+CM")
W_G = (0.1, 0.3, 1.0, 3.0, 10.0)
SIGMA = (0.5, 1.0, 2.0)
A_MIN = (0.01, 0.1)
IOU_MIN = (0.5, 0.7)
W_CM = (0.3, 1.0, 3.0)


@dataclass(frozen=True)
class Cell:
    aspect: str     # "none" (geometry only), C1, CM, K, K+CM
    temporal: str   # F (frame), V (today's majority), W (sum of linear scores), L (sum of log scores)
    geom: bool

    @property
    def name(self) -> str:
        if self.aspect == "none":
            return f"G+{self.temporal}"
        return f"{self.aspect}+{self.temporal}" + ("+g" if self.geom else "")

    @property
    def cost(self) -> int:
        """0 = no extra trained model (geometry measured at 1.05 ms/frame on M1, under the 2 ms threshold); 1 = K."""
        return int("K" in self.aspect)

    @property
    def n_params(self) -> int:
        return len(param_grid(self)[0])


GRID: list[Cell] = ([Cell("C1", "V", False)]
                    + [Cell(a, t, g) for a in ASPECTS for t in ("F", "W", "L") for g in (False, True)]
                    + [Cell("none", "F", True), Cell("none", "L", True)])


def _geom_grid(with_weight: bool) -> list[dict]:
    weights = W_G if with_weight else (1.0,)
    out = [{"w_g": w, "form": "log"} for w in weights]
    out += [{"w_g": w, "form": "quad", "sigma": s} for w in weights for s in SIGMA]
    return out if with_weight else [{k: v for k, v in p.items() if k != "w_g"} for p in out]


def param_grid(cell: Cell) -> list[dict]:
    """Tuning grid of a cell (fixed in the plan); [{}] when nothing is tuned."""
    axes: list[list[dict]] = []
    if "CM" in cell.aspect:
        axes.append([{"iou_min": v} for v in IOU_MIN])
    if cell.aspect == "K+CM":
        axes.append([{"w_cm": v} for v in W_CM])
    if cell.temporal == "L" and cell.aspect != "none":
        axes.append([{"a_min": v} for v in A_MIN])
    if cell.geom:
        axes.append(_geom_grid(with_weight=cell.aspect != "none"))
    return [{k: v for d in combo for k, v in d.items()} for combo in product(*axes)] if axes else [{}]


def _correct_by_scene(maps: dict[int, dict[str, str]], records: list[dict]) -> dict[str, list[int]]:
    """Per scene: [correct, total] over the frames of every seed."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for cls_map in maps.values():
        for r in records:
            c = out[scene_of(r["frame_id"])]
            c[0] += cls_map[r["frame_id"]] == r["drone"]
            c[1] += 1
    return out


def tune_loso(build: Callable[[dict], dict[int, dict[str, str]]], records: list[dict],
              grid: list[dict]) -> dict[str, dict]:
    """Per held-out scene: the parameters with the best frame accuracy on the other scenes (seeds pooled); ties go
    to the first parameters of the grid. `build(params)` returns {seed: class map}."""
    if not grid:
        raise ValueError("empty tuning grid")
    scenes = sorted({scene_of(r["frame_id"]) for r in records})
    counts = [_correct_by_scene(build(p), records) for p in grid]
    out = {}
    for s in scenes:
        def acc(c):
            ok = sum(c[t][0] for t in c if t != s)
            n = sum(c[t][1] for t in c if t != s)
            return ok / n if n else 0.0
        accs = [acc(c) for c in counts]
        out[s] = grid[accs.index(max(accs))]
    return out


def tune_all(build: Callable[[dict], dict[int, dict[str, str]]], records: list[dict], grid: list[dict]) -> dict:
    """Parameters with the best frame accuracy on all scenes (seeds pooled): frozen before any later test pass."""
    if not grid:
        raise ValueError("empty tuning grid")
    accs = []
    for p in grid:
        c = _correct_by_scene(build(p), records)
        accs.append(sum(v[0] for v in c.values()) / sum(v[1] for v in c.values()))
    return grid[accs.index(max(accs))]


def loso_maps(build: Callable[[dict], dict[int, dict[str, str]]], records: list[dict],
              chosen: dict[str, dict]) -> dict[int, dict[str, str]]:
    """Class map per seed where each scene uses the parameters of its own fold."""
    out: dict[int, dict[str, str]] = defaultdict(dict)
    cache: dict[str, dict[int, dict[str, str]]] = {}
    for s, params in chosen.items():
        key = repr(sorted(params.items()))
        maps = cache.setdefault(key, build(params))
        for seed, cls_map in maps.items():
            out[seed].update({r["frame_id"]: cls_map[r["frame_id"]] for r in records if scene_of(r["frame_id"]) == s})
    return dict(out)


def choose_cell(acc: dict[str, list[float]], boots_vs: Callable[[str], dict[str, list[tuple[float, float, float]]]],
                cells: dict[str, Cell], pose_ok: dict[str, bool]) -> str:
    """Selection rule (spec 6): best mean accuracy among cells that pass the pose constraint; its ties (zero inside
    the paired interval in at least one seed) at the lowest cost class; the best of those; a residual exact tie goes
    to the cell without geometry, then to the one with fewer tuned parameters. `boots_vs(ref)` gives, per cell, the
    per-seed (diff, lo, hi) of its accuracy minus that of `ref`."""
    mean = {n: sum(v) / len(v) for n, v in acc.items() if pose_ok.get(n, False)}
    if not mean:
        raise ValueError("no cell passes the pose constraint")
    best = max(mean, key=mean.get)
    boots = boots_vs(best)
    ties = [n for n in mean if n == best or any(lo <= 0.0 <= hi for _, lo, hi in boots[n])]
    cheapest = min(cells[n].cost for n in ties)
    pool = [n for n in ties if cells[n].cost == cheapest]
    top = max(mean[n] for n in pool)
    return min((n for n in pool if mean[n] == top), key=lambda n: (cells[n].geom, cells[n].n_params, n))


def pose_ok_from(boots_rot: list[tuple[float, float, float]], flip: float, flip_base: float,
                 max_flip_increase: float = 0.005) -> bool:
    """Pose constraint (spec 6): the median rotation is not significantly worse than the base (interval above zero
    in every seed) and the flip rate without drone 08 grows by at most 0.5 points."""
    worse = all(lo > 0.0 for _, lo, _ in boots_rot)
    return not worse and flip <= flip_base + max_flip_increase + 1e-12
