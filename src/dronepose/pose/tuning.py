"""Parameter selection for the global disambiguation: pure counting and selection logic (no I/O, no replay)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Counts:
    flips: int      # accepted frames with rotation error > 90 deg
    posed: int      # frames with a pose
    frames: int     # all frames

    @property
    def flip_rate(self) -> float:
        return self.flips / self.posed if self.posed else 0.0

    @property
    def coverage(self) -> float:
        return self.posed / self.frames if self.frames else 0.0


def scene_of(frame_id: str) -> str:
    return frame_id.split("_")[0]


def add_counts(items: list[Counts]) -> Counts:
    return Counts(sum(c.flips for c in items), sum(c.posed for c in items), sum(c.frames for c in items))


def counts_from_results(results_by_seq: dict[str, list[dict]]) -> dict[str, Counts]:
    """{'all': ..., 'no08': ...}. The Tello (drone 08) is reported apart: its errors come from keypoint identity
    confusion, which no choice between the two IPPE solutions can fix, so it would drown the signal."""
    out = {}
    for name, keep in (("all", lambda d: True), ("no08", lambda d: d != "08")):
        rs = [r for key, seq in results_by_seq.items() if keep(key.split("_")[1]) for r in seq]
        posed = [r for r in rs if r["rot_err_deg"] is not None]
        out[name] = Counts(sum(r["rot_err_deg"] > 90 for r in posed), len(posed), len(rs))
    return out


def select_best(rows: list[dict], min_coverage: float = 0.95) -> tuple[dict, bool]:
    """rows: {'params': {'reject_cost', 'z_cap', ...}, 'counts': Counts}. Rule: among rows with coverage >= min_coverage
    the lowest flip rate; ties: higher coverage, then lower reject_cost, then lower z_cap. If no row reaches the
    coverage floor, the row with the highest coverage (second value False)."""
    if not rows:
        raise ValueError("no rows to choose from")
    key = lambda r: (r["counts"].flip_rate, -r["counts"].coverage, r["params"]["reject_cost"], r["params"]["z_cap"])
    enough = [r for r in rows if r["counts"].coverage >= min_coverage]
    if enough:
        return min(enough, key=key), True
    return min(rows, key=lambda r: (-r["counts"].coverage, r["counts"].flip_rate, r["params"]["reject_cost"],
                                    r["params"]["z_cap"])), False
