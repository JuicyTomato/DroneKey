"""Image-folder dataset for the crop classifier: symlinks <out>/<split>/<drone>/<frame_id>.jpg to the YOLO crops."""
import json
from pathlib import Path


def drone_of_frame(frame_id: str) -> str:
    """Drone id of a frame id, e.g. '01_04_01_000_0001' -> '04'."""
    return frame_id.split("_")[1]


def build_imagefolder(crops_dir: Path, out: Path, splits: tuple[str, ...] = ("train", "val"),
                      limit: int | None = None) -> dict[str, int]:
    """Link every crop listed in crops_<split>.jsonl (with `limit`: that many, evenly spaced, so that every drone
    appears in a short smoke run); returns the count per split."""
    crops_dir, out = Path(crops_dir), Path(out)
    counts = {}
    for split in splits:
        with open(crops_dir / f"crops_{split}.jsonl") as f:
            ids = [json.loads(line)["frame_id"] for line in f if line.strip()]
        if limit is not None:
            ids = ids[::max(1, len(ids) // limit)][:limit]
        for fid in ids:
            dst = out / split / drone_of_frame(fid) / f"{fid}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_symlink():
                dst.symlink_to((crops_dir / "images" / split / f"{fid}.jpg").resolve())
        counts[split] = len(ids)
    return counts
