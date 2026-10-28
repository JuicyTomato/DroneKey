"""Image-folder dataset for the crop classifier: <out>/<split>/<drone>/<frame_id>.jpg, symlinks to the YOLO crops or
copies (on Kaggle the input mount is slow per file: copy to local disk once, then every epoch reads locally)."""
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def drone_of_frame(frame_id: str) -> str:
    """Drone id of a frame id, e.g. '01_04_01_000_0001' -> '04'."""
    return frame_id.split("_")[1]


def build_imagefolder(crops_dir: Path, out: Path, splits: tuple[str, ...] = ("train", "val"),
                      limit: int | None = None, copy: bool = False, threads: int = 32) -> dict[str, int]:
    """Link (or `copy`) every crop listed in crops_<split>.jsonl (with `limit`: that many, evenly spaced, so that every
    drone appears in a short smoke run); returns the count per split."""
    crops_dir, out = Path(crops_dir), Path(out)
    counts = {}
    for split in splits:
        with open(crops_dir / f"crops_{split}.jsonl") as f:
            ids = [json.loads(line)["frame_id"] for line in f if line.strip()]
        if limit is not None:
            ids = ids[::max(1, len(ids) // limit)][:limit]
        def one(fid: str) -> None:
            src, dst = crops_dir / "images" / split / f"{fid}.jpg", out / split / drone_of_frame(fid) / f"{fid}.jpg"
            if dst.exists() or dst.is_symlink():
                return
            if copy:
                shutil.copyfile(src, dst)
            else:
                dst.symlink_to(src.resolve())

        for d in {drone_of_frame(f) for f in ids}:
            (out / split / d).mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(one, ids))
        counts[split] = len(ids)
    return counts
