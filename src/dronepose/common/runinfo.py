"""Run metadata helpers shared by the training scripts (no ultralytics dependency)."""
import sys
from pathlib import Path


def read_commit(code_dir: Path) -> str:
    """Commit id from <code_dir>/COMMIT (written when the code is packaged), else "unknown"."""
    f = Path(code_dir) / "COMMIT"
    return f.read_text().strip() if f.exists() else "unknown"


def versions_and_gpu() -> tuple[dict, str | None]:
    import torch

    v = {"python": sys.version.split()[0], "torch": torch.__version__}
    try:
        import torchvision
        v["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    try:
        import albumentations
        v["albumentations"] = albumentations.__version__
    except ImportError:
        pass
    return v, (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
