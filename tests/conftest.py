import os
from pathlib import Path

import pytest


def _dataset_root():
    root = os.environ.get("SIXDRONESYN_ROOT")
    return Path(root) if root and Path(root).is_dir() else None


def pytest_collection_modifyitems(config, items):
    if _dataset_root() is not None:
        return
    skip = pytest.mark.skip(reason="SIXDRONESYN_ROOT non impostata o disco non collegato")
    for item in items:
        if "dataset" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def dataset_root():
    return _dataset_root()
