"""Composer inside Ultralytics: train images are composed when loaded, val images never.

The class swap happens after the dataset is built, so `cache="ram"` holds the original images and every epoch composes a
fresh copy.
"""
from pathlib import Path

import numpy as np
from ultralytics.data import YOLODataset
from ultralytics.models.yolo.pose import PoseTrainer


class CompositeDataset(YOLODataset):
    composer = None

    def load_image(self, i, rect_mode=True, resize_short=False):
        im, hw0, hw = super().load_image(i, rect_mode, resize_short)
        rng = np.random.default_rng(np.random.randint(2**31 - 1))  # seeded per worker by Ultralytics
        return self.composer(im.copy(), Path(self.im_files[i]).stem, rng), hw0, hw


def make_trainer(composer) -> type[PoseTrainer]:
    class CompositePoseTrainer(PoseTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                ds.__class__ = CompositeDataset
                ds.composer = composer
            return ds

    return CompositePoseTrainer


def train_kwargs(composer) -> dict:
    """Extra kwargs of model.train, for fresh runs and resumes alike: a resume without them would drop the composition."""
    return {} if composer is None else {"trainer": make_trainer(composer)}
