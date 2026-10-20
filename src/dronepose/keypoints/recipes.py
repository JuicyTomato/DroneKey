"""Training recipes for YOLO-Pose: model.train kwargs and pixel-only Albumentations (keypoints never move)."""
import inspect

_GEOM = {"fliplr": 0.5, "degrees": 30.0, "scale": 0.3, "translate": 0.1, "mosaic": 0.0,
         "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4}

# "final" keeps rotation, scale, translation and HSV of w1; only the Albumentations list changes
RECIPES: dict[str, dict] = {"w1": dict(_GEOM), "final": dict(_GEOM)}

_P = 0.3  # default probability of every pixel transform
_NOISE_STD = (0.01, 0.05)  # fraction of the 0-255 dynamic range


def _build(cls, new: dict, old: dict, **common):
    """Instantiate an Albumentations class with the 2.x (new) or 1.x (old) argument names, by constructor signature."""
    params = inspect.signature(cls.__init__).parameters
    kwargs = new if all(k in params for k in new) else old
    return cls(**kwargs, **common)


def albumentations_for(recipe: str, p: float | None = None) -> list | None:
    """Albumentations list for model.train(augmentations=...); None = Ultralytics default; p overrides every probability."""
    if recipe not in RECIPES:
        raise KeyError(recipe)
    if recipe == "w1":
        return None
    import albumentations as A

    p = _P if p is None else p
    var = (round((_NOISE_STD[0] * 255) ** 2, 4), round((_NOISE_STD[1] * 255) ** 2, 4))  # 1.x takes a variance
    return [
        _build(A.Downscale, {"scale_range": (0.25, 0.75)}, {"scale_min": 0.25, "scale_max": 0.75}, p=p),
        _build(A.GaussNoise, {"std_range": _NOISE_STD}, {"var_limit": var}, p=p),
        _build(A.ImageCompression, {"quality_range": (40, 95)}, {"quality_lower": 40, "quality_upper": 95}, p=p),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),
    ]
