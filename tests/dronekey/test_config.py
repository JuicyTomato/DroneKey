import copy

import pytest

from dronepose.keypoints.dronekey.config import load_config, resolve_variant


def test_load_config_values():
    cfg = load_config()
    assert cfg["model"] == {"d": 1024, "n_layers": 6, "nhead": 8, "ffn": 2048, "dropout": 0.1,
                            "wiring": "paper", "head": "gated", "cr_softplus": True}
    assert cfg["loss"] == {"kind": "pose_adaptive", "alpha": 5, "D": 10, "eps": 1e-4}
    assert cfg["train"]["lr"] == 1e-5 and cfg["train"]["seeds"] == [0, 1, 2] and cfg["train"]["epochs"] == 100
    assert cfg["input"]["flip_idx"] == [1, 0, 3, 2] and cfg["input"]["mean"][0] == 0.485
    assert cfg["recipe"] == "final" and cfg["ladder_epochs"] == 20
    assert cfg["overfit"]["n"] == 256 and set(cfg["variants"]) == {
        "G0", "G1", "G2", "G3", "G4", "G5", "G6", "paper", "code"}


def test_resolve_variant_g0_and_unknown():
    cfg = load_config()
    snapshot = copy.deepcopy(cfg)
    v = resolve_variant(cfg, "G0")
    assert set(v) == {"model", "loss", "train", "input", "recipe"}
    assert v["model"]["head"] == "mean_linear" and v["loss"]["kind"] == "mse"
    assert v["model"]["d"] == 1024 and v["loss"]["alpha"] == 5  # untouched keys keep base values
    assert cfg == snapshot  # not mutated
    v["model"]["d"] = 1
    assert cfg["model"]["d"] == 1024
    with pytest.raises(KeyError):
        resolve_variant(cfg, "X")


def test_resolve_variant_recipe_and_wiring():
    cfg = load_config()
    assert resolve_variant(cfg, "G5")["recipe"] == "none"
    assert resolve_variant(cfg, "G4")["recipe"] == "final"
    assert resolve_variant(cfg, "G6")["model"]["wiring"] == "code"
    assert resolve_variant(cfg, "paper")["model"]["wiring"] == "paper"
    assert resolve_variant(cfg, "code")["model"]["wiring"] == "code"
    assert resolve_variant(cfg, "G3")["loss"]["kind"] == "gaussian"
