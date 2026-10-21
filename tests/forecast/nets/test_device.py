import pytest

torch = pytest.importorskip("torch")

from dronepose.forecast.nets.device import load_nets_config, pick_device, seed_everything


def test_pick_device_cpu_forced():
    assert pick_device("cpu").type == "cpu"


def test_pick_device_auto_is_mps_or_cpu():
    assert pick_device().type in ("mps", "cpu")


def test_seed_everything_reproducible():
    seed_everything(3)
    a = torch.rand(4)
    seed_everything(3)
    assert torch.equal(a, torch.rand(4))


def test_config_values():
    c = load_nets_config()
    assert c["hidden"] == 128 and c["layers"] == 2 and c["batch"] == 512 and c["K"] == 8
    assert c["seeds"] == [0, 1, 2] and c["sizes"] == {"train_big": 50000, "select": 2000}
