import pytest

torch = pytest.importorskip("torch")

from dronepose.forecast.nets.models import ARMS, TrajNet, count_params

B, N, H = 4, 15, 30


def _inputs(scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, 9, generator=g)
    bases = {k: scale * torch.randn(B, H, 3, generator=g) for k in ("cv", "phys", "phys_mode", "kf_ca")}
    mc = torch.zeros(B, 10)
    mc[:, 1] = 1
    mc[:, 3 + 2] = 1
    return x, bases, mc


@pytest.mark.parametrize("arm", ARMS)
def test_output_shapes_all_arms(arm):
    x, bases, _ = _inputs()
    out = TrajNet(arm)(x, bases)
    assert out["pred"].shape == (B, H, 3)
    if arm == "R-F-P":
        assert out["w"].shape == (B,) and bool(((out["w"] >= 0) & (out["w"] <= 1)).all())
    else:
        assert out["w"] is None


def test_residual_starts_at_base():
    x, bases, _ = _inputs()
    assert torch.equal(TrajNet("R-CV")(x, bases)["pred"], bases["cv"])
    assert torch.equal(TrajNet("R-F")(x, bases)["pred"], bases["phys"])
    assert torch.equal(TrajNet("G")(x, bases)["pred"], torch.zeros(B, H, 3))
    p = TrajNet("R-F-P")(x, bases)
    torch.testing.assert_close(p["pred"], 0.5 * bases["phys"] + 0.5 * bases["kf_ca"])


def test_mode_variant_uses_mode_base():
    x, bases, mc = _inputs()
    net = TrajNet("R-F", use_mode=True)
    assert torch.equal(net(x, bases, mc)["pred"], bases["phys_mode"])
    with pytest.raises(ValueError):
        net(x, bases)                                                   # the variant needs mode and class


def test_mode_input_changes_output_after_training_step():
    x, bases, mc = _inputs()
    net = TrajNet("R-F", use_mode=True)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.1)
    assert not torch.allclose(net(x, bases, mc)["pred"], net(x, bases, torch.zeros_like(mc))["pred"])


def test_lstm_variant_runs():
    x, bases, _ = _inputs()
    net = TrajNet("R-F", rnn="lstm")
    assert isinstance(net.encoder, torch.nn.LSTM)
    assert net(x, bases)["pred"].shape == (B, H, 3)


def test_param_count():
    n = count_params(TrajNet("R-F"))
    assert 100_000 <= n <= 250_000


def test_forward_finite_on_extreme_bases():
    x, bases, _ = _inputs(scale=100.0)
    for arm in ARMS:
        net = TrajNet(arm)
        for p in net.parameters():
            torch.nn.init.normal_(p, std=0.1)
        out = net(x, bases)
        assert torch.isfinite(out["pred"]).all(), arm


def test_unknown_arm_refused():
    with pytest.raises(ValueError):
        TrajNet("R-X")
