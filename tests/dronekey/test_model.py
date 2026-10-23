import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dronepose.keypoints.dronekey.model import DroneKeyNet, TinyBackbone, build_model, count_params  # noqa: E402

D, N = 32, 3


def tiny(wiring="paper", head="gated", d=D, n_layers=N, **kw) -> DroneKeyNet:
    torch.manual_seed(0)
    return DroneKeyNet(d=d, n_layers=n_layers, nhead=8, ffn=2048, dropout=0.1, wiring=wiring, head=head,
                       pretrained=False, backbone=TinyBackbone(16), backbone_channels=16, **kw).eval()


def imgs(b=2):
    torch.manual_seed(1)
    return torch.randn(b, 3, 224, 224)


def test_tiny_backbone_gives_7x7_map():
    assert TinyBackbone(16)(imgs(1)).shape == (1, 16, 7, 7)


def test_paper_wiring_matches_encoder_hooks():
    m = tiny("paper")
    x = imgs()
    outs = []
    hooks = [layer.register_forward_hook(lambda mod, inp, out: outs.append(out))
             for layer in m.transformer_encoder.layers]
    with torch.no_grad():
        tokens = m.tokens(x)
        m.transformer_encoder(tokens)  # stacked encoder: intermediate outputs captured by the hooks
        for h in hooks:
            h.remove()
        x_ir, x_cr, gate = m.paper_terms(x)
    assert len(outs) == N
    for l in range(N):
        assert torch.allclose(x_ir[l], outs[l].mean(dim=0), atol=1e-5)
        assert torch.allclose(x_cr[l], m.compressed_representor(outs[l].mean(dim=0)), atol=1e-5)
    assert torch.allclose(gate, F.softmax(m.gate_generator(outs[-1].mean(dim=0)), dim=-1), atol=1e-5)


# --- Literal copy of the authors' KeypointModel (kkanuseobin/DroneKey, train.py), adapted ONLY to take a
# --- tiny backbone and a small d (the ResNet50 construction lines are replaced by the injected backbone).
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(_PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class _UpstreamKeypointModel(nn.Module):
    def __init__(self, num_keypoints, backbone, backbone_channels, d_model=1024, nhead=8, num_layers=6):
        super(_UpstreamKeypointModel, self).__init__()
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.resnet = backbone  # adapted: was resnet50(pretrained=True) without avgpool/fc
        self.feature_projection = nn.Conv2d(backbone_channels, d_model, kernel_size=1)  # adapted: was 2048
        self.pos_encoder = _PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.compressed_representor = nn.Linear(d_model, num_keypoints * 2)
        self.gate_generator = nn.Linear(d_model, num_layers)

    def forward(self, x):
        x = self.resnet(x)
        b, c, h, w = x.size()
        x = self.feature_projection(x)
        x = x.view(b, self.d_model, -1).permute(2, 0, 1)
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        compressed_representations = []
        for i, layer in enumerate(self.transformer_encoder.layers):
            layer_output, _ = layer.self_attn(x, x, x, need_weights=False)
            keypoint_rep = layer_output.mean(dim=0)
            compressed_rep = F.softplus(self.compressed_representor(keypoint_rep))
            compressed_representations.append(compressed_rep)
            if i == len(self.transformer_encoder.layers) - 1:
                gated_weights = F.softmax(self.gate_generator(keypoint_rep), dim=-1)
        compressed_representations = torch.stack(compressed_representations, dim=0)
        gated_weights = gated_weights.unsqueeze(2)
        compressed_representations = compressed_representations.permute(1, 0, 2)
        gated_sum_result = torch.sum(compressed_representations * gated_weights, dim=1)
        final_keypoints = torch.relu(gated_sum_result)
        return final_keypoints


def test_code_wiring_matches_train_py_forward():
    m = tiny("code")
    up = _UpstreamKeypointModel(4, TinyBackbone(16), 16, d_model=D, nhead=8, num_layers=N).eval()
    up.load_state_dict(m.state_dict())  # strict: same parameter names and shapes
    x = imgs()
    with torch.no_grad():
        assert torch.allclose(m(x), up(x), atol=1e-5)
    assert (m(x) > 0).any()  # non-trivial comparison


@pytest.mark.parametrize("l", [0, 1, 2])
def test_one_hot_gate_selects_layer(l):
    m = tiny("paper")
    with torch.no_grad():
        m.gate_generator.weight.zero_()
        m.gate_generator.bias.zero_()
        m.gate_generator.bias[l] = 60.0
        x = imgs()
        _, x_cr, gate = m.paper_terms(x)
        y = m(x)
    assert torch.allclose(gate[:, l], torch.ones(2), atol=1e-6)
    assert torch.allclose(y, torch.relu(x_cr[l]), atol=1e-5)
    other = torch.relu(x_cr[(l + 1) % N])
    assert not torch.allclose(y, other, atol=1e-5)


@pytest.mark.parametrize("wiring", ["paper", "code"])
def test_gate_positive_sums_to_one(wiring):
    m = tiny(wiring)
    with torch.no_grad():
        _, aux = m.forward_aux(imgs(3))
    g = aux["gate"]
    assert g.shape == (3, N) and (g > 0).all()
    assert torch.allclose(g.sum(dim=1), torch.ones(3), atol=1e-6)


@pytest.mark.parametrize("wiring,head", [("paper", "gated"), ("code", "gated"), ("paper", "last_layer"),
                                         ("paper", "mean_linear")])
@pytest.mark.parametrize("softplus", [False, True])
def test_output_shape_and_nonnegative(wiring, head, softplus):
    m = tiny(wiring, head, cr_softplus=softplus)
    with torch.no_grad():
        y, aux = m.forward_aux(imgs(3))
    assert y.shape == (3, 8) and (y >= 0).all()
    assert ("gate" in aux) == (head == "gated")
    assert m(imgs(3)).shape == (3, 8)


def test_last_layer_equals_relu_of_last_cr():
    m = tiny("paper", "last_layer")
    with torch.no_grad():
        x = imgs()
        _, x_cr, _ = m.paper_terms(x)
        assert torch.allclose(m(x), torch.relu(x_cr[-1]), atol=1e-6)


def test_mean_linear_is_pooled_backbone_linear():
    m = tiny("paper", "mean_linear")
    assert not hasattr(m, "transformer_encoder") and not hasattr(m, "feature_projection")
    x = imgs()
    with torch.no_grad():
        want = torch.relu(m.head_linear(m.resnet(x).mean(dim=(2, 3))))
        assert torch.allclose(m(x), want, atol=1e-6)


def test_cr_softplus_flag_changes_paper_output():
    a, b = tiny("paper"), tiny("paper", cr_softplus=True)
    b.load_state_dict(a.state_dict())
    x = imgs()
    with torch.no_grad():
        assert not torch.allclose(a(x), b(x))
        assert (b(x) > 0).all()


def test_invalid_combinations_raise():
    with pytest.raises(ValueError):
        tiny("code", "last_layer")
    with pytest.raises(ValueError):
        tiny("paper", "nope")
    with pytest.raises(ValueError):
        tiny("nope", "gated")


def test_build_model_tiny_uses_model_cfg():
    cfg = {"d": 32, "n_layers": 2, "nhead": 8, "ffn": 64, "dropout": 0.0, "wiring": "code", "head": "gated",
           "cr_softplus": False}
    m = build_model(cfg, pretrained=False, tiny=True)
    assert isinstance(m.resnet, TinyBackbone)
    assert m.eval()(imgs(1)).shape == (1, 8)
    assert count_params(m) == sum(p.numel() for p in m.parameters())


# Parameter counts: the paper's Tab. II/III only reproduce with the "code" wiring, whose registered (unused) template
# encoder layer counts too. ResNet50 is built for real (no weights); all 10 cases take a few seconds.
@pytest.mark.parametrize("L,paper_m", [(3, 59.21), (6, 84.41), (8, 101.22), (10, 118.02), (12, 134.82)])
def test_param_count_table2(L, paper_m):
    m = DroneKeyNet(d=1024, n_layers=L, wiring="code", pretrained=False)
    assert abs(count_params(m) / 1e6 - paper_m) <= 0.01


@pytest.mark.parametrize("d,paper_m", [(128, 27.92), (256, 33.24), (512, 46.63), (768, 63.69), (1024, 84.42)])
def test_param_count_table3(d, paper_m):
    m = DroneKeyNet(d=d, n_layers=6, wiring="code", pretrained=False)
    assert abs(count_params(m) / 1e6 - paper_m) <= 0.01
