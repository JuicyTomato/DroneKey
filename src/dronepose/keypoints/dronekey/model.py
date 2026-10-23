"""DroneKey keypoint network: ResNet50 -> 1x1 conv -> tokens + PE -> transformer encoder -> gated heads.

Two wirings: `paper` follows Eq. 1-6 of the paper (chained layers); `code` reproduces the authors' train.py
(parallel self-attentions on the same input, softplus on the CR, unused encoder output and unused template layer).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_OUT = 8  # 4 propellers x (x, y)
WIRINGS = ("paper", "code")
HEADS = ("gated", "last_layer", "mean_linear")


class PositionalEncoding(nn.Module):
    """Sinusoidal PE over the token axis; input is sequence-first [P, B, d] (same as the authors')."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :]


class TinyBackbone(nn.Module):
    """Stride-32 conv stack for tests / --tiny runs: [B,3,224,224] -> [B,channels,7,7]."""

    def __init__(self, channels: int = 16):
        super().__init__()
        layers, c_in = [], 3
        for _ in range(5):
            layers += [nn.Conv2d(c_in, channels, 3, stride=2, padding=1), nn.ReLU()]
            c_in = channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _resnet50_features(pretrained: bool) -> nn.Module:
    from torchvision.models import ResNet50_Weights, resnet50

    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(net.children())[:-2])  # drop avgpool + fc


class DroneKeyNet(nn.Module):
    def __init__(self, d: int = 1024, n_layers: int = 6, nhead: int = 8, ffn: int = 2048, dropout: float = 0.1,
                 wiring: str = "paper", head: str = "gated", cr_softplus: bool = False, pretrained: bool = True,
                 backbone: nn.Module | None = None, backbone_channels: int = 2048):
        super().__init__()
        if wiring not in WIRINGS:
            raise ValueError(f"wiring must be one of {WIRINGS}, got {wiring!r}")
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        if wiring == "code" and head == "last_layer":
            raise ValueError("wiring='code' is only defined for head='gated' (and the transformer-free mean_linear)")
        self.d, self.n_layers, self.wiring, self.head = d, n_layers, wiring, head
        self.cr_softplus = cr_softplus or wiring == "code"  # the authors' code always applies softplus
        self.resnet = backbone if backbone is not None else _resnet50_features(pretrained)
        if head == "mean_linear":
            self.head_linear = nn.Linear(backbone_channels, N_OUT)
            return
        self.feature_projection = nn.Conv2d(backbone_channels, d, kernel_size=1)
        self.pos_encoder = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ffn, dropout=dropout)
        if wiring == "code":
            self.encoder_layer = layer  # registered although unused, as in train.py (counts in the parameters)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)  # batch_first=False: no nested path
        self.compressed_representor = nn.Linear(d, N_OUT)  # one W shared by all layers
        if head == "gated":
            self.gate_generator = nn.Linear(d, n_layers)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Images [B,3,H,W] -> X^(0) = conv1x1(features) flattened + PE, shape [P, B, d]."""
        f = self.feature_projection(self.resnet(x))
        b = f.size(0)
        return self.pos_encoder(f.reshape(b, self.d, -1).permute(2, 0, 1))

    def _cr(self, x_ir: torch.Tensor) -> torch.Tensor:
        cr = self.compressed_representor(x_ir)
        return F.softplus(cr) if self.cr_softplus else cr

    def paper_terms(self, x: torch.Tensor):
        """Paper wiring: per-layer X_IR [N][B,d], X_CR [N][B,8] and the gate [B,N] (None without a gated head)."""
        h = self.tokens(x)
        x_ir, x_cr = [], []
        for layer in self.transformer_encoder.layers:
            h = layer(h)  # X_EN^(l) from X_EN^(l-1)
            x_ir.append(h.mean(dim=0))
            x_cr.append(self._cr(x_ir[-1]))
        gate = F.softmax(self.gate_generator(x_ir[-1]), dim=-1) if self.head == "gated" else None
        return x_ir, x_cr, gate

    def _code_terms(self, x: torch.Tensor):
        """Line-by-line port of the authors' forward (see tests for the literal copy)."""
        h = self.tokens(x)
        _ = self.transformer_encoder(h)  # computed and discarded, as in train.py
        x_cr, gate = [], None
        for i, layer in enumerate(self.transformer_encoder.layers):
            out, _ = layer.self_attn(h, h, h, need_weights=False)  # every layer sees the same input
            x_ir = out.mean(dim=0)
            x_cr.append(self._cr(x_ir))
            if i == len(self.transformer_encoder.layers) - 1:
                gate = F.softmax(self.gate_generator(x_ir), dim=-1)
        return None, x_cr, gate

    def forward_aux(self, x: torch.Tensor):
        """(y [B,8], aux); aux = {"gate": [B,N]} for the gated head, {} otherwise."""
        if self.head == "mean_linear":
            return torch.relu(self.head_linear(self.resnet(x).mean(dim=(2, 3)))), {}
        _, x_cr, gate = self.paper_terms(x) if self.wiring == "paper" else self._code_terms(x)
        if self.head == "last_layer":
            return torch.relu(x_cr[-1]), {}
        cr = torch.stack(x_cr, dim=1)  # [B, N, 8]
        return torch.relu((cr * gate.unsqueeze(2)).sum(dim=1)), {"gate": gate}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_aux(x)[0]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(model_cfg: dict, pretrained: bool = True, tiny: bool = False) -> DroneKeyNet:
    """Model from the `model` section of configs/dronekey.json; tiny=True swaps ResNet50 for TinyBackbone(16)."""
    kw = dict(d=model_cfg["d"], n_layers=model_cfg["n_layers"], nhead=model_cfg["nhead"], ffn=model_cfg["ffn"],
              dropout=model_cfg["dropout"], wiring=model_cfg["wiring"], head=model_cfg["head"],
              cr_softplus=model_cfg.get("cr_softplus", False))
    if tiny:
        return DroneKeyNet(**kw, pretrained=False, backbone=TinyBackbone(16), backbone_channels=16)
    return DroneKeyNet(**kw, pretrained=pretrained)
