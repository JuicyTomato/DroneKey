"""The arms of plan step 3.3 (spec §5.1-5.2): recurrent encoder of the observed window + MLP head.

All positions in the normalised units of features.Normaliser (metres / pos_scale). The last layer of the head (and of
the R-F-P gate) starts at zero, so a new residual net predicts its base and a new R-F-P the mean of physics and
Kalman CA (w = 0.5).
  G      pred = head(h)
  R-CV   pred = cv + head(h, cv)
  R-F    pred = phys + head(h, phys)              (phys_mode with use_mode)
  R-F-P  pred = w phys + (1 - w) kf_ca + head(h, phys, kf_ca),  w = sigmoid(gate(h, phys, kf_ca)) per sample
With use_mode the one-hot command mode and drone class (B, 10) join the head (and gate) input.
"""
import torch
from torch import nn

ARMS = ("G", "R-CV", "R-F", "R-F-P")
MODE_CLASS_DIM = 10


def _mlp(d_in: int, hidden, d_out: int) -> nn.Sequential:
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    last = nn.Linear(d, d_out)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


class TrajNet(nn.Module):
    def __init__(self, arm: str, rnn: str = "gru", hidden: int = 128, layers: int = 2, H: int = 30,
                 use_mode: bool = False, head=(128, 128), gate_hidden: int = 64, n_in: int = 9):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}: one of {ARMS}")
        if rnn not in ("gru", "lstm"):
            raise ValueError(f"unknown rnn {rnn!r}")
        self.arm, self.rnn, self.use_mode, self.H = arm, rnn, use_mode, H
        self.encoder = (nn.GRU if rnn == "gru" else nn.LSTM)(n_in, hidden, layers, batch_first=True)
        n_bases = {"G": 0, "R-CV": 1, "R-F": 1, "R-F-P": 2}[arm]
        d_in = hidden + n_bases * H * 3 + (MODE_CLASS_DIM if use_mode else 0)
        self.head = _mlp(d_in, head, H * 3)
        self.gate = _mlp(d_in, (gate_hidden,), 1) if arm == "R-F-P" else None

    def _base(self, bases: dict) -> torch.Tensor:
        if self.arm == "R-CV":
            return bases["cv"]
        return bases["phys_mode"] if self.use_mode else bases["phys"]

    def forward(self, x: torch.Tensor, bases: dict, mode_class: torch.Tensor | None = None) -> dict:
        B = x.shape[0]
        h = self.encoder(x)[0][:, -1]
        feats = [h]
        base = None if self.arm == "G" else self._base(bases)
        if base is not None:
            feats.append(base.reshape(B, -1))
        if self.arm == "R-F-P":
            feats.append(bases["kf_ca"].reshape(B, -1))
        if self.use_mode:
            if mode_class is None:
                raise ValueError("the 'with mode' variant needs the mode and class one-hot (B, 10)")
            feats.append(mode_class)
        z = torch.cat(feats, dim=1)
        delta = self.head(z).reshape(B, self.H, 3)
        if self.arm == "G":
            return {"pred": delta, "w": None}
        if self.arm == "R-F-P":
            w = torch.sigmoid(self.gate(z))[:, 0]
            mix = w[:, None, None] * base + (1 - w[:, None, None]) * bases["kf_ca"]
            return {"pred": mix + delta, "w": w}
        return {"pred": base + delta, "w": None}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
