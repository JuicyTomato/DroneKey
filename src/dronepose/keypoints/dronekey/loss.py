"""DroneKey losses: MSE, Gaussian (Sigma = I) and pose-adaptive Mahalanobis (paper form: d = sqrt(q)), all in float32 (spec 5.3)."""
import math

import torch
from torch import nn
import torch.nn.functional as F


def keypoint_covariance(y: torch.Tensor) -> torch.Tensor:
    """Biased (1/K) 2x2 covariance of the K keypoints of each sample: [B,K,2] -> [B,2,2]."""
    c = y - y.mean(dim=1, keepdim=True)
    return c.transpose(1, 2) @ c / y.shape[1]


def scale_schedule(epoch: int, alpha: float = 5.0, D: float = 10.0, epochs: int = 100, epochs_ref: int = 100) -> float:
    """S(t) = D * exp(-0.01 * alpha * r * t), t = 0-based epoch, r = (epochs_ref-1)/(epochs-1).

    r = 1 for the reference schedule; with fewer epochs the last epoch reaches the S of the reference's last epoch
    (spec 5.3 intent; the literal alpha*100/T would stop at exp(-4.75) instead of exp(-4.95) for T = 20).
    """
    r = (epochs_ref - 1) / (epochs - 1) if epochs > 1 else 1.0
    return D * math.exp(-0.01 * alpha * r * epoch)


def gaussian_terms(delta: torch.Tensor, cov_t: torch.Tensor) -> torch.Tensor:
    """Paper Eq. 10-11: 1 - exp(-d/2), d = sqrt(delta^T cov_t^-1 delta) per keypoint: [B,K,2], [B,2,2] -> [B,K]."""
    sol = torch.linalg.solve(cov_t, delta.transpose(1, 2))  # [B,2,K]
    q = (delta.transpose(1, 2) * sol).sum(dim=1)
    return 1.0 - torch.exp(-0.5 * torch.sqrt(q.clamp_min(1e-12)))  # clamp: finite gradient at q = 0


class KeypointLoss(nn.Module):
    """kind: 'mse' | 'gaussian' (Sigma_t = S(t) I + eps I) | 'pose_adaptive' (Sigma from the target keypoints)."""

    def __init__(self, kind: str = "pose_adaptive", alpha: float = 5.0, D: float = 10.0, eps: float = 1e-4,
                 epochs: int = 100):
        super().__init__()
        if kind not in ("mse", "gaussian", "pose_adaptive"):
            raise ValueError(f"unknown loss kind {kind!r}")
        self.kind, self.alpha, self.D, self.eps, self.epochs = kind, alpha, D, eps, epochs
        self.scale = scale_schedule(0, alpha, D, epochs)

    def set_epoch(self, epoch: int) -> None:
        self.scale = scale_schedule(epoch, self.alpha, self.D, self.epochs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = pred.float(), target.float()
        if self.kind == "mse":
            return F.mse_loss(pred, target)
        B = pred.shape[0]
        p, t = pred.reshape(B, -1, 2), target.reshape(B, -1, 2)
        eye = torch.eye(2, dtype=t.dtype, device=t.device)
        base = keypoint_covariance(t.detach()) if self.kind == "pose_adaptive" else eye.expand(B, 2, 2)
        cov_t = self.scale * base + self.eps * eye
        return gaussian_terms(t - p, cov_t).mean()
