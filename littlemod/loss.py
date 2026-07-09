"""Density loss L_dens = QFL + lambda * Dice, plus the GT-guided annealing mix for cold-start.

QFL reuses ultralytics `VarifocalLoss` (quality-focal that takes a soft `gt_score` target — exactly
the density map). Dice is new (grep-confirmed absent from the repo) and handles foreground sparsity.
Annealing (plan 4): S_tilde = (1-beta)*S + beta*D + eps, beta 1->0 over the first 30% of epochs, so
the P2 branch sees true small-object regions before the router is trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.utils.loss import VarifocalLoss


def dice_loss(pred_prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on the density map. pred_prob, target in [0,1], shape [B,1,H,W]."""
    p = pred_prob.flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1.0 - (2.0 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


class DensityLoss(nn.Module):
    """L_dens = QFL(S, D) + lambda_dice * Dice(sigmoid(S), D)."""

    def __init__(self, lambda_dice: float = 1.0, fg_thresh: float = 0.0):
        super().__init__()
        self.vfl = VarifocalLoss()          # QFL surrogate: quality-aware focal BCE on logits
        self.lambda_dice = lambda_dice
        self.fg_thresh = fg_thresh

    def forward(self, s_logits: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """s_logits, d: [B,1,gh,gw]. VarifocalLoss wants [B, n] and reduces mean(1).sum()."""
        b = s_logits.shape[0]
        pred = s_logits.reshape(b, -1)
        gt = d.reshape(b, -1)
        label = (gt > self.fg_thresh).float()               # foreground indicator
        qfl = self.vfl(pred, gt, label) / max(b, 1)          # -> per-image mean
        dice = dice_loss(s_logits.sigmoid(), d)
        return qfl + self.lambda_dice * dice, {"qfl": float(qfl.detach()), "dice": float(dice.detach())}


def anneal_beta(epoch: int, total_epochs: int, frac: float = 0.3) -> float:
    """Linear 1 -> 0 over the first `frac` of training (plan 4)."""
    span = max(1.0, frac * total_epochs)
    return float(max(0.0, 1.0 - epoch / span))


def anneal_mix(s_prob: torch.Tensor, d: torch.Tensor, beta: float, eps_std: float = 0.0) -> torch.Tensor:
    """Selection target during training: S_tilde = (1-beta)*S + beta*D + eps. Select top-k on this."""
    out = (1.0 - beta) * s_prob + beta * d
    if eps_std > 0:
        out = out + torch.randn_like(out) * eps_std
    return out
