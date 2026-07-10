"""Density loss L_dens = QFL + lambda * Dice, plus the GT-guided annealing mix for cold-start.

QFL reuses ultralytics `VarifocalLoss` (quality-focal that takes a soft `gt_score` target — exactly
the density map). Dice is new (grep-confirmed absent from the repo) and handles foreground sparsity.
Annealing (plan 4): S_tilde = (1-beta)*S + beta*D + eps, beta 1->0 over the first 30% of epochs, so
the P2 branch sees true small-object regions before the router is trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quality_focal_loss(pred_logits: torch.Tensor, target: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """QFL / Generalized Focal Loss for a CONTINUOUS [0,1] target (Li et al. 2020).

    -|y - sigma(x)|^beta * BCE(x, y), computed via BCE-with-logits for stability. Unlike the repo's
    VarifocalLoss (asymmetric, suppresses negatives -> flat output), QFL weights every cell by how far
    its prediction is from the density target, so it drives the router to match D's *ranking*, not to
    collapse to zero. No equivalent exists in ultralytics (grep-confirmed), hence built here.
    """
    p = pred_logits.sigmoid()
    modulator = (target - p).abs().pow(beta)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    return (modulator * bce).mean()


def dice_loss(pred_prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on the density map. pred_prob, target in [0,1], shape [B,1,H,W]."""
    p = pred_prob.flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1.0 - (2.0 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


class DensityLoss(nn.Module):
    """L_dens = QFL(S, D) + lambda_dice * Dice(sigmoid(S), D)."""

    def __init__(self, lambda_dice: float = 1.0, beta: float = 2.0):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.beta = beta

    def forward(self, s_logits: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, dict]:
        qfl = quality_focal_loss(s_logits, d, self.beta)
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
