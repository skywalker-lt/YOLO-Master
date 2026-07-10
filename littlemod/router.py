"""Spatial density router — the single router that gates the sparse P2 branch (and is reusable as
the MoE gate, per the "one router" goal).

Reuses ultralytics `Conv` / `DWConv`. Distinct from the repo's `EfficientSpatialRouter` /
`AdaptiveRoutingLayer`, which pool to image-level expert logits (the image-wise routing this project
replaces): this keeps the full spatial grid and emits a per-cell density S, top-k'd over CELLS
(expert-choice, fixed capacity), not over experts.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DensityRouter(nn.Module):
    """GroupNorm conv head -> raw density logits [B,1,gh,gw].

    NOT ultralytics `Conv` (which carries BatchNorm): on a small router over a frozen detector's
    features, BN's batch statistics diverge from its running stats, giving a large train/eval gap
    (train R@0.2=0.95 but eval=0.34) -> the val-recall oscillation. GroupNorm behaves identically in
    train and eval, killing the gap. Three 3x3 convs give the receptive field to aggregate
    small-object evidence into a per-cell density. Sigmoid applied by the caller.
    """

    def __init__(self, c_in: int, c: int = 64, groups: int = 8, layers: int = 3):
        super().__init__()
        c = max(groups, (c // groups) * groups)
        blocks = [nn.Conv2d(c_in, c, 3, 1, 1), nn.GroupNorm(groups, c), nn.SiLU()]
        for _ in range(layers - 1):
            blocks += [nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(groups, c), nn.SiLU()]
        blocks += [nn.Conv2d(c, 1, 1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def select_topk(s: torch.Tensor, k: int) -> torch.Tensor:
        """Expert-choice selection: indices of the top-k CELLS by density. Returns [B, k] (static k).

        `s` may be logits or probs — top-k is identical. `k` is a compile-time constant, so the
        downstream gather has static shape (no dynamic-shape / ScatterND — TensorRT/NPU-friendly).
        """
        b = s.shape[0]
        return torch.topk(s.reshape(b, -1), k, dim=1).indices


class MultiLevelDensityRouter(nn.Module):
    """Fuse P3 (fine, stride 8) + P4 (stride 16) onto the P4 grid, then a GN head -> density logits.

    P3 resolves small objects better than P4 but is 2x the resolution; it's downsampled (stride-2 conv)
    to the P4 grid and concatenated with P4, giving the router the fine detail P4 alone lacks. Emits
    [B,1,gh_p4,gw_p4]. Same static top-k selection as DensityRouter (reuse `DensityRouter.select_topk`).
    """

    def __init__(self, c_p3: int, c_p4: int, c: int = 64, groups: int = 8, layers: int = 3,
                 c_p3_out: int = 32):
        super().__init__()
        c = max(groups, (c // groups) * groups)
        c_p3_out = max(groups, (c_p3_out // groups) * groups)
        self.down = nn.Sequential(nn.Conv2d(c_p3, c_p3_out, 3, 2, 1), nn.GroupNorm(groups, c_p3_out), nn.SiLU())
        blocks = [nn.Conv2d(c_p3_out + c_p4, c, 3, 1, 1), nn.GroupNorm(groups, c), nn.SiLU()]
        for _ in range(layers - 1):
            blocks += [nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(groups, c), nn.SiLU()]
        blocks += [nn.Conv2d(c, 1, 1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.down(p3), p4], dim=1))
