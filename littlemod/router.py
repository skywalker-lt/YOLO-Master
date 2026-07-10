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

from ultralytics.nn.modules.conv import Conv, DWConv


class DensityRouter(nn.Module):
    """1x1 (C->C/r) -> 3x3 mix -> DW 3x3 -> 1x1 (->1). Emits raw density logits [B,1,gh,gw].

    The plan's minimal 1x1->DW->1x1 is too weak on narrow neck features (C=74 on EsMoE-N -> a ~1.6k
    router that overfits only ~0.83 on P4). Adding one 3x3 mixing conv at C/2 lifts the overfit ceiling
    to ~0.95 (near the 1.0 oracle) at ~16k params / <0.1 GFLOPs @ 40x40. Sigmoid applied by the caller
    (loss consumes logits; top-k is monotonic under sigmoid).
    """

    def __init__(self, c_in: int, reduction: int = 2):
        super().__init__()
        c = max(16, c_in // reduction)
        self.stem = Conv(c_in, c, 1)            # 1x1 reduce (ultralytics Conv = conv+bn+act)
        self.mix = Conv(c, c, 3)                # 3x3 channel-mix + receptive field
        self.dw = DWConv(c, c, 3)               # depthwise 3x3
        self.head = nn.Conv2d(c, 1, 1)          # raw logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dw(self.mix(self.stem(x))))

    @staticmethod
    def select_topk(s: torch.Tensor, k: int) -> torch.Tensor:
        """Expert-choice selection: indices of the top-k CELLS by density. Returns [B, k] (static k).

        `s` may be logits or probs — top-k is identical. `k` is a compile-time constant, so the
        downstream gather has static shape (no dynamic-shape / ScatterND — TensorRT/NPU-friendly).
        """
        b = s.shape[0]
        return torch.topk(s.reshape(b, -1), k, dim=1).indices
