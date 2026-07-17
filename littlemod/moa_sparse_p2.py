"""MoA-gated, UoMoE-fed, sparse-P2 block — composes the three mixtures this project has proven
in isolation into one P2/4 head. Builds directly on the littlemod sparse-P2 work.

    P2 dense feature (stride-4)
       ├─ UltraOptimizedMoE ──────────► richer P2 feature (shared expert + top-k routed experts)
       └─ _MoARouter (spatial, [B,G,H,W]) ──► per-cell keep-score + MoA aux loss (anti-collapse)
             │
       train: feat * softgate  (DENSE; the router gets gradient — learns to keep object cells and
                                 zero empty ones, which the entanglement probe showed *beats* dense P2)
       eval : top-rho cells -> SparseP2DetectHead runs cv2/cv3 only there (bit-exact, static-k, TRT-safe)

Why each piece (see DESIGN_sparse_p2.md and the deep-analysis report):
  * UoMoE — the ONE genuinely-sparse mixture in the repo (batch-subset dispatch, not compute-then-mask).
    Its always-on shared expert is why it trains where ES_MoE routing-collapses. Here it supplies the
    P2 feature "expert capacity".
  * MoA router — the ONE spatial (per-token [B,G,H,W]) router in the repo. UoMoE's own router is
    image-level (pools over H,W), so it cannot localise; MoA's can. We reuse it as the keep/skip gate.
    Its GShard balance+z+entropy aux loss (``_moa_router_aux_loss``) stops the gate collapsing to
    all-keep / all-skip. No GT density labels needed (unlike ``DensityRouter`` + ``DensityLoss``).
  * Sparse-P2 — the bit-exact gather/scatter of the Detect P2 branch (cv2[0]/cv3[0]) — the real
    +3.9 GFLOPs of adding P2 — computed only at the gate's top-rho cells.

Train dense / eval sparse — the same contract UoMoE and MoA already use.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.moa.moa import _MoARouter, _moa_router_aux_loss
from ultralytics.nn.modules.moe.modules import UltraOptimizedMoE, _registry_set  # SAME registry the
# unified loss collector reads (_collect_moe_aux_loss imports MOE_LOSS_REGISTRY from moe.modules).
# NB: moe._common has a DISTINCT registry object — publishing there would silently drop the gate aux.

from .router import DensityRouter          # reuse the static, TRT-safe select_topk
from .sparse_p2 import SparseP2DetectHead, _macs_conv_branch


class MoASparseP2(nn.Module):
    """UoMoE-fed P2 feature block with a MoA spatial keep-gate.

    Args:
        c_in:        input channels of the dense P2 (stride-4) feature.
        c2:          output channels of the P2 feature (defaults to ``c_in``).
        num_experts: UoMoE expert count.
        top_k:       UoMoE active experts (genuine sparse dispatch when ``top_k < num_experts``).
        expert_type: UoMoE expert flavour ('simple' | 'ghost' | 'inverted').
        cell:        routing-cell size in px on the P2 feature (4 -> stride-16 routing).
        halo:        receptive-field halo for the sparse detect gather (cv2/cv3 RF radius; 2 for C3k2-ish).
        gate_temperature: MoA router softmax temperature (annealed by the trainer, like MoA/MoT).
        aux_coeff:   weight on the MoA router aux loss (anti-collapse).
        keep_ratio:  eval budget rho — fraction of cells whose P2 detect branch is actually computed.
        target_keep: optional soft keep-rate prior added to the aux loss during training (nudges the
                     mean gate toward ``keep_ratio`` so the learned ordering matches the eval budget).
    """

    NUM_GROUPS = 2  # group 0 = "compute" (object-dense), group 1 = "skip" (empty)

    def __init__(self, c_in: int, c2: int | None = None, num_experts: int = 4, top_k: int = 2,
                 expert_type: str = "inverted", cell: int = 4, halo: int = 2,
                 gate_temperature: float = 1.0, aux_coeff: float = 0.01,
                 keep_ratio: float = 0.2, target_keep: float | None = None):
        super().__init__()
        c2 = c2 or c_in
        self.c_in, self.c2 = c_in, c2
        self.cell, self.halo = cell, halo
        self.aux_coeff = aux_coeff
        self.keep_ratio = keep_ratio
        self.target_keep = keep_ratio if target_keep is None else target_keep

        # Expert capacity for the P2 feature (genuinely sparse: top_k of num_experts + shared expert).
        self.moe = UltraOptimizedMoE(c_in, c2, num_experts=num_experts, top_k=top_k,
                                     expert_type=expert_type)
        # Spatial keep/skip gate (MoA router machinery, per-cell over the grid).
        self.gate = _MoARouter(c_in, num_groups=self.NUM_GROUPS, temperature=gate_temperature)

        # Diagnostics / aux-loss publishing (same protocol MoA uses: a ``last_aux_loss`` attribute the
        # C2fMoA-style collector reads and folds into the unified mixture loss).
        self.last_aux_loss: torch.Tensor = torch.zeros(())
        self.last_routing_snapshot: dict = {}

    # -- keep-score: MoA "compute" channel, pooled to the routing-cell grid -----------------------
    def _keep_grid(self, probs: torch.Tensor) -> torch.Tensor:
        """[B,G,H,W] soft probs -> [B,1,gh,gw] per-cell keep score (mean of the 'compute' channel)."""
        keep = probs[:, 0:1]                                   # [B,1,H,W]
        return F.avg_pool2d(keep, self.cell, self.cell)        # [B,1,gh,gw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (soft-gated in train / hard-gated in eval) P2 feature. ``self.keep_idx`` is set
        for a downstream sparse Detect head to consume; ``self.last_aux_loss`` is published for the loss."""
        B, _, H, W = x.shape
        feat = self.moe(x)                                     # [B, c2, H, W]  expert capacity
        probs, logits = self.gate(x, return_logits=True)      # [B, G, H, W]  spatial soft routing
        keep_grid = self._keep_grid(probs)                    # [B, 1, gh, gw]
        gh, gw = keep_grid.shape[-2:]

        if self.training:
            # Soft, DENSE: multiply the P2 feature by the upsampled keep-score so the router receives
            # gradient (which cells help the detect loss). Empty cells learn keep->0 (== the probe's
            # feature-gating denoising win). No GT density labels.
            soft = F.interpolate(keep_grid, size=(H, W), mode="nearest")
            out = feat * soft

            aux = _moa_router_aux_loss(probs, logits, self.aux_coeff)
            if self.target_keep is not None:                  # nudge mean keep toward the eval budget
                aux = aux + self.aux_coeff * (keep_grid.mean() - self.target_keep).pow(2)
            self.last_aux_loss = aux
            # Publish keyed on self so the unified loss collector (_collect_moe_aux_loss) folds it in
            # alongside the internal UoMoE's own registry entry — different module ids, both counted.
            _registry_set(self, aux)
            self.keep_idx = None
        else:
            # Hard, SPARSE: top-rho cells by keep-score; zero the P2 feature elsewhere (feature-gating).
            k = max(1, round(self.keep_ratio * gh * gw))
            keep_idx = DensityRouter.select_topk(keep_grid, k)         # [B, k] static
            mask = torch.zeros(B, 1, gh, gw, device=x.device, dtype=feat.dtype)
            mask.reshape(B, -1).scatter_(1, keep_idx, 1.0)
            out = feat * F.interpolate(mask, size=(H, W), mode="nearest")
            self.last_aux_loss = x.new_zeros(())
            self.keep_idx = keep_idx                            # for SparseP2DetectHead

        with torch.no_grad():
            kg = keep_grid.detach().float()
            self.last_routing_snapshot = {
                "num_experts": self.moe.num_experts, "top_k": self.moe.top_k,
                "keep_mean": float(kg.mean()), "keep_max": float(kg.max()),
                "aux_loss": float(self.last_aux_loss.detach()),
            }
        return out


# ---------------------------------------------------------------------------------------------------
# Self-contained smoke test (CPU, random tensors — no checkpoint needed)
# ---------------------------------------------------------------------------------------------------
def _smoke(c_in=64, H=160, W=160, num_experts=4, top_k=2, rho=0.2):
    torch.manual_seed(0)
    blk = MoASparseP2(c_in, num_experts=num_experts, top_k=top_k, keep_ratio=rho)
    x = torch.randn(2, c_in, H, W, requires_grad=True)

    # (1) train forward: soft gate, aux finite, router receives gradient
    blk.train()
    y = blk(x)
    assert y.shape == (2, c_in, H, W), y.shape
    aux = blk.last_aux_loss
    (y.mean() + aux).backward()
    grad_router = blk.gate.router[-1].weight.grad
    router_ok = grad_router is not None and grad_router.abs().sum().item() > 0
    print(f"[1] train  y={tuple(y.shape)}  aux={float(aux):.4e}  "
          f"router-grad={'flows ✓' if router_ok else 'DEAD ✗'}  snap={blk.last_routing_snapshot}")

    # (2) UoMoE is genuinely sparse: top_k < num_experts and a shared always-on expert
    moe_sparse = blk.moe.top_k < blk.moe.num_experts and hasattr(blk.moe, "shared_expert")
    print(f"[2] UoMoE   experts={blk.moe.num_experts} top_k={blk.moe.top_k} shared-expert={hasattr(blk.moe,'shared_expert')}  "
          f"{'sparse ✓' if moe_sparse else 'DENSE ✗'}")

    # (3) eval forward: hard top-rho gate produces a keep_idx of the right budget
    blk.eval()
    with torch.no_grad():
        ye = blk(x)
    gh = gw = H // blk.cell
    k = max(1, round(rho * gh * gw))
    kept_cells = (F.avg_pool2d((ye != 0).any(1, keepdim=True).float(), blk.cell, blk.cell) > 0).sum().item()
    print(f"[3] eval    keep_idx={tuple(blk.keep_idx.shape)} (k={k}/{gh*gw}={rho:.0%})  "
          f"nonzero-cells~={int(kept_cells)}  {'sparse ✓' if blk.keep_idx.shape[1]==k else '✗'}")

    # (4) SparseP2DetectHead driven by the MoA gate is BIT-EXACT vs dense-then-mask, with FLOP win
    cv2_0 = nn.Sequential(nn.Conv2d(c_in, 16, 3, 1, 1), nn.SiLU(), nn.Conv2d(16, 64, 1)).eval()   # reg/DFL-like
    cv3_0 = nn.Sequential(nn.Conv2d(c_in, 16, 3, 1, 1), nn.SiLU(), nn.Conv2d(16, 8, 1)).eval()    # cls-like
    feat = torch.randn(2, c_in, H, W)
    with torch.no_grad():
        reg_d, cls_d = cv2_0(feat), cv3_0(feat)                                # dense reference
        sp = SparseP2DetectHead(cv2_0, cv3_0, cell=blk.cell, halo=blk.halo)
        reg_s, cls_s = sp(feat, blk.keep_idx)
    maxerr = 0.0
    for b in range(2):
        for cid in blk.keep_idx[b].tolist():
            gy, gx = (cid // gw) * blk.cell, (cid % gw) * blk.cell
            e1 = (reg_s[b, :, gy:gy+blk.cell, gx:gx+blk.cell] - reg_d[b, :, gy:gy+blk.cell, gx:gx+blk.cell]).abs().max().item()
            e2 = (cls_s[b, :, gy:gy+blk.cell, gx:gx+blk.cell] - cls_d[b, :, gy:gy+blk.cell, gx:gx+blk.cell]).abs().max().item()
            maxerr = max(maxerr, e1, e2)
    pos = 0
    for b in range(2):
        for cid in blk.keep_idx[b].tolist():
            gy, gx = cid // gw, cid % gw
            ph = min(H, gy*blk.cell+blk.cell+blk.halo) - max(0, gy*blk.cell-blk.halo)
            pw = min(W, gx*blk.cell+blk.cell+blk.halo) - max(0, gx*blk.cell-blk.halo)
            pos += ph * pw
    pos /= 2
    dense_macs = _macs_conv_branch(cv2_0, H*W) + _macs_conv_branch(cv3_0, H*W)
    sparse_macs = _macs_conv_branch(cv2_0, pos) + _macs_conv_branch(cv3_0, pos)
    print(f"[4] sparse-detect  max|err| at kept cells = {maxerr:.2e}  {'BIT-EXACT ✓' if maxerr < 1e-4 else 'MISMATCH ✗'}  "
          f"| P2-detect {2*dense_macs/1e9:.3f}G -> {2*sparse_macs/1e9:.3f}G ({sparse_macs/dense_macs:.2f}x)")

    ok = router_ok and moe_sparse and blk.keep_idx.shape[1] == k and maxerr < 1e-4
    print("RESULT:", "ALL PASS ✓" if ok else "FAIL ✗")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MoA-gated UoMoE sparse-P2 block — smoke test")
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    a = ap.parse_args()
    _smoke(num_experts=a.experts, top_k=a.top_k, rho=a.rho)
