"""GT density target + Router Recall@rho — the "free supervision" for sparse P2 routing.

Genuinely new code (no equivalent in ultralytics): a Gaussian-splat density map D built from
small-object GT boxes on the routing grid, and the Recall@rho diagnostic. Reuses the ultralytics
batch format only (`bboxes` normalized xywh, `batch_idx`, `cls`) — no new tensor plumbing.

Grid convention: routing on a (gh, gw) grid at `stride` (e.g. 20x20 @ stride 32 for 640 input).
Cell (i, j) center in cell units = (j + 0.5, i + 0.5); a GT at normalized (cx, cy) sits at
cell coord (cx * gw, cy * gh).
"""
from __future__ import annotations

import torch


def _small_mask_and_weight(size_px: torch.Tensor, small_thresh: float | None,
                           w_range: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Small-object keep-mask + per-object weight w_i = clip(small_thresh / size, wmin, wmax).

    Bigger weight for smaller objects (they need the P2 resolution most). small_thresh=None keeps all.
    """
    keep = torch.ones_like(size_px, dtype=torch.bool) if small_thresh is None else (size_px < small_thresh)
    ref = small_thresh if small_thresh else size_px.new_tensor(64.0)
    w_i = torch.clamp(ref / size_px.clamp(min=1.0), w_range[0], w_range[1])
    return keep, w_i


def build_density_target(boxes_xywh: torch.Tensor, batch_idx: torch.Tensor, batch_size: int,
                         grid_hw: tuple[int, int], img_size: int, stride: int,
                         small_thresh: float | None = 64.0,
                         w_range: tuple[float, float] = (1.0, 3.0)) -> torch.Tensor:
    """Gaussian-splat density target D in [0,1] on the routing grid.

    D(p) = max_i  w_i * exp(-||p - c_i||^2 / (2 * sigma_i^2)),  sigma_i = max(w_box,h_box)_px / (2*stride).
    `max` (not sum) -> stable [0,1] range for QFL/Dice and ranking-oriented (top-k), per the plan.

    Args:
        boxes_xywh: [N,4] normalized cx,cy,w,h (ultralytics batch format).
        batch_idx:  [N] image index per box.
        grid_hw:    (gh, gw) routing grid.
        img_size:   input side (px), e.g. 640.
        stride:     routing stride (px), e.g. 32.
    Returns:
        D: [B, 1, gh, gw] float in [0,1].
    """
    gh, gw = grid_hw
    dev = boxes_xywh.device
    D = torch.zeros(batch_size, 1, gh, gw, device=dev)
    if boxes_xywh.numel() == 0:
        return D

    size_px = torch.maximum(boxes_xywh[:, 2], boxes_xywh[:, 3]) * img_size          # [N]
    keep, w_i = _small_mask_and_weight(size_px, small_thresh, w_range)
    sigma_cells = (size_px / (2.0 * stride)).clamp(min=0.5)                          # [N], >= half a cell

    # cell-center grid in cell units: [gh*gw, 2] as (x, y)
    yy, xx = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing="ij")
    cells = torch.stack([xx.reshape(-1) + 0.5, yy.reshape(-1) + 0.5], dim=1).float()  # [gh*gw, 2]

    cx = boxes_xywh[:, 0] * gw
    cy = boxes_xywh[:, 1] * gh
    for b in range(batch_size):
        m = keep & (batch_idx == b)
        if not m.any():
            continue
        centers = torch.stack([cx[m], cy[m]], dim=1)                                 # [n, 2]
        d2 = (cells[:, None, :] - centers[None, :, :]).pow(2).sum(-1)                # [gh*gw, n]
        g = w_i[m][None, :] * torch.exp(-d2 / (2.0 * sigma_cells[m][None, :].pow(2)))  # [gh*gw, n]
        D[b, 0] = g.max(dim=1).values.reshape(gh, gw).clamp(0.0, 1.0)
    return D


@torch.no_grad()
def router_recall_at_rho(S: torch.Tensor, boxes_xywh: torch.Tensor, batch_idx: torch.Tensor,
                         batch_size: int, grid_hw: tuple[int, int], img_size: int, rho: float,
                         small_thresh: float | None = 64.0) -> tuple[float, int]:
    """Fraction of small-object GT centers whose routing cell is in the top-(rho*ncells) cells of S.

    The plan's ceiling diagnostic: >=~97% at rho=20% => method viable; ~85% => fix the target design.

    Returns (recall, n_small_gts).
    """
    gh, gw = grid_hw
    k = max(1, int(round(rho * gh * gw)))
    covered = total = 0
    size_px = torch.maximum(boxes_xywh[:, 2], boxes_xywh[:, 3]) * img_size
    keep = torch.ones_like(size_px, dtype=torch.bool) if small_thresh is None else (size_px < small_thresh)
    for b in range(batch_size):
        m = keep & (batch_idx == b)
        if not m.any():
            continue
        topk_cells = set(torch.topk(S[b].reshape(-1), k).indices.tolist())
        cj = (boxes_xywh[m, 0] * gw).long().clamp(0, gw - 1)
        ci = (boxes_xywh[m, 1] * gh).long().clamp(0, gh - 1)
        cell_ids = (ci * gw + cj).tolist()
        total += len(cell_ids)
        covered += sum(c in topk_cells for c in cell_ids)
    return (covered / total if total else 0.0), total
