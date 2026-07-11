"""B3 — SparseP2DetectHead: compute the P2 (stride-4) detect branch ONLY at selected routing cells.

The P2 detect branch (Detect.cv2[0] reg/DFL + cv3[0] cls, ~3.2 GFLOPs = the real cost of adding P2) is
computed densely over 160x160 today, but its outputs at unselected cells are masked away. This wraps the
TRAINED cv2[0]/cv3[0] (ONE RULE: reuse, don't rebuild) and computes them only at the router-selected
cells, bit-exactly:

  routing cell = 4x4 px block on the stride-4 feature (stride-16 routing). cv2[0]/cv3[0] have RF radius 2,
  so gathering an 8x8 patch (cell + 2px halo) and running the branch on it yields a CENTER 4x4 output
  identical to dense (the halo supplies exactly the receptive field; F.unfold(padding=2) reproduces the
  dense boundary zero-pad; BN/SiLU are pointwise). Unselected cells output 0 (== the dense-then-mask ref).

Realized cost at rho: N_sel cells x (8x8 patch positions) vs dense (H/4)x(W/4). At rho=0.2 -> 0.8x the
dense branch (the 8x8/4x4 = 4x halo overhead x rho). This module is the measurement vehicle: prove
bit-exactness vs dense-then-mask, then count MACs / time it / try to export.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseP2DetectHead(nn.Module):
    def __init__(self, cv2_0: nn.Module, cv3_0: nn.Module, cell: int = 4, halo: int = 2):
        super().__init__()
        self.cv2, self.cv3 = cv2_0, cv3_0            # trained P2 reg + cls branches (reused)
        self.cell, self.halo = cell, halo
        self.patch = cell + 2 * halo                 # 8

    def forward(self, feat: torch.Tensor, keep_idx: torch.Tensor):
        """feat [B,C,H,W] dense P2 feature; keep_idx [B,k] flat cell ids on the (H/cell)x(W/cell) grid.
        Returns (reg, cls) each [B, out, H, W] with selected cells filled, unselected = 0.

        Interior cells use one batched 8x8 (cell+halo) unfold-gather -> bit-exact. Boundary-ring cells
        use image-clipped patches (the conv's own pad reproduces dense's edge zero-pad; an 8x8 zero-halo
        patch would instead RECOMPUTE conv over the halo and diverge). Boundary cells are few (outer ring)."""
        B, C, H, W = feat.shape
        cell, h, P = self.cell, self.halo, self.patch
        gh, gw = H // cell, W // cell
        rc, cc = self._out_ch(feat)
        out_reg = feat.new_zeros(B, rc, H, W)
        out_cls = feat.new_zeros(B, cc, H, W)

        for b in range(B):
            cells = keep_idx[b]
            gy, gx = cells // gw, cells % gw
            interior = (gy > 0) & (gy < gh - 1) & (gx > 0) & (gx < gw - 1)

            ci = cells[interior]
            if ci.numel():
                cols = F.unfold(feat[b:b + 1], P, stride=cell, padding=h).view(C, P, P, gh * gw)
                sel = cols[:, :, :, ci].permute(3, 0, 1, 2)                  # [ni, C, P, P]
                reg = self.cv2(sel)[:, :, h:h + cell, h:h + cell]            # [ni, rc, cell, cell]
                cls = self.cv3(sel)[:, :, h:h + cell, h:h + cell]
                yy, xx = (ci // gw) * cell, (ci % gw) * cell                 # [ni]
                dy = torch.arange(cell, device=feat.device)
                Y = (yy[:, None, None] + dy[None, :, None]).expand(-1, cell, cell).reshape(-1)   # [ni*cell*cell]
                X = (xx[:, None, None] + dy[None, None, :]).expand(-1, cell, cell).reshape(-1)
                out_reg[b][:, Y, X] = reg.permute(1, 0, 2, 3).reshape(rc, -1)  # (ni,dy,dx)-ordered cols
                out_cls[b][:, Y, X] = cls.permute(1, 0, 2, 3).reshape(cc, -1)

            for c in cells[~interior].tolist():                             # boundary ring: clipped patches
                gyy, gxx = c // gw, c % gw
                y0, y1 = max(0, gyy * cell - h), min(H, gyy * cell + cell + h)
                x0, x1 = max(0, gxx * cell - h), min(W, gxx * cell + cell + h)
                patch = feat[b:b + 1, :, y0:y1, x0:x1]
                r, cl = self.cv2(patch), self.cv3(patch)
                oy, ox = gyy * cell - y0, gxx * cell - x0
                out_reg[b, :, gyy * cell:gyy * cell + cell, gxx * cell:gxx * cell + cell] = r[0, :, oy:oy + cell, ox:ox + cell]
                out_cls[b, :, gyy * cell:gyy * cell + cell, gxx * cell:gxx * cell + cell] = cl[0, :, oy:oy + cell, ox:ox + cell]
        return out_reg, out_cls

    def _out_ch(self, feat):
        if not hasattr(self, "_oc"):
            with torch.no_grad():
                probe = feat[:1, :, :self.patch, :self.patch]
                self._oc = (self.cv2(probe).shape[1], self.cv3(probe).shape[1])
        return self._oc


def _macs_conv_branch(branch, positions):
    """MACs to run a conv branch over `positions` output pixels (sum over its Conv2d layers)."""
    tot = 0
    for m in branch.modules():
        if isinstance(m, nn.Conv2d):
            tot += m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] * positions / m.groups
    return tot


def bit_exact_test(weights="EsMoE-N-TPH-640/VisDrone_EsMoE-N-TPH/weights/best.pt", rho=0.2, seed_cells=None):
    import warnings
    warnings.filterwarnings("ignore")
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.modules import ES_MOE
    m = YOLO(weights).model.eval()
    for mod in m.modules():
        if isinstance(mod, ES_MOE):
            mod.use_sparse_inference = False
    det = m.model[-1]
    cv2_0, cv3_0 = det.cv2[0], det.cv3[0]

    B, C, H, W = 2, cv2_0[0].conv.in_channels, 160, 160
    feat = torch.randn(B, C, H, W)
    gh = gw = H // 4
    k = max(1, round(rho * gh * gw))
    # deterministic selection without Math.random-style calls: strided pick per image
    keep = torch.stack([torch.arange(b, b + k * 3, 3)[:k] % (gh * gw) for b in range(B)])

    with torch.no_grad():
        reg_d, cls_d = cv2_0(feat), cv3_0(feat)                    # dense reference
        sparse = SparseP2DetectHead(cv2_0, cv3_0)
        reg_s, cls_s = sparse(feat, keep)

    # compare at every selected cell's 4x4 block
    maxerr = 0.0
    for b in range(B):
        for cell in keep[b].tolist():
            gy, gx = (cell // gw) * 4, (cell % gw) * 4
            e1 = (reg_s[b, :, gy:gy + 4, gx:gx + 4] - reg_d[b, :, gy:gy + 4, gx:gx + 4]).abs().max().item()
            e2 = (cls_s[b, :, gy:gy + 4, gx:gx + 4] - cls_d[b, :, gy:gy + 4, gx:gx + 4]).abs().max().item()
            maxerr = max(maxerr, e1, e2)
    # realized positions: interior cells run an 8x8 patch, boundary cells a clipped (<=8x8) patch
    pos = 0
    for b in range(B):
        gy, gx = keep[b] // gw, keep[b] % gw
        for yy, xx in zip(gy.tolist(), gx.tolist()):
            ph = min(H, yy * 4 + 4 + 2) - max(0, yy * 4 - 2)
            pw = min(W, xx * 4 + 4 + 2) - max(0, xx * 4 - 2)
            pos += ph * pw
    pos /= B
    dense_macs = _macs_conv_branch(cv2_0, H * W) + _macs_conv_branch(cv3_0, H * W)
    sparse_macs = _macs_conv_branch(cv2_0, pos) + _macs_conv_branch(cv3_0, pos)  # per image
    print(f"[B3 bit-exact] rho={rho} k={k}/{gh*gw} cells | max abs err at selected = {maxerr:.2e}")
    print(f"[B3 flops] P2 detect branch dense={2*dense_macs/1e9:.3f} GFLOPs -> sparse={2*sparse_macs/1e9:.3f} "
          f"GFLOPs ({sparse_macs/dense_macs:.2f}x, saves {2*(dense_macs-sparse_macs)/1e9:.3f} G)")
    ok = maxerr < 1e-4
    print("RESULT:", "BIT-EXACT ✓" if ok else "MISMATCH ✗")
    return ok


def latency_test(weights="EsMoE-N-TPH-640/VisDrone_EsMoE-N-TPH/weights/best.pt", rho=0.2, iters=50, batch=1):
    import time
    import warnings
    warnings.filterwarnings("ignore")
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.modules import ES_MOE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = YOLO(weights).model.eval().to(dev)
    for mod in m.modules():
        if isinstance(mod, ES_MOE):
            mod.use_sparse_inference = False
    det = m.model[-1]
    cv2_0, cv3_0 = det.cv2[0], det.cv3[0]
    C = cv2_0[0].conv.in_channels
    feat = torch.randn(batch, C, 160, 160, device=dev)
    gh = gw = 40
    k = max(1, round(rho * gh * gw))
    keep = torch.arange(0, k * 3, 3)[:k].remainder(gh * gw).unsqueeze(0).expand(batch, -1).to(dev)
    sparse = SparseP2DetectHead(cv2_0, cv3_0).to(dev)

    def bench(fn):
        for _ in range(10):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / iters * 1e3      # ms

    # per-stage micro-benchmarks (interior path only — the bulk) to see WHERE the time goes
    P, h = sparse.patch, sparse.halo
    ci = keep[0]
    def unfold_only():
        return F.unfold(feat, P, stride=4, padding=h)
    cols_c = F.unfold(feat, P, stride=4, padding=h).view(batch, C, P, P, gh * gw)
    sel = cols_c[:, :, :, :, ci].permute(0, 4, 1, 2, 3).reshape(-1, C, P, P)
    def conv_only():
        return cv2_0(sel), cv3_0(sel)

    with torch.no_grad():
        t_dense = bench(lambda: (cv2_0(feat), cv3_0(feat)))
        t_sparse = bench(lambda: sparse(feat, keep))
        t_unfold = bench(unfold_only)
        t_conv = bench(conv_only)
    print(f"[B3 latency] rho={rho} k={k} batch={batch} on {dev}")
    print(f"    dense P2-detect = {t_dense:.3f} ms")
    print(f"    sparse total    = {t_sparse:.3f} ms  ({t_sparse/t_dense:.2f}x dense)  "
          f"{'FASTER ✓' if t_sparse < t_dense else 'SLOWER'}")
    print(f"    stage breakdown: unfold={t_unfold:.3f}  sparse-conv={t_conv:.3f}  "
          f"(rest = gather+scatter+boundary = {t_sparse - t_unfold - t_conv:.3f} ms)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--weights", default="EsMoE-N-TPH-640/VisDrone_EsMoE-N-TPH/weights/best.pt")
    ap.add_argument("--latency", action="store_true", help="also benchmark GPU latency")
    ap.add_argument("--batch", type=int, default=1, help="batch size for latency (tiny convs utilize better at higher batch)")
    a = ap.parse_args()
    bit_exact_test(a.weights, a.rho)
    if a.latency:
        latency_test(a.weights, a.rho, batch=a.batch)
