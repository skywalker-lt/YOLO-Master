"""Step 2, calibrated: base/dense/sparse mAP via ultralytics' OFFICIAL validator (DetMetrics, rect=True,
its exact NMS), so base/dense match the reported figure and the retained-% is trustworthy.

Injects P2 masking into DetectionValidator.postprocess (before NMS): base = all P2 anchors zeroed,
sparse = P2 kept only in the GT-density (oracle) top-rho cells. base uses the standalone no-P2 model.

  python -m littlemod.step2_val --weights EsMoE-N-TPH-640/VisDrone_EsMoE-N-TPH/weights/best.pt \
      --baseline-weights runs/baseline/EsMoE-N_VisDrone/weights/best.pt --grid 40 --rho 0.15
"""
from __future__ import annotations

import argparse

import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import DEFAULT_CFG, LOGGER, SETTINGS

from littlemod.density import build_density_target


class MaskedValidator(DetectionValidator):
    """DetectionValidator that masks P2 (stride-4) anchors before NMS. Set attrs after construction:
    mask_mode in {dense, base, sparse}, grid, rho, small_thresh, stride_p2."""

    mask_mode = "dense"
    routing_stride = 16
    rho = 0.2
    small_thresh = 64.0
    stride_p2 = 4
    _p2cell = None

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        self._batch = batch                      # stash GT for the oracle in postprocess
        return batch

    def postprocess(self, preds):
        if self.mask_mode != "dense":
            y = preds[0] if isinstance(preds, (list, tuple)) else preds
            y = self._apply_mask(y.clone())
            preds = (y, *preds[1:]) if isinstance(preds, (list, tuple)) else y
        return super().postprocess(preds)

    def _apply_mask(self, y):
        H, W = int(self._batch["img"].shape[-2]), int(self._batch["img"].shape[-1])
        ps = self.stride_p2
        ph, pw = H // ps, W // ps                       # P2 feature grid (rect)
        n_p2 = ph * pw
        if self.mask_mode == "base":
            y[:, 4:, :n_p2] = 0.0
            return y
        rr = max(1, self.routing_stride // ps)          # P2 cells per routing cell (16//4=4)
        gh, gw = ph // rr, pw // rr
        idx = torch.arange(n_p2, device=y.device)
        gy, gx = idx // pw, idx % pw
        p2cell = (gy // rr).clamp(max=gh - 1) * gw + (gx // rr).clamp(max=gw - 1)   # rect-aware cell id
        k = max(1, round(self.rho * gh * gw))
        for b in range(y.shape[0]):
            m = self._batch["batch_idx"] == b
            if not m.any():
                y[b, 4:, :n_p2] = 0.0
                continue
            d = build_density_target(self._batch["bboxes"][m], torch.zeros(int(m.sum()), device=y.device),
                                     1, (gh, gw), W, self.routing_stride, self.small_thresh)
            keepcell = torch.topk(d.reshape(-1), min(k, gh * gw)).indices
            keep = torch.isin(p2cell, keepcell)
            y[b, 4:, :n_p2][:, ~keep] = 0.0
        return y

def run(weights, mode, routing_stride, rho, small_thresh, data, imgsz, batch, workers):
    ym = YOLO(weights)
    try:
        from ultralytics.nn.modules.moe.modules import ES_MOE
        for mod in ym.model.modules():
            if isinstance(mod, ES_MOE):
                mod.use_sparse_inference = False   # dense MoE eval (matches --no-sparse-eval)
    except Exception:
        pass
    args = get_cfg(DEFAULT_CFG)
    args.data, args.imgsz, args.batch, args.workers = data, imgsz, batch, workers
    args.conf, args.iou, args.rect, args.split = 0.001, 0.7, True, "val"
    args.plots, args.verbose, args.save_json = False, False, False
    v = MaskedValidator(args=args)
    v.mask_mode, v.routing_stride, v.rho, v.small_thresh = mode, routing_stride, rho, small_thresh
    v.stride_p2 = int(ym.model.model[-1].stride[0])
    m = v(model=ym.model)
    return float(m["metrics/mAP50(B)"]), float(m["metrics/mAP50-95(B)"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="4-scale P2 model")
    ap.add_argument("--baseline-weights", default="runs/baseline/EsMoE-N_VisDrone/weights/best.pt")
    ap.add_argument("--data", default="VisDrone.yaml")
    ap.add_argument("--datasets-dir", default="/data/datasets")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--routing-stride", type=int, default=16, help="16 (~40x40 grid) or 32 (~20x20)")
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--small-thresh", type=float, default=64.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    SETTINGS["datasets_dir"] = a.datasets_dir

    LOGGER.info("[step2-cal] official-validator eval (DetMetrics, rect=True)")
    b50, b = run(a.baseline_weights, "dense", a.routing_stride, a.rho, a.small_thresh, a.data, a.imgsz, a.batch, a.workers)
    d50, d = run(a.weights, "dense", a.routing_stride, a.rho, a.small_thresh, a.data, a.imgsz, a.batch, a.workers)
    s50, s = run(a.weights, "sparse", a.routing_stride, a.rho, a.small_thresh, a.data, a.imgsz, a.batch, a.workers)
    LOGGER.info(f"  baseline (no P2)   : mAP50={b50:.4f}  mAP50-95={b:.4f}")
    LOGGER.info(f"  dense P2           : mAP50={d50:.4f}  mAP50-95={d:.4f}")
    LOGGER.info(f"  sparse P2 (rho={a.rho}): mAP50={s50:.4f}  mAP50-95={s:.4f}")
    gain, ret = d50 - b50, s50 - b50
    LOGGER.info(f"  P2 gain (dense-base)={gain:+.4f}  sparse retains={ret:+.4f} "
                f"({ret/gain*100 if gain > 0 else 0:.0f}%)")


if __name__ == "__main__":
    main()
