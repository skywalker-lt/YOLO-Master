"""Step 2 (upper-bound proxy): does routing the P2 branch to oracle-selected cells retain its AP gain?

Runs a trained 4-scale (P2/P3/P4/P5) detector on VisDrone val three ways and compares mAP:
  base   - all P2 (stride-4) anchors masked  -> P3/P4/P5 only (no P2 branch)
  dense  - full model                         -> P2 computed everywhere
  sparse - P2 anchors kept ONLY in the GT-density (ORACLE) top-rho cells; the rest masked

(dense - base) = P2's total gain; (sparse - base) = what oracle routing retains. Masking a *dense*-
trained P2 is an UPPER BOUND on a trained sparse head, so if `sparse` collapses to `base` the method
is dead. On VisDrone 94% of objects are small, so overall mAP tracks AP_small closely; we also report
a small-only (<64px GT) mAP.

Reuses ultralytics for model/data/NMS/AP and littlemod.density for the oracle selection. No training.

  python -m littlemod.step2_oracle_mask --weights EsMoE-N-TPH-640/VisDrone_EsMoE-N-TPH/weights/best.pt \
      --grid 40 --rho 0.2
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, LOGGER, SETTINGS
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy

from littlemod.density import build_density_target

IOUV = torch.linspace(0.5, 0.95, 10)


def match(det, gt_box, gt_cls):
    """Per-image TP matrix [n_det, 10] via IoU at 0.5:0.95 (ultralytics box_iou), greedy per threshold."""
    correct = np.zeros((det.shape[0], 10), dtype=bool)
    if gt_box.shape[0] == 0 or det.shape[0] == 0:
        return correct
    iou = box_iou(gt_box, det[:, :4])                                   # [n_gt, n_det]
    cc = gt_cls[:, None] == det[:, 5]                                   # class match
    iou = (iou * cc).cpu().numpy()
    for k, thr in enumerate(IOUV.tolist()):
        gi, di = np.nonzero(iou >= thr)
        if gi.size:
            m = np.stack([gi, di, iou[gi, di]], 1)
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), k] = True
    return correct


def evaluate(model, baseline_model, loader, dev, imgsz, grid, rho, stride_p2, small_thresh, conf, iou_nms, max_det, limit=0):
    """Return {cond: (mAP50, mAP5095)} for base/dense/sparse, plus small-only variants."""
    gh = gw = grid
    p2_side = imgsz // stride_p2                                        # 160 for 640/4
    n_p2 = p2_side * p2_side
    # precompute routing cell of each P2 anchor
    idx = torch.arange(n_p2, device=dev)
    gy, gx = idx // p2_side, idx % p2_side
    p2_cell = ((gy * gh) // p2_side) * gw + ((gx * gw) // p2_side)      # [n_p2] cell id
    conds = ("base", "dense", "sparse")
    stats = {c: {"tp": [], "conf": [], "pcls": [], "tcls": []} for c in conds}
    stats_s = {c: {"tp": [], "conf": [], "pcls": [], "tcls": []} for c in conds}

    for bi, batch in enumerate(loader):
        if limit and bi >= limit:
            break
        img = batch["img"].to(dev).float() / 255
        with torch.no_grad():
            y = model(img)[0]                                          # P2 model [B, 4+nc, N]
            yb = baseline_model(img)[0] if baseline_model is not None else None   # real standalone baseline
        B = img.shape[0]
        for b in range(B):
            # GT for this image (letterbox-normalized xywh -> px xyxy)
            m = batch["batch_idx"] == b
            gtb = xywh2xyxy(batch["bboxes"][m].to(dev)) * imgsz if m.any() else torch.zeros((0, 4), device=dev)
            gtc = batch["cls"][m].to(dev).view(-1) if m.any() else torch.zeros((0,), device=dev)
            gsz = (torch.maximum((gtb[:, 2] - gtb[:, 0]), (gtb[:, 3] - gtb[:, 1])) < small_thresh
                   if gtb.shape[0] else torch.zeros((0,), dtype=torch.bool, device=dev))
            # oracle top-k cells for this image
            keepcell = None
            if m.any():
                d = build_density_target(batch["bboxes"][m].to(dev), torch.zeros(int(m.sum()), device=dev),
                                         1, (gh, gw), imgsz, imgsz // gh, small_thresh)
                k = max(1, int(round(rho * gh * gw)))
                keepcell = torch.topk(d.reshape(-1), k).indices
            for cond in conds:
                if cond == "base":
                    if baseline_model is not None:
                        yc = yb[b:b + 1]                                # standalone baseline, full output
                    else:
                        yc = y[b:b + 1].clone(); yc[:, 4:, :n_p2] = 0.0  # (fallback: crippled masked-P2)
                elif cond == "sparse" and keepcell is not None:
                    yc = y[b:b + 1].clone()
                    keep = torch.isin(p2_cell, keepcell)
                    yc[:, 4:, :n_p2][:, :, ~keep] = 0.0
                else:                                                   # dense
                    yc = y[b:b + 1].clone()
                det = non_max_suppression(yc, conf, iou_nms, max_det=max_det)[0]   # [n,6] xyxy conf cls
                tp = match(det, gtb, gtc)
                for st, mask in ((stats, None), (stats_s, gsz)):
                    gt_c = gtc.cpu().numpy() if mask is None else gtc[mask].cpu().numpy()
                    st[cond]["tp"].append(tp); st[cond]["conf"].append(det[:, 4].cpu().numpy())
                    st[cond]["pcls"].append(det[:, 5].cpu().numpy()); st[cond]["tcls"].append(gt_c)

    def summarize(S):
        out = {}
        for c in conds:
            tp = np.concatenate(S[c]["tp"]); cf = np.concatenate(S[c]["conf"])
            pc = np.concatenate(S[c]["pcls"]); tc = np.concatenate(S[c]["tcls"])
            r = ap_per_class(tp, cf, pc, tc, plot=False)
            ap = r[5] if isinstance(r, tuple) else r.ap        # ap_per_class returns tuple; ap is index 5 (per-class per-iou)
            ap = np.asarray(ap)
            out[c] = (float(ap[:, 0].mean()), float(ap.mean()))
        return out
    return summarize(stats), summarize(stats_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="4-scale P2 model")
    ap.add_argument("--baseline-weights", default="runs/baseline/EsMoE-N_VisDrone/weights/best.pt", help="standalone no-P2 baseline for the base row")
    ap.add_argument("--data", default="VisDrone.yaml")
    ap.add_argument("--datasets-dir", default="/data/datasets")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--grid", type=int, default=40, help="routing grid side (40=stride16, 20=stride32)")
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--small-thresh", type=float, default=64.0)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap val batches (smoke test)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    SETTINGS["datasets_dir"] = a.datasets_dir
    global IOUV
    IOUV = IOUV.to(dev)

    cfg = get_cfg(DEFAULT_CFG); cfg.imgsz = a.imgsz
    data = check_det_dataset(a.data)
    ds = build_yolo_dataset(cfg, data["val"], a.batch, data, mode="val", rect=False, stride=32)
    loader = build_dataloader(ds, a.batch, a.workers, shuffle=False)

    model = YOLO(a.weights).model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        from ultralytics.nn.modules.moe.modules import ES_MOE
        for mod in model.modules():
            if isinstance(mod, ES_MOE):
                mod.use_sparse_inference = False
    except Exception:
        pass
    baseline_model = None
    if a.baseline_weights:
        baseline_model = YOLO(a.baseline_weights).model.to(dev).eval()
        for p in baseline_model.parameters():
            p.requires_grad_(False)
        try:
            from ultralytics.nn.modules.moe.modules import ES_MOE
            for mod in baseline_model.modules():
                if isinstance(mod, ES_MOE):
                    mod.use_sparse_inference = False
        except Exception:
            pass
    stride_p2 = int(model.model[-1].stride[0])
    LOGGER.info(f"[step2] grid={a.grid} rho={a.rho} P2 stride={stride_p2} ({a.imgsz//stride_p2}x{a.imgsz//stride_p2})")

    allm, sm = evaluate(model, baseline_model, loader, dev, a.imgsz, a.grid, a.rho, stride_p2, a.small_thresh,
                        a.conf, a.iou, a.max_det, a.limit)
    LOGGER.info("[step2] === mAP (all objects) ===")
    for c in ("base", "dense", "sparse"):
        LOGGER.info(f"  {c:6s}: mAP50={allm[c][0]:.4f}  mAP50-95={allm[c][1]:.4f}")
    LOGGER.info("[step2] === mAP (small <64px only) ===")
    for c in ("base", "dense", "sparse"):
        LOGGER.info(f"  {c:6s}: mAP50={sm[c][0]:.4f}  mAP50-95={sm[c][1]:.4f}")
    d, s, b = sm["dense"][0], sm["sparse"][0], sm["base"][0]
    LOGGER.info(f"[step2] AP_small50: P2 gain (dense-base)={d-b:+.4f}  oracle-sparse retains={s-b:+.4f} "
                f"({(s-b)/(d-b)*100 if d>b else 0:.0f}% of the gain at rho={a.rho})")


if __name__ == "__main__":
    main()
