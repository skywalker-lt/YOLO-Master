"""Validation step 1: train the DensityRouter ALONE (detector frozen) on VisDrone, log Recall@rho.

The cheapest falsification (plan §6.1): if a tiny router supervised by the GT density map can't rank
the small-object regions (Recall@rho >= ~97% @ 20%), the whole idea is dead — find out in a day.

ONE RULE — reuses ultralytics for everything but the router/density/loss:
  - model load + frozen forward (DetectionModel)
  - VisDrone dataloader + batch format (build_yolo_dataset / build_dataloader / check_det_dataset)
  - the neck feature is captured with a forward_pre_hook on the Detect head's input [P3,P4,P5].
Only the router's parameters receive gradients.

Run on a GPU pod (this repo + /data are on the shared network volume):
  python -m littlemod.train_router --weights runs/baseline/EsMoE-N_VisDrone/weights/best.pt \
      --level 2 --epochs 50 --batch 32 --out runs/littlemod/step1
"""
from __future__ import annotations

import argparse
import csv
import os

import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, LOGGER, SETTINGS

from littlemod.density import build_density_target, router_recall_at_rho
from littlemod.loss import DensityLoss
from littlemod.router import DensityRouter

RHOS = (0.1, 0.2, 0.3)


def make_loader(cfg, data, split, batch, workers, stride):
    ds = build_yolo_dataset(cfg, data[split], batch, data, mode=split, rect=(split == "val"), stride=stride)
    return build_dataloader(ds, batch, workers, shuffle=(split == "train"))


def neck_feature(model, detect_idx, level, img):
    """Frozen forward; return the captured neck feature at `level` (0=P3,1=P4,2=P5)."""
    cap = {}
    h = model.model[detect_idx].register_forward_pre_hook(lambda m, a: cap.__setitem__("f", a[0]))
    try:
        with torch.no_grad():
            model(img)
    finally:
        h.remove()
    return cap["f"][level]


@torch.no_grad()
def eval_recall(model, detect_idx, level, router, loader, dev, imgsz, small_thresh, max_batches=0):
    router.eval()
    cov = {r: 0.0 for r in RHOS}; tot = {r: 0 for r in RHOS}
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        img = batch["img"].to(dev).float() / 255
        p = neck_feature(model, detect_idx, level, img)
        s = router(p)
        gh, gw = s.shape[-2:]
        for r in RHOS:
            rec, n = router_recall_at_rho(s, batch["bboxes"].to(dev), batch["batch_idx"].to(dev),
                                          img.shape[0], (gh, gw), imgsz, r, small_thresh)
            cov[r] += rec * n; tot[r] += n
    return {r: (cov[r] / tot[r] if tot[r] else 0.0) for r in RHOS}


@torch.no_grad()
def oracle_ceiling(loader, dev, imgsz, gh, gw, stride, small_thresh):
    """Recall@rho using S = the true density D — the max achievable at this grid/rho/small-thresh.
    If this is < ~0.97 @0.2, no router can beat it -> fix the grid/target, not the router."""
    cov = {r: 0.0 for r in RHOS}; tot = {r: 0 for r in RHOS}
    for batch in loader:
        b = batch["img"].shape[0]
        bx = batch["bboxes"].to(dev); bi = batch["batch_idx"].to(dev)
        d = build_density_target(bx, bi, b, (gh, gw), imgsz, stride, small_thresh)
        for r in RHOS:
            rec, n = router_recall_at_rho(d, bx, bi, b, (gh, gw), imgsz, r, small_thresh)
            cov[r] += rec * n; tot[r] += n
    return {r: (cov[r] / tot[r] if tot[r] else 0.0) for r in RHOS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/baseline/EsMoE-N_VisDrone/weights/best.pt")
    ap.add_argument("--data", default="VisDrone.yaml")
    ap.add_argument("--datasets-dir", default="/data/datasets")
    ap.add_argument("--level", type=int, default=2, help="neck scale for the router: 0=P3,1=P4,2=P5(stride32)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reduction", type=int, default=4)
    ap.add_argument("--small-thresh", type=float, default=64.0)
    ap.add_argument("--lambda-dice", type=float, default=1.0)
    ap.add_argument("--out", default="runs/littlemod/step1")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    SETTINGS["datasets_dir"] = a.datasets_dir           # persist VisDrone on the shared volume

    # data (auto-downloads VisDrone on first run)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = a.imgsz
    cfg.mosaic = 0.0; cfg.mixup = 0.0; cfg.copy_paste = 0.0   # 1 real image per sample -> clean supervision
    data = check_det_dataset(a.data)
    stride = 640 // 20 * (a.imgsz // 640) if a.level == 2 else (16 if a.level == 1 else 8)
    train_loader = make_loader(cfg, data, "train", a.batch, a.workers, 32)
    val_loader = make_loader(cfg, data, "val", a.batch, a.workers, 32)

    # frozen detector
    model = YOLO(a.weights).model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    detect_idx = len(model.model) - 1

    # DENSE MoE inference — the default sparse path collapses EsMoE features (the same reason
    # --no-sparse-eval exists); dense gave the router +0.20 R@0.2 in probing. No-op on non-MoE models.
    try:
        from ultralytics.nn.modules.moe.modules import ES_MOE
        n_moe = 0
        for mod in model.modules():
            if isinstance(mod, ES_MOE):
                mod.use_sparse_inference = False; n_moe += 1
        LOGGER.info(f"[step1] dense MoE inference: use_sparse_inference=False on {n_moe} ES_MOE module(s)")
    except Exception as e:
        LOGGER.warning(f"[step1] could not set dense MoE inference ({e})")

    # probe one batch to size the router (channels of the chosen level) + confirm grid
    probe = next(iter(train_loader))
    pf = neck_feature(model, detect_idx, a.level, probe["img"].to(dev).float() / 255)
    c_in, gh, gw = pf.shape[1], pf.shape[2], pf.shape[3]
    real_stride = a.imgsz // gw
    LOGGER.info(f"[step1] level={a.level} feature=[{c_in},{gh},{gw}] stride={real_stride} k(20%)={int(0.2*gh*gw)}")

    ceil = oracle_ceiling(val_loader, dev, a.imgsz, gh, gw, real_stride, a.small_thresh)
    LOGGER.info("[step1] ORACLE ceiling (S=D): " + " ".join(f"R@{r}={ceil[r]:.3f}" for r in RHOS)
                + "  <- the trained router can't exceed this; if R@0.2 < ~0.97 fix grid/target, not the router")

    router = DensityRouter(c_in, reduction=a.reduction).to(dev)
    crit = DensityLoss(lambda_dice=a.lambda_dice)
    opt = torch.optim.AdamW(router.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project="yolo-grandmaster", name=f"step1-router-L{a.level}", config=vars(a))

    csv_path = os.path.join(a.out, "recall.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "qfl", "dice"] + [f"recall@{r}" for r in RHOS])

    best = -1.0
    for ep in range(a.epochs):
        router.train()
        run_loss = run_qfl = run_dice = 0.0; nb = 0
        for batch in train_loader:
            img = batch["img"].to(dev).float() / 255
            p = neck_feature(model, detect_idx, a.level, img)         # frozen, no grad
            s = router(p)                                            # grad flows to router only
            d = build_density_target(batch["bboxes"].to(dev), batch["batch_idx"].to(dev),
                                     img.shape[0], (gh, gw), a.imgsz, real_stride, a.small_thresh)
            loss, parts = crit(s, d)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss.detach()); run_qfl += parts["qfl"]; run_dice += parts["dice"]; nb += 1
        sched.step()
        rec = eval_recall(model, detect_idx, a.level, router, val_loader, dev, a.imgsz, a.small_thresh)
        row = [ep, run_loss / nb, run_qfl / nb, run_dice / nb] + [rec[r] for r in RHOS]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        LOGGER.info(f"[step1] ep{ep:03d} loss={row[1]:.4f} qfl={row[2]:.4f} dice={row[3]:.4f} "
                    + " ".join(f"R@{r}={rec[r]:.3f}" for r in RHOS))
        if run:
            run.log({"epoch": ep, "train/loss": row[1], "train/qfl": row[2], "train/dice": row[3],
                     **{f"val/recall@{r}": rec[r] for r in RHOS}})
        if rec[0.2] > best:
            best = rec[0.2]
            torch.save({"router": router.state_dict(), "level": a.level, "c_in": c_in,
                        "grid": (gh, gw), "recall@0.2": best, "epoch": ep},
                       os.path.join(a.out, "router_best.pt"))
    LOGGER.info(f"[step1] done. best Recall@0.2 = {best:.3f}. curve -> {csv_path}")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
