#!/usr/bin/env python3
"""Train v0.1-P2-N with NWD (Normalized Wasserstein Distance) label assignment on AI-TOD-v2 — standalone.

New route after the UoMoE-MoA-P2 negative result: instead of adding capacity to the P2 head (which
hurt), keep the proven v0.1-P2-N architecture UNCHANGED and fix the actual AI-TOD failure mode — IoU is
degenerate for ~12px objects (a 1px shift drops IoU 0.5->0), so TaskAligned assignment mis-matches tiny
GTs and CIoU gives a near-flat gradient. NWD models boxes as 2-D Gaussians and stays smooth at tiny
scale (Wang et al.). We blend it into the assigner's localization metric: overlap=(1-r)*CIoU+r*NWD.
Zero FLOPs added; the change is purely in label assignment (ultralytics/utils/tal.py + metrics.py).

Baseline to beat: v0.1-P2-N @ AI-TOD-v2 (mAP50 0.264 / mAP50-95 0.120 @ep30). Same architecture here, so
any delta is attributable to NWD assignment alone.

Recipe (baked in, same as the other AI-TOD runs): imgsz 800, lora_r=0, optimizer=auto, from-scratch.
No ES_MOE in v0.1-P2-N -> no --no-sparse-eval needed.

Examples:
    python littlemod/train_p2_nwd.py --check-build
    python littlemod/train_p2_nwd.py --epochs 300 --batch 64 --nwd-ratio 0.5 --cache disk
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CFG = "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2.yaml"   # v0.1-P2-N (unchanged architecture)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train v0.1-P2-N + NWD assignment on AI-TOD-v2.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--batch", type=int, default=64, help="v0.1-P2-N is 22.5 GFLOPs (lighter than MoA-P2); 64 should fit. Lower if OOM.")
    p.add_argument("--nwd-ratio", type=float, default=0.5, help="Blend weight for NWD in assignment (0=stock IoU, 1=pure NWD).")
    p.add_argument("--nwd-const", type=float, default=12.8, help="NWD constant C (~mean object px; 12.8 for AI-TOD).")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache", nargs="?", const="ram", default=False,
                   help="'--cache disk' = on-disk .npy (use on pods). Omit to disable.")
    p.add_argument("--data", default="AI-TOD-v2.yaml")
    p.add_argument("--project", default="runs/littlemod-moa/aitodv2")
    p.add_argument("--name", default="v0.1-P2-N-NWD")
    p.add_argument("--check-build", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    if args.check_build:
        m = DetectionModel(str(ROOT / CFG), ch=3, nc=8, verbose=False)
        print(f"[build-ok] v0.1-P2-N: {sum(p.numel() for p in m.parameters())/1e6:.3f}M  ({CFG})")
        return 0

    run_dir = Path(args.project) / args.name
    last_pt = run_dir / "weights" / "last.pt"
    resume = last_pt.exists()
    model = YOLO(str(last_pt) if resume else str(ROOT / CFG))
    print(f"[{'resume' if resume else 'train'}] {args.name}: cfg={CFG} data={args.data} imgsz={args.imgsz} "
          f"batch={args.batch}  NWD ratio={args.nwd_ratio} const={args.nwd_const}", flush=True)

    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, seed=args.seed, deterministic=True,
        project=args.project, name=args.name, exist_ok=True, pretrained=False,
        lora_r=0, optimizer="auto",
        nwd_ratio=args.nwd_ratio, nwd_const=args.nwd_const,   # <-- the only change vs the v0.1-P2-N baseline
        val=True, plots=True, cache=args.cache, patience=args.patience, amp=args.amp, resume=resume,
    )
    print(f"[done] weights -> {run_dir/'weights'/'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
