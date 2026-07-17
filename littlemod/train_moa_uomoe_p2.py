#!/usr/bin/env python3
"""Train the UoMoE-MoA-P2-N *baseline* (dense P2) on AI-TOD-v2 — standalone.

This is the reference run for the littlemod-moa experiment: UoMoE backbone experts + a MoA spatial
keep-gate at the P2/4 detect block (littlemod.moa_sparse_p2.MoASparseP2), evaluated DENSELY
(use_sparse_p2=False, the default). Its mAP is the number the window-gated sparse variant is measured
against; get the sparse number afterwards by evaluating the SAME best.pt with
``set_sparse_p2(model, True, rho)`` — no retrain (train dense / eval sparse).

Independent of scripts/reproduce/*. It bakes in the same AI-TOD-v2 recipe those scripts use, because
the defaults silently sabotage from-scratch baselines (learned the hard way — see the flags below):

  * imgsz 800     — AI-TOD-v2 is ~800px crops; the v0.1-N / UoMoE AI-TOD-v2 baselines trained @800,
                    NOT the ultralytics default 640. Match it or the comparison is invalid.
  * lora_r=0      — repo default.yaml ships lora_r=16, which would silently LoRA-fy the run
                    (train ~24% of params). r=0 disables LoRA (apply_lora becomes a no-op).
  * optimizer=auto— default.yaml drifted to AdamW@0.01 (10x too high -> NaN'd AI-TOD EsMoE-N, mAP 0);
                    auto -> SGD@0.01 (mom 0.9) for long runs.
  * pretrained=False, deterministic, seed 42, patience 0 (no early stop).

No ES_MOE here (uses UoMoE + MoA), so there is no sparse-eval-collapse to correct — the dense baseline
is just the model's native eval.

Examples:
    python littlemod/train_moa_uomoe_p2.py --check-build
    python littlemod/train_moa_uomoe_p2.py --epochs 300 --batch 64 --device 0
    python littlemod/train_moa_uomoe_p2.py --epochs 300 --batch 32 --cache disk   # MFS pod (ram can hang)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Run as `python littlemod/train_moa_uomoe_p2.py` puts littlemod/ (not repo root) on sys.path[0], so
# `import littlemod` — which tasks.py does lazily to register MoASparseP2 — would fail. Put root first.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CFG = "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2-moa-uomoe.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train UoMoE-MoA-P2-N (dense baseline) on AI-TOD-v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=800, help="AI-TOD-v2 crops are ~800px; matches the v0.1/UoMoE baselines.")
    p.add_argument("--batch", type=int, default=64, help="Lower (e.g. 32) if OOM — P2 @800 + the MoA block is heavier.")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache", nargs="?", const="ram", default=False,
                   help="'--cache'/'--cache ram' = RAM, '--cache disk' = on-disk .npy, omit to disable. "
                        "On network-volume (MFS) pods 'ram' can hang the val loader; use 'disk'.")
    p.add_argument("--data", default="AI-TOD-v2.yaml")
    p.add_argument("--project", default="runs/littlemod-moa/aitodv2")
    p.add_argument("--name", default="UoMoE-MoA-P2-N")
    p.add_argument("--check-build", action="store_true", help="Instantiate the model and exit.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    if args.check_build:
        m = DetectionModel(str(ROOT / CFG), ch=3, nc=8, verbose=False)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"[build-ok] UoMoE-MoA-P2-N: {n:.3f}M params  ({CFG})")
        return 0

    run_dir = Path(args.project) / args.name
    last_pt = run_dir / "weights" / "last.pt"
    resume = last_pt.exists()
    model = YOLO(str(last_pt) if resume else str(ROOT / CFG))
    print(f"[{'resume' if resume else 'train'}] {args.name}: cfg={CFG} data={args.data} "
          f"imgsz={args.imgsz} batch={args.batch} epochs={args.epochs}  (dense P2 baseline)", flush=True)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=False,
        lora_r=0,            # disable the default.yaml lora_r=16 (would silently LoRA-fy the run)
        optimizer="auto",    # SGD@0.01, not the drifted AdamW@0.01 that NaN'd AI-TOD baselines
        val=True,
        plots=True,
        cache=args.cache,
        patience=args.patience,
        amp=args.amp,
        resume=resume,
    )
    print(f"[done] weights -> {run_dir/'weights'/'best.pt'}\n"
          f"[next] sparse variant (no retrain): load best.pt, "
          f"littlemod.moa_sparse_p2.set_sparse_p2(model.model, True, rho=0.2), then val().", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
