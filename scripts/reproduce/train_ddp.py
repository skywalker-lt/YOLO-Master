#!/usr/bin/env python3
"""DDP-only trainer for YOLO-Master baselines — every model × {VisDrone, SKU-110K, AI-TOD-v2}.

This is a MULTI-GPU-ONLY script. It refuses to run on a single GPU or CPU: pass a comma-separated
``--device`` with at least two GPUs (e.g. ``--device 0,1``). It uses Ultralytics' native DDP — this
(parent) process prepares the dataset once, then Ultralytics auto-spawns one worker per GPU via
``torch.distributed.run`` and the workers do the training. ``--batch`` is the TOTAL batch, split
evenly across the GPUs (keep it divisible by the GPU count).

Self-contained: it depends only on the repo's already-committed library behaviour and needs NO
further edits to ``ultralytics/``. Specifically it relies on three fixes that live in the library so
they reach the auto-spawned workers:
  * ``BaseTrainer`` builds DDP with ``broadcast_buffers=False`` for MoE models (NCCL can't broadcast
    the per-rank stats buffers some MoE blocks register on CPU),
  * ``validate()`` broadcasts only CUDA EMA buffers, and
  * the ``es_moe_dense_eval`` training arg forces ES_MOE onto the dense eval path (``--no-sparse-eval``)
    — a serialized arg, so it survives auto-spawn where a runtime callback would not.
The MoE load-balance losses were also made warning-free (no ``c10d::allreduce_`` spam).

Models (``--model``): v0.1-N, EsMoE-N, UoMoE-N, UoMoE-P2-N, EsMoE-P2-N, v0.1-P2-N, or ``all``.
Datasets (``--dataset``): VisDrone / SKU-110K (imgsz 640), AI-TOD-v2 (imgsz 800).

Recipe baked in (matches the per-dataset reproduce scripts): from-scratch, ``lora_r=0`` (repo default
would silently LoRA-fy), ``optimizer=auto`` -> SGD@0.01 (repo default drifted to AdamW@0.01 which NaN'd
AI-TOD), ``deterministic``, ``seed 42``, ``patience 0``. ES_MOE models keep the as-shipped sparse eval
unless ``--no-sparse-eval`` is passed (then eval is corrected to match training).

Examples:
    python scripts/reproduce/train_ddp.py --dataset VisDrone  --model UoMoE-N    --device 0,1 --batch 128
    python scripts/reproduce/train_ddp.py --dataset AI-TOD-v2 --model v0.1-P2-N  --device 0,1,2,3 --batch 128
    python scripts/reproduce/train_ddp.py --dataset SKU-110K  --model EsMoE-N    --device 0,1 --no-sparse-eval
    python scripts/reproduce/train_ddp.py --dataset VisDrone  --model UoMoE-N    --device 0,1 --batch 256 --lr0 0.04
    python scripts/reproduce/train_ddp.py --dataset VisDrone  --model all        --device 0,1 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Registry: 3 datasets x 6 models (cfg paths verified against the repo)        #
# --------------------------------------------------------------------------- #
DATASETS = {
    "VisDrone":  {"data": "VisDrone.yaml",  "imgsz": 640, "nc": 10, "project": "runs/reproduce/visdrone"},
    "SKU-110K":  {"data": "SKU-110K.yaml",  "imgsz": 640, "nc": 1,  "project": "runs/reproduce/sku110k"},
    "AI-TOD-v2": {"data": "AI-TOD-v2.yaml", "imgsz": 800, "nc": 8,  "project": "runs/reproduce/aitodv2"},
}

# uses_esmoe=True -> contains ES_MOE blocks whose default sparse eval collapses mAP; --no-sparse-eval
# forwards es_moe_dense_eval=True to correct it. False models are train/eval-consistent (flag is a no-op).
MODELS = {
    "v0.1-N":     {"cfg": "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",         "esmoe": False},
    "EsMoE-N":    {"cfg": "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml",            "esmoe": True},
    "UoMoE-N":    {"cfg": "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-uomoe.yaml",    "esmoe": False},
    "UoMoE-P2-N": {"cfg": "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2-uomoe.yaml", "esmoe": False},
    "EsMoE-P2-N": {"cfg": "ultralytics/cfg/models/master/v0/det/yolo-master-n-p2.yaml",         "esmoe": True},
    "v0.1-P2-N":  {"cfg": "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2.yaml",       "esmoe": False},
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _gpu_count(device: str) -> int:
    """GPUs implied by a --device string ('0,1,2,3' -> 4; '0'/'cpu'/'' -> handled by caller)."""
    if isinstance(device, str) and device.strip() not in ("", "cpu", "mps"):
        return len([d for d in device.split(",") if d.strip() != ""])
    return 0


def _last_epoch(results_csv: Path) -> int | None:
    if not results_csv.exists():
        return None
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    try:
        return int(float({k.strip(): v for k, v in rows[-1].items()}["epoch"]))
    except (KeyError, ValueError):
        return None


def _final_metrics(results_csv: Path) -> str:
    if not results_csv.exists():
        return "(no results.csv)"
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "(empty results.csv)"
    r = {k.strip(): v for k, v in rows[-1].items()}
    g = lambda k: r.get(k, "?")  # noqa: E731
    return (f"epoch={g('epoch')} mAP50={g('metrics/mAP50(B)')} mAP50-95={g('metrics/mAP50-95(B)')}")


def build_opt(args: argparse.Namespace) -> dict:
    """Optimizer/LR kwargs. Default 'auto' -> SGD@0.01 (auto IGNORES lr0). --lr0 forces SGD and pins
    the auto recipe's momentum/warmup so only the LR differs (for large-batch linear scaling)."""
    opt = {"optimizer": args.optimizer}
    if args.lr0 is not None:
        opt["lr0"] = args.lr0
        if args.optimizer == "auto":
            opt["optimizer"] = "SGD"
        opt["momentum"] = 0.9
        opt["warmup_bias_lr"] = 0.0
    return opt


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_one(args: argparse.Namespace, ds: dict, model_name: str, project: Path) -> dict:
    from ultralytics import YOLO

    ms = MODELS[model_name]
    run_name = f"{args.dataset}_{model_name}"
    run_dir = project / run_name
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"
    done = _last_epoch(run_dir / "results.csv")

    if best_pt.exists() and done is not None and done + 1 >= args.epochs:
        print(f"[skip] {run_name}: already complete at epoch {done}", flush=True)
        return {"model": model_name, "status": "skipped"}

    dense_eval = ms["esmoe"] and not args.sparse_eval  # ES_MOE dense eval (library-applied via arg)
    imgsz = args.imgsz or ds["imgsz"]
    opt = build_opt(args)

    if last_pt.exists() and done is not None:
        print(f"[resume] {run_name}: {last_pt} epoch={done} -> {args.epochs}", flush=True)
        model = YOLO(str(last_pt))
        resume = True
    else:
        print(f"[train] {run_name}: cfg={ms['cfg']} data={ds['data']} imgsz={imgsz} "
              f"batch={args.batch} ddp={_gpu_count(args.device)}x dense_eval={dense_eval} "
              f"optimizer={opt['optimizer']}" + (f" lr0={args.lr0}" if args.lr0 is not None else ""), flush=True)
        model = YOLO(str(ROOT / ms["cfg"]))
        resume = False

    start = time.time()
    model.train(
        data=ds["data"],
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        device=args.device,           # comma-list -> Ultralytics native DDP auto-spawn
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=run_name,
        exist_ok=True,
        pretrained=False,
        lora_r=0,                     # disable default.yaml lora_r=16 (would silently LoRA-fy the run)
        es_moe_dense_eval=dense_eval, # serialized arg -> reaches DDP workers (no runtime callback needed)
        **opt,                        # optimizer (auto->SGD@0.01) + optional --lr0 override
        val=True,
        plots=True,
        cache=args.cache,
        patience=args.patience,
        amp=args.amp,
        resume=resume,
        verbose=args.verbose,
    )
    print(f"[done] {run_name}: {_final_metrics(run_dir / 'results.csv')}  ({time.time() - start:.1f}s)", flush=True)
    return {"model": model_name, "status": "resumed" if resume else "ok"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASETS),
                   help="Which dataset to train on.")
    p.add_argument("--model", required=True, choices=list(MODELS) + ["all"],
                   help="Model to train, or 'all' for every model on this dataset.")
    p.add_argument("--device", required=True,
                   help="Comma-separated GPU ids, >=2 (e.g. '0,1'). REQUIRED: this script is DDP-only.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64, help="TOTAL batch, split evenly across GPUs.")
    p.add_argument("--imgsz", type=int, default=None, help="Override the per-dataset default (640 / 800).")
    p.add_argument("--optimizer", default="auto",
                   help="Default 'auto' -> SGD@0.01 (tuned recipe; IGNORES --lr0). --lr0 forces SGD.")
    p.add_argument("--lr0", type=float, default=None,
                   help="Initial LR override for large-batch scaling (e.g. --lr0 0.04 for batch 256). "
                        "Forces SGD and pins the auto recipe's momentum=0.9 / warmup_bias_lr=0.")
    p.add_argument("--workers", type=int, default=16, help="Dataloader workers PER GPU.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache", nargs="?", const="ram", default=False,
                   help="'--cache'/'--cache ram' = RAM, '--cache disk' = on-disk .npy, omit to disable.")
    p.add_argument("--sparse-eval", action=argparse.BooleanOptionalAction, default=True,
                   help="ES_MOE sparse eval. Default True = as-shipped (collapses mAP). --no-sparse-eval "
                        "opts into corrected dense eval (train==eval). No-op for non-ES_MOE models.")
    p.add_argument("--project", default=None, help="Override the per-dataset run project directory.")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled",
                   help="Sets WANDB_MODE for Ultralytics' native W&B (inherited by workers). Default off; "
                        "per-epoch metrics always go to results.csv regardless.")
    p.add_argument("--check-build", action="store_true", help="Instantiate the selected model(s) and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    ds = DATASETS[args.dataset]
    models = list(MODELS) if args.model == "all" else [args.model]
    project = Path(args.project) if args.project else (ROOT / ds["project"])

    # W&B via env so the auto-spawned workers (which inherit this process's env) pick it up.
    os.environ["WANDB_MODE"] = args.wandb_mode

    # --- utility modes (no DDP needed) ---
    if args.check_build:
        from ultralytics.nn.tasks import DetectionModel
        for m in models:
            mdl = DetectionModel(str(ROOT / MODELS[m]["cfg"]), ch=3, nc=ds["nc"], verbose=False)
            print(f"[build-ok] {m}: {sum(p.numel() for p in mdl.parameters()) / 1e6:.3f}M  ({MODELS[m]['cfg']})")
        return 0

    # --- DDP-only enforcement ---
    n_gpu = _gpu_count(args.device)
    if n_gpu < 2:
        print(f"[error] This script is DDP-ONLY. --device={args.device!r} implies {n_gpu} GPU(s). "
              f"Pass at least two, e.g. --device 0,1 (single-GPU/CPU is intentionally not supported here; "
              f"use scripts/reproduce/reproduce_{{visdrone,sku110k,aitodv2}}.py for that).", file=sys.stderr)
        return 2
    if args.batch % n_gpu:
        print(f"[warn] batch={args.batch} not divisible by {n_gpu} GPUs; Ultralytics floor-divides, "
              f"so each rank gets {args.batch // n_gpu} (remainder dropped).", flush=True)

    imgsz = args.imgsz or ds["imgsz"]
    print(f"[train_ddp] dataset={args.dataset} data={ds['data']} imgsz={imgsz} project={project}\n"
          f"            models={models}  device={args.device} ({n_gpu} GPUs, native DDP)  "
          f"batch={args.batch}(total) epochs={args.epochs}  wandb={args.wandb_mode}", flush=True)
    for m in models:
        note = "ES_MOE dense-eval" if (MODELS[m]["esmoe"] and not args.sparse_eval) else \
               ("ES_MOE sparse-eval (as-shipped)" if MODELS[m]["esmoe"] else "no ES_MOE")
        print(f"  - {m:<11} {note}")

    if args.dry_run:
        return 0

    project.mkdir(parents=True, exist_ok=True)
    statuses = []
    for m in models:
        try:
            statuses.append(train_one(args, ds, m, project))
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[fail] {m}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            statuses.append({"model": m, "status": "failed", "error": str(exc)})

    print(f"\n[train_ddp] DONE — {args.dataset}")
    for st in statuses:
        print("  ", st)
    ok = {"ok", "resumed", "skipped"}
    return 0 if all(s.get("status") in ok for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
