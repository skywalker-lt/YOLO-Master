#!/usr/bin/env python3
"""Shared logic for the per-dataset YOLO-Master baseline reproduction scripts.

Both reproduce_visdrone.py and reproduce_sku110k.py train the two nano release
variants from their YAML configs (from scratch) and log per-epoch metrics
(mAP50, mAP50-95, box/cls/dfl/moe_loss) to each run's results.csv, plus an
aggregated summary.csv.

Models
------
  - YOLO-Master-v0.1-N  -> ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml
      (MoE block: OptimizedMOEImproved -- train/eval-consistent, always-on shared
       expert; no sparse-inference issue.)
  - YOLO-Master-EsMoE-N -> ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml
      (MoE block: ES_MOE. Its default eval path (`use_sparse_inference=True`)
       prunes to ~1 unnormalized expert while training blends all experts, which
       collapses validation mAP.)

Sparse vs dense evaluation (EsMoE-N only)
-----------------------------------------
By DEFAULT the scripts reproduce the model exactly as shipped -- ES_MOE keeps
`use_sparse_inference=True`, so EsMoE-N's validation mAP collapses. This is
intentional: the default is a faithful, unmodified reproduction.

Pass ``--no-sparse-eval`` to opt into the CORRECTED evaluation. It is an explicit
flag (not a silent default) so the change is visible in the command you ran. It
forwards ``es_moe_dense_eval=True`` to ``model.train``; the trainer then flips
`ES_MOE.use_sparse_inference=False` on both the live model and its EMA in
``_setup_train`` (before any validation and before checkpoints are written from
the EMA), so per-epoch val, the saved .pt, and final eval all use the same dense
forward as training. Driving it through a serialized arg (not a runtime callback)
means it also applies under Multi-GPU DDP, where auto-spawned workers rebuild the
trainer from args alone. v0.1-N / UoMoE have no ES_MOE modules, so it is a no-op there.

Multi-GPU (DDP)
---------------
Pass a comma-separated ``--device`` to use Ultralytics' native multi-GPU DDP::

    python scripts/reproduce/reproduce_visdrone.py --device 0,1 --batch 64 ...

Ultralytics handles it: this (parent) process prepares the dataset once in
``Trainer.__init__`` (so there is no multi-rank download/convert race) and then
auto-spawns one worker per GPU via ``torch.distributed.run``; the workers do the
training. ``--batch`` is the TOTAL batch, split evenly across the GPUs (keep it
divisible by the GPU count).

NOTE: the auto-spawned workers rebuild the trainer from its args alone. Anything
that must reach them therefore goes through a serialized arg, not a runtime
callback: the ES_MOE ``--no-sparse-eval`` dense-eval is forwarded as
``es_moe_dense_eval`` and applied in the trainer's ``_setup_train``, and the
CPU-buffer DDP fixes live in ``ultralytics/engine/trainer.py`` -- so both apply on
every rank. The only parent-only piece is this script's custom W&B logger (added
via ``add_callback``); Ultralytics' own W&B integration still runs in the workers,
and for these runs W&B is typically ``--wandb-mode offline`` anyway.
"""
from __future__ import annotations

import argparse
import csv
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/moe_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "val/moe_loss",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    cfg: str
    uses_esmoe: bool = False  # True if the model contains ES_MOE blocks (sparse-eval sensitive)


@dataclass(frozen=True)
class DatasetSpec:
    name: str          # short tag, e.g. "VisDrone"
    data: str          # dataset yaml, e.g. "VisDrone.yaml"
    project: str       # e.g. "runs/reproduce/visdrone"


# The two shared nano baselines. EsMoE-N gets dense validation (--no-sparse-eval).
MODELS = (
    ModelSpec("v0.1-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml", uses_esmoe=False),
    ModelSpec("EsMoE-N", "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml", uses_esmoe=True),
)

# The four variants derived from the two baselines. Shared across every dataset script so the
# reproducible model set stays consistent (dataset-agnostic cfgs -- nc is bound at build time):
#   * P2/4 head  -> EsMoE-P2-N, v0.1-P2-N  (extra stride-4 head for tiny objects)
#   * UltraOptimizedMoE swap -> UoMoE-N (shared expert + batched sparse compute + ultra-light
#     router; ~20-30% fewer GFLOPs at equal params) and UoMoE-P2-N (that swap + the P2/4 head).
# The UoMoE models are sparse train==eval, so uses_esmoe=False (--no-sparse-eval is a no-op for them).
VARIANTS = (
    ModelSpec("EsMoE-P2-N", "ultralytics/cfg/models/master/v0/det/yolo-master-n-p2.yaml", uses_esmoe=True),
    ModelSpec("v0.1-P2-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2.yaml", uses_esmoe=False),
    ModelSpec("UoMoE-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-uomoe.yaml", uses_esmoe=False),
    ModelSpec("UoMoE-P2-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-uomoe-p2.yaml", uses_esmoe=False),
)


# --------------------------------------------------------------------------- #
# Real-time W&B per-epoch logging                                              #
# --------------------------------------------------------------------------- #
# Metrics logged every epoch: mAP50, mAP50-95, box_loss, cls_loss, moe_loss
# (train + val variants where available). One W&B run per (dataset, model).
_WANDB_METRICS = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
    "train/box_loss": "train/box_loss",
    "train/cls_loss": "train/cls_loss",
    "train/moe_loss": "train/moe_loss",
    "val/box_loss": "val/box_loss",
    "val/cls_loss": "val/cls_loss",
    "val/moe_loss": "val/moe_loss",
}


def _make_wandb_callbacks(run_name: str, dataset: "DatasetSpec", spec: "ModelSpec",
                          args: argparse.Namespace, dense_val: bool) -> dict:
    """Return trainer callbacks that stream per-epoch metrics to Weights & Biases.

    Robust by design: if wandb is missing or init fails (e.g. not logged in for
    online mode), a warning is emitted and training continues without wandb.
    """
    from ultralytics.utils import LOGGER

    state = {"run": None}

    def on_train_start(trainer):
        try:
            import wandb
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"[reproduce] wandb unavailable ({exc}); continuing without it.")
            return
        try:
            state["run"] = wandb.init(
                project=args.wandb_project,
                entity=(args.wandb_entity or None),
                name=run_name,
                mode=args.wandb_mode,
                reinit=True,
                config={
                    "model": spec.name, "cfg": spec.cfg,
                    "dataset": dataset.name, "data": dataset.data,
                    "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
                    "seed": args.seed, "dense_val": dense_val,
                },
            )
            url = getattr(state["run"], "url", None)
            LOGGER.info(f"[reproduce] wandb run '{run_name}' [{args.wandb_mode}] -> {url}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                f"[reproduce] wandb init failed ({exc}); continuing without wandb. "
                f"For a live URL run `wandb login` first, or use --wandb-mode offline."
            )
            state["run"] = None

    def on_fit_epoch_end(trainer):
        run = state["run"]
        if run is None:
            return
        data = {}
        try:
            data.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
        except Exception:  # noqa: BLE001
            pass
        try:
            data.update(trainer.metrics or {})
        except Exception:  # noqa: BLE001
            pass
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        log = {"epoch": epoch}
        for out_key, src_key in _WANDB_METRICS.items():
            v = data.get(src_key)
            if v is not None:
                try:
                    log[out_key] = float(v)
                except (TypeError, ValueError):
                    pass
        try:
            run.log(log, step=epoch)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"[reproduce] wandb log failed at epoch {epoch}: {exc}")

    def on_train_end(trainer):
        run = state["run"]
        if run is not None:
            try:
                run.finish()
            except Exception:  # noqa: BLE001
                pass
            state["run"] = None

    return {"on_train_start": on_train_start,
            "on_fit_epoch_end": on_fit_epoch_end,
            "on_train_end": on_train_end}


# --------------------------------------------------------------------------- #
# Summary CSV                                                                  #
# --------------------------------------------------------------------------- #
def _read_last_metrics(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {k.strip(): v for k, v in rows[-1].items()} if rows else {}


def _float_or_blank(value: str | None) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def write_summary(project: Path, dataset: DatasetSpec, models=MODELS, sparse_eval: bool = True) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    out = project / "summary.csv"
    fieldnames = ["dataset", "model", "cfg", "run_dir", "dense_eval", "epoch", *METRIC_KEYS]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for spec in models:
            run_dir = project / f"{dataset.name}_{spec.name}"
            res = _read_last_metrics(run_dir / "results.csv")
            row = {
                "dataset": dataset.name,
                "model": spec.name,
                "cfg": spec.cfg,
                "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
                "dense_eval": (spec.uses_esmoe and not sparse_eval) if spec.uses_esmoe else "n/a",
                "epoch": res.get("epoch", ""),
            }
            for k in METRIC_KEYS:
                row[k] = _float_or_blank(res.get(k))
            w.writerow(row)
    return out


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def _completed_epoch(run_dir: Path) -> int | None:
    val = _read_last_metrics(run_dir / "results.csv").get("epoch")
    try:
        return int(float(val)) if val not in (None, "") else None
    except ValueError:
        return None


def train_one(args: argparse.Namespace, dataset: DatasetSpec, spec: ModelSpec, project: Path,
              overrides: dict | None = None) -> dict:
    from ultralytics import YOLO

    run_name = f"{dataset.name}_{spec.name}"
    run_dir = project / run_name
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"
    done = _completed_epoch(run_dir)

    if best_pt.exists() and done is not None and done + 1 >= args.epochs:
        print(f"[skip] {run_name}: complete at epoch {done}", flush=True)
        return {"model": spec.name, "status": "skipped"}

    # Corrected dense evaluation is opt-in via --no-sparse-eval, and only affects
    # ES_MOE models (v0.1-N has none, so it is a no-op there).
    dense_eval = spec.uses_esmoe and not args.sparse_eval
    if last_pt.exists() and done is not None:
        print(f"[resume] {run_name}: {last_pt} epoch={done} -> {args.epochs}", flush=True)
        model = YOLO(str(last_pt))
        resume = True
    else:
        print(f"[train] {run_name}: cfg={spec.cfg} data={dataset.data} "
              f"sparse_eval={args.sparse_eval} dense_eval={dense_eval}", flush=True)
        model = YOLO(str(ROOT / spec.cfg))
        resume = False

    if args.wandb and args.wandb_mode != "disabled":
        for event, fn in _make_wandb_callbacks(run_name, dataset, spec, args, dense_eval).items():
            model.add_callback(event, fn)

    # Optimizer / LR. Default 'auto' -> SGD@0.01 (the tuned recipe; auto IGNORES lr0). If --lr0 is
    # given, honor it: auto would discard it, so force SGD and pin the auto recipe's momentum (0.9)
    # and warmup_bias_lr (0.0) so ONLY the LR differs from the proven baseline. Use for large-batch
    # linear scaling, e.g. batch 256 -> --lr0 0.04 (4x the batch-64 default).
    opt = {"optimizer": args.optimizer}
    if args.lr0 is not None:
        opt["lr0"] = args.lr0
        if args.optimizer == "auto":
            opt["optimizer"] = "SGD"
        opt["momentum"] = 0.9
        opt["warmup_bias_lr"] = 0.0
        print(f"[opt] {run_name}: optimizer={opt['optimizer']} lr0={args.lr0} momentum=0.9 "
              f"warmup_bias_lr=0.0 (only LR differs from the auto->SGD@0.01 recipe)", flush=True)

    start = time.time()
    model.train(
        data=dataset.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=run_name,
        exist_ok=True,
        pretrained=False,
        lora_r=0,  # full from-scratch baseline: repo default.yaml ships lora_r=16, which would
                   # silently LoRA-fy the run (train ~24% of params). r=0 disables LoRA (apply_lora no-op).
        # ES_MOE dense eval (--no-sparse-eval). Library-applied via this serialized arg so it
        # survives DDP auto-spawn (workers get it). Only sent when actually enabled: it is a
        # fork-specific arg, and older/stock ultralytics rejects unknown keys outright
        # ("'es_moe_dense_eval' is not a valid YOLO argument"). Models without ES_MOE never need it.
        **({"es_moe_dense_eval": True} if dense_eval else {}),
        **opt,  # optimizer (default 'auto' -> SGD@0.01) + optional --lr0 override (built above). We never
                # rely on default.yaml's optimizer (drifted to AdamW@0.01, 10x too high; NaN'd AI-TOD).
        val=True,
        plots=True,
        cache=args.cache,
        patience=args.patience,
        amp=args.amp,
        resume=resume,
        verbose=args.verbose,
        # Per-dataset protocol pins, applied LAST so they beat default.yaml. Needed because
        # default.yaml is tuned for the paper's LoRA/MoLoRA experiments, not for published
        # baselines: e.g. it ships cos_lr=True while the released COCO runs used cos_lr=False.
        **(overrides or {}),
    )
    return {"model": spec.name, "status": "resumed" if resume else "ok",
            "duration_s": f"{time.time() - start:.1f}"}


def build_parser(dataset: DatasetSpec, models=MODELS) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Reproduce YOLO-Master {', '.join(m.name for m in models)} baselines on {dataset.name}.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=300, help="Recommended ~300 (adjust to GPU budget).")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64,
                   help="TOTAL batch size. Under multi-GPU DDP it is split evenly across GPUs, "
                        "so keep it divisible by the GPU count.")
    p.add_argument("--device", default="0",
                   help="GPU id(s). A single id ('0') trains on one GPU; a comma list ('0,1,2,3') "
                        "runs Ultralytics' native multi-GPU DDP (one auto-spawned worker per GPU). "
                        "'cpu' for CPU.")
    p.add_argument("--workers", type=int, default=16, help="Dataloader workers PER GPU under DDP.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--optimizer", default="auto",
                   help="Optimizer. Default 'auto' -> SGD@0.01 (the tuned recipe). NOTE: 'auto' IGNORES "
                        "--lr0; pass --lr0 (which forces SGD) or an explicit optimizer to set the LR.")
    p.add_argument("--lr0", type=float, default=None,
                   help="Initial LR override for large-batch scaling (e.g. --lr0 0.04 for batch 256). "
                        "Forces optimizer=SGD if 'auto' (which would ignore lr0) and pins the auto recipe's "
                        "momentum=0.9 / warmup_bias_lr=0 so ONLY the LR differs from the batch-64 baseline.")
    p.add_argument("--cache", nargs="?", const="ram", default=False,
                   help="Cache images: '--cache'/'--cache ram' = RAM, '--cache disk' = on-disk .npy, "
                        "omit to disable. On network-volume (MFS) pods 'ram' can hang building the val "
                        "loader; 'disk' avoids that but writes .npy back to the same volume.")
    p.add_argument("--project", default=dataset.project)
    p.add_argument("--model", choices=[m.name for m in models] + ["both"], default="both",
                   help=f"Which model to train: {', '.join(m.name for m in models)}, or both (default).")
    p.add_argument("--sparse-eval", action=argparse.BooleanOptionalAction, default=True,
                   help="ES_MOE sparse inference at validation/inference. Default True reproduces "
                        "EsMoE-N as-is (its sparse-eval path collapses mAP). Pass --no-sparse-eval "
                        "to opt into the CORRECTED dense evaluation (train==eval). No-op for v0.1-N.")
    # --- Weights & Biases real-time per-epoch logging ---
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                   help="Stream mAP50/mAP50-95/box/cls/moe loss to W&B each epoch (default on). Use --no-wandb to disable.")
    p.add_argument("--wandb-project", default="yolo-master-reproduce", help="W&B project name.")
    p.add_argument("--wandb-entity", default="", help="W&B entity/team (optional).")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online",
                   help="online needs `wandb login`; offline logs locally (sync later); disabled turns it off.")
    p.add_argument("--check-build", action="store_true", help="Instantiate both models and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    p.add_argument("--summary-only", action="store_true", help="Only (re)write summary.csv from existing runs.")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def run_dataset(dataset: DatasetSpec, models=MODELS, overrides: dict | None = None) -> int:
    """Entry point used by the per-dataset scripts."""
    args = build_parser(dataset, models).parse_args()
    project = Path(args.project) if Path(args.project).is_absolute() else ROOT / args.project
    specs = list(models) if args.model == "both" else [m for m in models if m.name == args.model]

    wandb_desc = "off" if (not args.wandb or args.wandb_mode == "disabled") else args.wandb_mode
    ddp_desc = f"  ddp={len(str(args.device).split(','))}x (native)" if "," in str(args.device) else ""
    print(f"[reproduce:{dataset.name}] data={dataset.data}  project={project}  "
          f"sparse_eval={args.sparse_eval}  wandb={wandb_desc}{ddp_desc}")
    for s in specs:
        dense = s.uses_esmoe and not args.sparse_eval
        note = f"dense_eval={dense}" if s.uses_esmoe else "no ES_MOE (sparse-eval n/a)"
        print(f"  - {s.name:<8} cfg={s.cfg}  {note}")

    if args.dry_run:
        return 0
    if args.check_build:
        from ultralytics.nn.tasks import DetectionModel
        for s in specs:
            m = DetectionModel(str(ROOT / s.cfg), ch=3, nc=80, verbose=False)
            print(f"[build-ok] {s.name}: {sum(p.numel() for p in m.parameters()) / 1e6:.3f}M  ({s.cfg})")
        return 0
    if args.summary_only:
        print("[summary]", write_summary(project, dataset, specs, sparse_eval=args.sparse_eval))
        return 0

    project.mkdir(parents=True, exist_ok=True)
    statuses = []
    for s in specs:
        try:
            statuses.append(train_one(args, dataset, s, project, overrides))
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {s.name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            statuses.append({"model": s.name, "status": "failed", "error": str(exc)})
            if args.stop_on_failure:
                break
        finally:
            try:
                write_summary(project, dataset, specs, sparse_eval=args.sparse_eval)
            except OSError as e:
                print(f"[summary-warn] {e}", flush=True)

    print(f"\n[reproduce:{dataset.name}] DONE")
    for st in statuses:
        print("  ", st)
    ok = {"ok", "resumed", "skipped"}
    return 0 if all(st.get("status") in ok for st in statuses) else 1
