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
registers a training callback that flips `ES_MOE.use_sparse_inference=False` on
both the live model and its EMA at `on_pretrain_routine_end` (before any
validation and before checkpoints are written from the EMA), so per-epoch val,
the saved .pt, and final eval all use the same dense forward as training.
v0.1-N has no ES_MOE modules, so the flag is a no-op there.

Multi-GPU (DDP)
---------------
Launch the SAME script under ``torchrun`` for data-parallel multi-GPU training::

    torchrun --nproc_per_node=4 scripts/reproduce/reproduce_visdrone.py --batch 64 ...

torchrun is REQUIRED (rather than Ultralytics' ``--device 0,1,2,3`` auto-spawn):
the auto-spawn regenerates the trainer in a fresh subprocess from its args alone,
which DROPS the Python callbacks these scripts attach -- the ES_MOE dense-eval fix
and the W&B logger -- silently reintroducing the EsMoE mAP collapse. Under torchrun
our script runs in every rank, so those callbacks are present on all of them.

``--batch`` is the TOTAL batch, split evenly across GPUs (must be divisible by the
GPU count). Rank 0 owns all side-effects: console logs, W&B, and summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Distributed (DDP) context -- populated by torchrun before the process starts #
# --------------------------------------------------------------------------- #
RANK = int(os.environ.get("RANK", "-1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "-1"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
UNDER_TORCHRUN = "LOCAL_RANK" in os.environ


def is_main_process() -> bool:
    """True on the single process (non-DDP) or on global rank 0 (DDP)."""
    return RANK in (-1, 0)


def _ddp_device(requested: str) -> str:
    """Under torchrun, force ``device`` to span the whole world (GPUs 0..WORLD_SIZE-1).

    Ultralytics derives its ``world_size`` from ``args.device``; under torchrun the
    real world size comes from the launcher, so the device string must list one GPU
    per rank or every rank would collide on a single GPU. Outside torchrun the
    caller's value is passed through untouched (single-GPU or CPU as given).
    """
    if UNDER_TORCHRUN and WORLD_SIZE > 1:
        return ",".join(str(i) for i in range(WORLD_SIZE))
    return requested


def _teardown_ddp() -> None:
    """Destroy the process group between models so the next model can re-init it.

    Ultralytics' ``_do_train`` never tears the group down; when we train several
    models in one torchrun invocation (``--model both``) the second ``_setup_ddp``
    would raise 'process group already initialized'. The barrier lets rank 0 finish
    its final_eval before any rank tears down. No-op outside DDP.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:  # noqa: BLE001  -- teardown is best-effort; never mask a training error
        pass


def _device_gpu_count(device) -> int:
    """Number of GPUs implied by a --device value ('0,1,2,3' -> 4; '0'/'cpu' -> 1)."""
    if isinstance(device, (list, tuple)):
        return len(device)
    if isinstance(device, str) and device not in ("", "cpu", "mps"):
        return len([d for d in device.split(",") if d != ""])
    return 1


def _prestage_dataset(data: str) -> None:
    """Download + prepare the dataset ONCE, in this single parent process.

    Ultralytics initializes the DDP process group only inside ``_do_train`` -- AFTER
    the trainer's ``__init__`` dataset check -- so its ``torch_distributed_zero_first``
    download-once barrier is a no-op under torchrun (the group isn't up yet). Without
    this, every rank would download and, worse, run the label conversion concurrently
    (e.g. VisDrone ``visdrone2yolo`` moves images and rmtree's source dirs), which
    races and can corrupt a first-time dataset build. Native auto-spawn avoids it
    because the single parent prepares data before any subprocess starts; we do the
    same here, right before relaunching under torchrun. No-op once the data exists.
    """
    try:
        from ultralytics.data.utils import check_det_dataset

        print(f"[reproduce] pre-staging dataset '{data}' in one process before DDP launch "
              f"(avoids concurrent multi-rank download/convert races)...", flush=True)
        check_det_dataset(data, autodownload=True)
    except Exception as exc:  # noqa: BLE001  -- fall back to per-rank check rather than block training
        print(f"[reproduce][WARN] dataset pre-stage failed ({type(exc).__name__}: {exc}); "
              f"ranks will each run their own check (first-time build may race).", flush=True)


def _maybe_reexec_under_torchrun(args: argparse.Namespace, data: str | None = None) -> None:
    """Re-exec this script under torchrun when a multi-GPU --device is requested.

    Keeps the simple ``--device 0,1,2,3`` UX while getting CORRECT DDP: launching
    via torchrun (``python -m torch.distributed.run``) means our script -- and the
    callbacks it attaches (ES_MOE dense-eval fix, W&B logger) -- runs in every rank,
    unlike Ultralytics' built-in auto-spawn which regenerates the trainer from args
    alone and drops those callbacks. ``os.execv`` replaces this process, so it never
    returns; each rank then re-enters with ``LOCAL_RANK`` set and trains directly.

    Before relaunching, the dataset is pre-staged once (see ``_prestage_dataset``) so
    the ranks never race on a first-time download/convert.

    No-op when already under torchrun, on a single GPU/CPU, or in a no-train utility
    mode (--dry-run / --check-build / --summary-only).
    """
    if UNDER_TORCHRUN:
        return
    if getattr(args, "dry_run", False) or getattr(args, "check_build", False) or getattr(args, "summary_only", False):
        return
    n = _device_gpu_count(args.device)
    if n <= 1:
        return
    if data:
        _prestage_dataset(data)
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n}",
           os.path.abspath(sys.argv[0]), *sys.argv[1:]]
    print(f"[reproduce] multi-GPU device={args.device!r} -> relaunching under torchrun ({n} ranks):\n"
          f"    {' '.join(cmd)}", flush=True)
    os.execv(sys.executable, cmd)  # replaces the current process; does not return


def _make_ddp_disable_buffer_broadcast_callback():
    """Return a callback that turns off DDP buffer broadcasting for MoE/MoA models.

    The MoE blocks (``ultralytics/nn/modules/moe/modules.py``) register per-rank stats
    buffers LAZILY during the first forward -- ``load_balancing_loss`` /
    ``expert_usage_counts`` / ``training_step`` -- and some are created with no device
    (e.g. ``torch.tensor(0.0)``), so they land on CPU after ``model.to(device)`` already
    ran. DDP's default per-forward buffer broadcast (rank0 -> others) then hands a CPU
    tensor to NCCL (CUDA-only) -> ``RuntimeError: No backend type associated with device
    type cpu`` on the 2nd iteration. These are non-persistent, per-rank statistics that
    must NOT be overwritten by rank 0's copy anyway, so disabling the broadcast is both
    the fix and the correct semantics. Fires on every rank after the model is DDP-wrapped
    (on_train_start), before the first forward.
    """
    from ultralytics.utils import LOGGER

    state = {"logged": False}

    def _apply(trainer):
        model = getattr(trainer, "model", None)
        # Only a DistributedDataParallel-wrapped model has ``broadcast_buffers``; on a
        # single GPU trainer.model is the plain module, so this is a safe no-op there.
        if model is not None and getattr(model, "broadcast_buffers", False):
            model.broadcast_buffers = False
            if not state["logged"]:
                LOGGER.info("[reproduce] DDP broadcast_buffers=False (MoE per-rank stats buffers are "
                            "registered lazily/on CPU; avoids the NCCL CPU-buffer broadcast crash)")
                state["logged"] = True

    return _apply


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


# Both datasets train the same two models. EsMoE-N gets dense validation.
MODELS = (
    ModelSpec("v0.1-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml", uses_esmoe=False),
    ModelSpec("EsMoE-N", "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml", uses_esmoe=True),
)


# --------------------------------------------------------------------------- #
# Dense-validation callback for ES_MOE                                         #
# --------------------------------------------------------------------------- #
def _make_dense_inference_callback():
    """Return a trainer callback that sets ES_MOE.use_sparse_inference=False.

    Applied to both trainer.model and trainer.ema.ema so per-epoch validation
    (which runs on the EMA), the EMA-derived checkpoints, and the final eval all
    take the dense forward path that matches training.
    """
    from ultralytics.nn.modules.moe.modules import ES_MOE
    from ultralytics.utils import LOGGER

    state = {"logged": False}

    def _apply(trainer):
        targets = []
        model = getattr(trainer, "model", None)
        if model is not None:
            targets.append(model)
        ema = getattr(trainer, "ema", None)
        if ema is not None and getattr(ema, "ema", None) is not None:
            targets.append(ema.ema)

        count = 0
        for target in targets:
            for module in target.modules():
                if isinstance(module, ES_MOE):
                    module.use_sparse_inference = False
                    count += 1
        if count and not state["logged"]:
            LOGGER.info(f"[reproduce] EsMoE dense validation enabled: "
                        f"use_sparse_inference=False on {count} ES_MOE module(s)")
            state["logged"] = True

    return _apply


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
        if not is_main_process():  # under DDP only rank 0 owns the W&B run
            return
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


def train_one(args: argparse.Namespace, dataset: DatasetSpec, spec: ModelSpec, project: Path) -> dict:
    from ultralytics import YOLO

    run_name = f"{dataset.name}_{spec.name}"
    run_dir = project / run_name
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"
    done = _completed_epoch(run_dir)

    if best_pt.exists() and done is not None and done + 1 >= args.epochs:
        if is_main_process():
            print(f"[skip] {run_name}: complete at epoch {done}", flush=True)
        return {"model": spec.name, "status": "skipped"}

    # DDP: --batch is the TOTAL batch; Ultralytics floor-divides it across ranks, so a
    # non-divisible batch silently drops the remainder. Warn (once, on rank 0).
    if WORLD_SIZE > 1 and args.batch % WORLD_SIZE and is_main_process():
        print(f"[reproduce][WARN] batch={args.batch} is not divisible by WORLD_SIZE={WORLD_SIZE}; "
              f"each rank gets {args.batch // WORLD_SIZE} (remainder dropped).", flush=True)

    # Corrected dense evaluation is opt-in via --no-sparse-eval, and only affects
    # ES_MOE models (v0.1-N has none, so it is a no-op there).
    dense_eval = spec.uses_esmoe and not args.sparse_eval
    if last_pt.exists() and done is not None:
        if is_main_process():
            print(f"[resume] {run_name}: {last_pt} epoch={done} -> {args.epochs}", flush=True)
        model = YOLO(str(last_pt))
        resume = True
    else:
        if is_main_process():
            print(f"[train] {run_name}: cfg={spec.cfg} data={dataset.data} "
                  f"sparse_eval={args.sparse_eval} dense_eval={dense_eval}"
                  f"{f' ddp_world={WORLD_SIZE}' if WORLD_SIZE > 1 else ''}", flush=True)
        model = YOLO(str(ROOT / spec.cfg))
        resume = False

    if dense_eval:
        cb = _make_dense_inference_callback()
        model.add_callback("on_pretrain_routine_end", cb)
        model.add_callback("on_train_start", cb)

    # DDP: disable per-forward buffer broadcast so NCCL never tries to broadcast a MoE
    # stats buffer that was lazily registered on CPU (crashes on the 2nd iteration).
    if WORLD_SIZE > 1:
        model.add_callback("on_train_start", _make_ddp_disable_buffer_broadcast_callback())

    if args.wandb and args.wandb_mode != "disabled":
        for event, fn in _make_wandb_callbacks(run_name, dataset, spec, args, dense_eval).items():
            model.add_callback(event, fn)

    start = time.time()
    model.train(
        data=dataset.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=_ddp_device(args.device),
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=run_name,
        exist_ok=True,
        pretrained=False,
        lora_r=0,  # full from-scratch baseline: repo default.yaml ships lora_r=16, which would
                   # silently LoRA-fy the run (train ~24% of params). r=0 disables LoRA (apply_lora no-op).
        optimizer="auto",  # match the VisDrone/SKU baselines: repo default.yaml drifted to AdamW,
                           # but auto -> SGD@0.01 (mom 0.9, warmup_bias_lr 0) for long runs. AdamW@0.01
                           # (10x too high) is what NaN'd AI-TOD EsMoE-N and stuck mAP at 0.
        val=True,
        plots=True,
        cache=args.cache,
        patience=args.patience,
        amp=args.amp,
        resume=resume,
        verbose=args.verbose,
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
                        "triggers multi-GPU DDP by auto-relaunching this script under torchrun. "
                        "'cpu' for CPU.")
    p.add_argument("--workers", type=int, default=16, help="Dataloader workers PER GPU under DDP.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
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


def run_dataset(dataset: DatasetSpec, models=MODELS) -> int:
    """Entry point used by the per-dataset scripts."""
    args = build_parser(dataset, models).parse_args()

    # Multi-GPU: if --device lists several GPUs and we're not already under torchrun,
    # pre-stage the dataset once, then relaunch the whole script under torchrun so every
    # rank keeps this script's callbacks. Replaces the process (never returns) for the
    # training path.
    _maybe_reexec_under_torchrun(args, data=dataset.data)

    project = Path(args.project) if Path(args.project).is_absolute() else ROOT / args.project
    specs = list(models) if args.model == "both" else [m for m in models if m.name == args.model]

    if is_main_process():
        wandb_desc = "off" if (not args.wandb or args.wandb_mode == "disabled") else args.wandb_mode
        ddp_desc = f"  ddp={WORLD_SIZE}x (torchrun)" if WORLD_SIZE > 1 else ""
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
            if is_main_process():
                print(f"[build-ok] {s.name}: {sum(p.numel() for p in m.parameters()) / 1e6:.3f}M  ({s.cfg})")
        return 0
    if args.summary_only:
        if is_main_process():
            print("[summary]", write_summary(project, dataset, specs, sparse_eval=args.sparse_eval))
        return 0

    project.mkdir(parents=True, exist_ok=True)
    statuses = []
    for s in specs:
        try:
            statuses.append(train_one(args, dataset, s, project))
        except Exception as exc:  # noqa: BLE001
            rank_tag = f" (rank {RANK})" if WORLD_SIZE > 1 else ""
            print(f"[fail] {s.name}{rank_tag}: {type(exc).__name__}: {exc}", flush=True)
            if is_main_process():
                traceback.print_exc()
            statuses.append({"model": s.name, "status": "failed", "error": str(exc)})
            if args.stop_on_failure:
                break
        finally:
            if is_main_process():
                try:
                    write_summary(project, dataset, specs, sparse_eval=args.sparse_eval)
                except OSError as e:
                    print(f"[summary-warn] {e}", flush=True)
            # Tear the DDP group down between models so the next model can re-init it
            # (Ultralytics leaves it up). All ranks must call this; no-op outside DDP.
            _teardown_ddp()

    if is_main_process():
        print(f"\n[reproduce:{dataset.name}] DONE")
        for st in statuses:
            print("  ", st)
    ok = {"ok", "resumed", "skipped"}
    return 0 if all(st.get("status") in ok for st in statuses) else 1
