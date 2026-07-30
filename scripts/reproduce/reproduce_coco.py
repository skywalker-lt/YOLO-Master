#!/usr/bin/env python3
"""Train YOLO-Master UoMoE-N on full COCO from scratch (with an optional matched baseline).

Why this script exists
----------------------
There is no published UoMoE checkpoint anywhere: the `UltraOptimizedMoE` body is
upstream code (Tencent, 3d3dd7a, 2026-01-04) but the `yolo-master-n-uomoe.yaml`
config is ours (70e65dd), and no release asset or README row covers it. So a COCO
UoMoE-N is a new result. Default is a single UoMoE-N run; the matched v0.1-N
baseline is opt-in (`--model both`) for when a controlled comparison is warranted.

How to use this (scout first, control later)
-------------------------------------------
Default is UoMoE-N alone -- one run, ~$500, is enough to learn whether UoMoE is
even interesting on COCO. Only if it looks good is it worth paying for the
controlled baseline (`--model both`, on the same machine, same settings).

Reading the scout run against the published numbers is ASYMMETRIC, because ours is
from-scratch and theirs is warm-started (see below):

  * UoMoE-N >= ~0.427  ->  MEANINGFUL. It matched a warm-started model from
                           scratch, so the architecture is doing real work.
  * UoMoE-N <  ~0.427  ->  INCONCLUSIVE. Could be the missing pretrained init
                           rather than the architecture. That is when you run
                           `--model both` to settle it.

The comparison problem
----------------------
The released nano weights report COCO val mAP50-95 of 0.4289 (v0.1-N) and 0.4269
(EsMoE-N). Their embedded `train_args` show how they were produced:

    data=coco.yaml  epochs=700  batch=256  imgsz=640  optimizer=auto(->SGD)
    lr0=0.01  lrf=0.01  cos_lr=False  close_mosaic=10  copy_paste=0.1
    pretrained=TRUE   model=<a previous run's last.pt>

That `pretrained=True` plus an init from an earlier checkpoint means those numbers
are NOT from-scratch runs, and that init is not reproducible from this repo.
Comparing a from-scratch UoMoE-N against them would handicap our own model.

So `--model both` trains both from scratch (`pretrained=False`, inherited from
_reproduce_common) under byte-identical settings, and UoMoE-N is compared to the
v0.1-N produced HERE -- not to the published number. That is the control; run it
when the scout result justifies the second $500.

Protocol
--------
Matches the released COCO recipe except for the initialisation:

    epochs 700   batch 256   imgsz 640   SGD lr0 0.01 lrf 0.01
    momentum 0.937   weight_decay 0.0005   warmup_epochs 3.0
    cos_lr False   close_mosaic 10   copy_paste 0.1
    mosaic 1.0  mixup 0.0  scale 0.5  translate 0.1  fliplr 0.5  degrees 0
    pretrained False   lora_r 0   deterministic   seed 42   patience 0

`cos_lr=False` and `copy_paste=0.1` are pinned HERE because default.yaml is tuned
for the paper's LoRA/MoLoRA experiments and ships cos_lr=True / copy_paste=0.0;
without these pins the run would silently deviate from the published protocol.
`pretrained=False` and `lora_r=0` (default.yaml ships lora_r=16) come from
_reproduce_common, as does optimizer='auto' -- never default.yaml's AdamW@0.01,
which is 10x too high and NaN'd the AI-TOD baselines.

Cost
----
700 epochs x 118,287 images = ~83M image-passes. On 8xA100 expect roughly 40-45 h
per model (~$500 at $12/h); nano YOLO training is usually CPU-bound on mosaic, so provision vCPUs,
not just GPUs (`workers` is auto-set below to cpu_count // n_gpus, which is also
the ceiling `build_dataloader` enforces).

Calibrate before committing: run 3 epochs, take the SECOND or third epoch time
(the first includes the label-cache scan), multiply by 700. Watch nvidia-smi -- if
GPU util sits under ~60% you are dataloader-starved and a faster GPU buys nothing.

Examples
--------
    # calibration (~20 min) -- same settings, 3 epochs
    python scripts/reproduce/reproduce_coco.py --model UoMoE-N --epochs 3 --device 0,1,2,3,4,5,6,7

    # scout run -- UoMoE-N alone (the default)
    python scripts/reproduce/reproduce_coco.py --device 0,1,2,3,4,5,6,7

    # controlled pair, same machine/settings -- only if the scout looks good
    python scripts/reproduce/reproduce_coco.py --model both --device 0,1,2,3,4,5,6,7
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import MODELS, VARIANTS, DatasetSpec, run_dataset  # noqa: E402

# The matched pair: our new UoMoE-N and the v0.1-N baseline it is measured against.
# Both are OptimizedMOEImproved/UltraOptimizedMoE (sparse train==eval), so
# --no-sparse-eval is a no-op for both and the ES_MoE eval-collapse trap does not apply.
_BY_NAME = {m.name: m for m in MODELS + VARIANTS}
MODELS_COCO = (_BY_NAME["UoMoE-N"], _BY_NAME["v0.1-N"])

DATASET = DatasetSpec(
    name="COCO",
    data="coco.yaml",
    project="runs/reproduce/coco",
)

# Pinned to the released COCO runs' train_args, applied after the shared kwargs so they
# beat default.yaml. Everything else in the recipe already matches default.yaml.
COCO_OVERRIDES = {
    "cos_lr": False,      # default.yaml ships True; the released COCO runs used linear decay
    "copy_paste": 0.1,    # default.yaml ships 0.0; the released COCO runs used 0.1
}


def _default_workers(device: str) -> int:
    """cpu_count // n_gpus -- the same ceiling build_dataloader applies.

    Asking for more is silently clamped, and asking for less leaves the augmentation
    pipeline (4 JPEG decodes + warp per mosaic sample) short, which is the usual
    bottleneck at nano scale.
    """
    n_gpu = max(1, len([d for d in str(device).split(",") if d.strip() != ""]))
    return max(1, (os.cpu_count() or 8) // n_gpu)


if __name__ == "__main__":
    # COCO needs different defaults than the vertical-dataset scripts (which default to
    # 300 epochs / batch 64). Inject them only if the user did not pass them explicitly,
    # so an override on the command line always wins.
    if not any(a.startswith("--epochs") for a in sys.argv[1:]):
        sys.argv += ["--epochs", "700"]
    if not any(a.startswith("--batch") for a in sys.argv[1:]):
        sys.argv += ["--batch", "256"]
    if not any(a.startswith("--patience") for a in sys.argv[1:]):
        sys.argv += ["--patience", "0"]        # never early-stop a baseline
    if not any(a.startswith("--model") for a in sys.argv[1:]):
        sys.argv += ["--model", "UoMoE-N"]     # scout run first; --model both for the controlled pair
    if not any(a.startswith("--workers") for a in sys.argv[1:]):
        dev = "0"
        for i, a in enumerate(sys.argv[1:]):
            if a == "--device" and i + 2 <= len(sys.argv[1:]):
                dev = sys.argv[i + 2]
            elif a.startswith("--device="):
                dev = a.split("=", 1)[1]
        sys.argv += ["--workers", str(_default_workers(dev))]

    print(f"[reproduce:COCO] protocol pins: {COCO_OVERRIDES}  "
          f"(pretrained=False, lora_r=0, optimizer=auto->SGD from _reproduce_common)")
    raise SystemExit(run_dataset(DATASET, models=MODELS_COCO, overrides=COCO_OVERRIDES))
