#!/usr/bin/env python3
"""Reproduce YOLO-Master nano baselines on VisDrone.

VisDrone (aerial, dense small objects), built-in config VisDrone.yaml. The two
shared nano baselines (v0.1-N, EsMoE-N) plus the UltraOptimizedMoE swaps UoMoE-N
and UoMoE-P2-N (deployment-optimized MoE -- shared expert like v0.1, ultra-light
router + batched compute; sparse train==eval, so uses_esmoe=False and
--no-sparse-eval is a no-op for them). By default the models are reproduced as-is
(EsMoE-N keeps its sparse eval, which collapses mAP). Add --no-sparse-eval to opt
into the corrected dense evaluation for EsMoE-N (train==eval); the v0.1 and UoMoE
models are unaffected.

Examples:
    python scripts/reproduce/reproduce_visdrone.py --check-build
    python scripts/reproduce/reproduce_visdrone.py --epochs 300 --batch 64                 # as-is
    python scripts/reproduce/reproduce_visdrone.py --model EsMoE-N --no-sparse-eval        # corrected
    python scripts/reproduce/reproduce_visdrone.py --model UoMoE-N --epochs 300            # UltraOptimizedMoE
    python scripts/reproduce/reproduce_visdrone.py --model v0.1-N --no-wandb
    python scripts/reproduce/reproduce_visdrone.py --wandb-project my-proj --wandb-mode offline
    # Multi-GPU DDP (Ultralytics native): comma-list --device; --batch is the TOTAL, split across GPUs.
    python scripts/reproduce/reproduce_visdrone.py --model UoMoE-N --device 0,1,2,3 --batch 128
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import MODELS, VARIANTS, DatasetSpec, run_dataset  # noqa: E402

# The two shared baselines plus the four shared variants (P2/4-head + UltraOptimizedMoE
# swaps -- see _reproduce_common.VARIANTS). Full, consistent model set across datasets.
MODELS_VISDRONE = MODELS + VARIANTS

DATASET = DatasetSpec(
    name="VisDrone",
    data="VisDrone.yaml",
    project="runs/reproduce/visdrone",
)


if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET, models=MODELS_VISDRONE))
