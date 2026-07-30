#!/usr/bin/env python3
"""Reproduce the YOLO-Master nano models on SKU-110K.

SKU-110K (retail, dense products, single class), built-in config SKU-110K.yaml.
The two shared nano baselines (v0.1-N, EsMoE-N) plus the four shared variants
(EsMoE-P2-N, v0.1-P2-N, UoMoE-N, UoMoE-P2-N -- see _reproduce_common.VARIANTS),
exposed here for a consistent model set across every dataset. Note SKU-110K is
large-object retail, so the P2/4 tiny-object head is not expected to help; the
variants are provided for completeness and are not tuned/tested on this dataset.
By default the models are reproduced as-is (the ES_MOE models keep their sparse
eval, which collapses mAP). Add --no-sparse-eval to opt into the corrected dense
evaluation for the EsMoE models (train==eval); the v0.1 and UoMoE models are unaffected.

Examples:
    python scripts/reproduce/reproduce_sku110k.py --check-build
    python scripts/reproduce/reproduce_sku110k.py --epochs 300 --batch 64                  # as-is
    python scripts/reproduce/reproduce_sku110k.py --model EsMoE-N --no-sparse-eval         # corrected
    python scripts/reproduce/reproduce_sku110k.py --model v0.1-N --no-wandb
    python scripts/reproduce/reproduce_sku110k.py --wandb-project my-proj --wandb-mode offline
    # Multi-GPU DDP (Ultralytics native): comma-list --device; --batch is the TOTAL, split across GPUs.
    python scripts/reproduce/reproduce_sku110k.py --device 0,1,2,3 --batch 128 --epochs 300
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import MODELS, VARIANTS, DatasetSpec, run_dataset  # noqa: E402

# The two shared baselines plus the four shared variants -- full, consistent model set.
MODELS_SKU110K = MODELS + VARIANTS

DATASET = DatasetSpec(
    name="SKU-110K",
    data="SKU-110K.yaml",
    project="runs/reproduce/sku110k",
)


if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET, models=MODELS_SKU110K))
