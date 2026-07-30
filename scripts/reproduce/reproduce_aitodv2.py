#!/usr/bin/env python3
"""Reproduce YOLO-Master nano baselines on AI-TOD-v2.

AI-TOD-v2 (aerial tiny-object detection, 8 classes, ~800px crops), config
AI-TOD-v2.yaml. The two shared nano baselines (v0.1-N, EsMoE-N) plus the AI-TOD
tiny-object variants: EsMoE-P2-N and v0.1-P2-N (extra P2/4 head), and the
UltraOptimizedMoE swaps UoMoE-N / UoMoE-P2-N (deployment-optimized MoE — shared
expert like v0.1, ultra-light router + batched compute, ~20-30% fewer GFLOPs at
equal params; sparse train==eval, so uses_esmoe=False and --no-sparse-eval is a
no-op for them). By default the models are reproduced as-is (the ES_MOE models
keep their sparse eval, which collapses mAP). Add --no-sparse-eval to opt into
the corrected dense evaluation for the EsMoE models (train==eval); the v0.1 and
UoMoE models are unaffected.

Examples:
    python scripts/reproduce/reproduce_aitodv2.py --check-build
    python scripts/reproduce/reproduce_aitodv2.py --epochs 300 --batch 64                  # as-is
    python scripts/reproduce/reproduce_aitodv2.py --model EsMoE-P2-N --no-sparse-eval       # corrected
    python scripts/reproduce/reproduce_aitodv2.py --model UoMoE-P2-N --epochs 300           # UltraOptimizedMoE + P2
    python scripts/reproduce/reproduce_aitodv2.py --model v0.1-N --no-wandb
    python scripts/reproduce/reproduce_aitodv2.py --wandb-project my-proj --wandb-mode offline
    # Multi-GPU DDP (Ultralytics native): comma-list --device; --batch is the TOTAL, split across GPUs.
    python scripts/reproduce/reproduce_aitodv2.py --model UoMoE-P2-N --device 0,1,2,3 --batch 128
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import MODELS, VARIANTS, DatasetSpec, run_dataset  # noqa: E402

# The two shared baselines plus the four shared variants (P2/4-head + UltraOptimizedMoE
# swaps -- see _reproduce_common.VARIANTS). The P2 variants are the tiny-object heads that
# matter most on AI-TOD-v2. EsMoE-P2-N is uses_esmoe=True, so --no-sparse-eval gives it the
# corrected dense evaluation too (else its sparse-eval mAP collapses like EsMoE-N).
MODELS_AITOD = MODELS + VARIANTS

DATASET = DatasetSpec(
    name="AI-TOD-v2",
    data="AI-TOD-v2.yaml",
    project="runs/reproduce/aitodv2",
)


if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET, models=MODELS_AITOD))
