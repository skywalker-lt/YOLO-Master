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
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import DatasetSpec, ModelSpec, MODELS, run_dataset  # noqa: E402

# The shared two baselines plus the AI-TOD tiny-object variants.
MODELS_AITOD = MODELS + (
    # EsMoE-P2-N: ES_MOE base + P2/4 head. uses_esmoe=True -> --no-sparse-eval gives it the
    # corrected dense evaluation too (else its sparse-eval mAP collapses like EsMoE-N).
    ModelSpec("EsMoE-P2-N", "ultralytics/cfg/models/master/v0/det/yolo-master-n-p2.yaml", uses_esmoe=True),
    # v0.1 base (stable OptimizedMOEImproved) + P2/4 head -> tiny-object variant, no ES_MOE
    # routing collapse. uses_esmoe=False (train==eval consistent; --no-sparse-eval is a no-op).
    ModelSpec("v0.1-P2-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2.yaml", uses_esmoe=False),
    # UltraOptimizedMoE swaps: v0.1 backbone/head with the 3 ModularRouterExpertMoE blocks
    # replaced by UltraOptimizedMoE (shared expert -> stable, batched sparse compute + ultra-light
    # router -> ~20-30% fewer GFLOPs at equal params). Sparse train==eval, so uses_esmoe=False.
    ModelSpec("UoMoE-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-uomoe.yaml", uses_esmoe=False),
    ModelSpec("UoMoE-P2-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2-uomoe.yaml", uses_esmoe=False),
)

DATASET = DatasetSpec(
    name="AI-TOD-v2",
    data="AI-TOD-v2.yaml",
    project="runs/reproduce/aitodv2",
)


if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET, models=MODELS_AITOD))
