#!/usr/bin/env python3
"""Reproduce YOLO-Master-v0.1-N, EsMoE-N, and EsMoE-P2-N baselines on AI-TOD-v2.

AI-TOD-v2 (aerial tiny-object detection, 8 classes, ~800px crops), config
AI-TOD-v2.yaml. Same two nano baselines as VisDrone/SKU-110K, plus EsMoE-P2-N
(the extra P2/4 head). By default the models are reproduced as-is (the ES_MOE
models keep their sparse eval, which collapses mAP). Add --no-sparse-eval to opt
into the corrected dense evaluation for EsMoE-N and EsMoE-P2-N (train==eval);
v0.1-N is unaffected.

Examples:
    python scripts/reproduce/reproduce_aitodv2.py --check-build
    python scripts/reproduce/reproduce_aitodv2.py --epochs 300 --batch 64                  # as-is
    python scripts/reproduce/reproduce_aitodv2.py --model EsMoE-P2-N --no-sparse-eval       # corrected
    python scripts/reproduce/reproduce_aitodv2.py --model v0.1-N --no-wandb
    python scripts/reproduce/reproduce_aitodv2.py --wandb-project my-proj --wandb-mode offline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import DatasetSpec, ModelSpec, MODELS, run_dataset  # noqa: E402

# The shared two baselines plus EsMoE-P2-N (P2/4 head variant). It has ES_MOE
# blocks, so uses_esmoe=True -> --no-sparse-eval gives it the corrected dense
# evaluation too (else its sparse-eval mAP collapses like EsMoE-N).
MODELS_AITOD = MODELS + (
    ModelSpec("EsMoE-P2-N", "ultralytics/cfg/models/master/v0/det/yolo-master-n-p2.yaml", uses_esmoe=True),
    # v0.1 base (stable OptimizedMOEImproved) + P2/4 head -> tiny-object variant, no ES_MOE
    # routing collapse. uses_esmoe=False (train==eval consistent; --no-sparse-eval is a no-op).
    ModelSpec("v0.1-P2-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2.yaml", uses_esmoe=False),
)

DATASET = DatasetSpec(
    name="AI-TOD-v2",
    data="AI-TOD-v2.yaml",
    project="runs/reproduce/aitodv2",
)


if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET, models=MODELS_AITOD))
