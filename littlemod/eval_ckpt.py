"""Clean dense-MoE eval of one or more checkpoints via the calibrated validator (full + small mAP).

Task #5: identify the figure-exact EsMoE-N baseline among candidate checkpoints. The dir named
`baseline/EsMoE-N` shares weights with `result-esmoen-sparse-visdrone` (the SPARSE-trained variant,
dense-evals 0.325); the plain non-sparse EsMoE-N (`result-esmoen-visdrone`) should hit the figure's
0.3499. Reuses step2_val.run(mode="dense") — no masking, dense MoE (use_sparse_inference=False) — so
these numbers are directly comparable to the base/dense/sparse rows.

  python -m littlemod.eval_ckpt \
      runs/baseline/EsMoE-N_VisDrone/weights/best.pt \
      scripts/reproduce/results/result-esmoen-visdrone/weights/best.pt \
      --datasets-dir /data/datasets
"""
from __future__ import annotations

import argparse

from ultralytics.utils import LOGGER, SETTINGS

from littlemod.step2_val import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+", help="one or more checkpoints to eval clean")
    ap.add_argument("--data", default="VisDrone.yaml")
    ap.add_argument("--datasets-dir", default="/data/datasets")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--small-thresh", type=float, default=64.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    SETTINGS["datasets_dir"] = a.datasets_dir
    st = int(a.small_thresh)
    LOGGER.info("[eval] clean dense-MoE eval (official validator, rect=True, --no-sparse)")
    for w in a.weights:
        f50, f, s50, s = run(w, "dense", 16, 0.15, a.small_thresh, a.data, a.imgsz, a.batch, a.workers)
        LOGGER.info(f"  {w}")
        LOGGER.info(f"      full mAP50={f50:.4f} mAP50-95={f:.4f} | small<{st}px mAP50={s50:.4f} mAP50-95={s:.4f}")


if __name__ == "__main__":
    main()
