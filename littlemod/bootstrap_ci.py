"""Paired bootstrap 95% CI for Δ(small mAP50) = causal-gated − dense  (0 GPU).

The eval is deterministic (same weights+images -> identical predictions), so the ONLY noise in the Δ is
which images landed in val. This resamples the val images with replacement — PAIRED: the same resampled
image set scores both conditions each draw — recomputes small mAP50 with the SAME ap_per_class the
validator uses, and reports the Δ distribution + 95% CI. If the CI spans 0, "sparse beats dense" is not
significant on this val set (retract it, keep the parity claim).

Feed it the dump dir from `step2_val ... --dump-dir <dir>` (writes dense.pkl + causal_rho{rho}.pkl):
  python -m littlemod.bootstrap_ci runs/littlemod/bootstrap --rho 0.3 --n 2000
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

from ultralytics.utils.metrics import ap_per_class


def small50(stats, idx):
    """small mAP50 over the images in `idx` (with-replacement allowed), via the validator's matcher."""
    if len(idx) == 0:
        return 0.0
    tp = np.concatenate([stats["tp"][i] for i in idx])
    cf = np.concatenate([stats["conf"][i] for i in idx])
    pc = np.concatenate([stats["pred_cls"][i] for i in idx])
    tc = np.concatenate([stats["target_cls"][i] for i in idx])
    if tc.size == 0:
        return 0.0
    r = ap_per_class(tp, cf, pc, tc, plot=False)
    ap = np.asarray(r[5])                         # [n_cls, n_iou]; col 0 = AP50
    return float(ap[:, 0].mean()) if ap.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--rho", type=float, default=0.3, help="which causal_rho{rho}.pkl to test")
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with open(f"{a.dump_dir}/dense.pkl", "rb") as f:
        dense = pickle.load(f)
    with open(f"{a.dump_dir}/causal_rho{a.rho}.pkl", "rb") as f:
        causal = pickle.load(f)
    N = len(dense["tp"])
    assert len(causal["tp"]) == N, f"image-count mismatch: dense {N} vs causal {len(causal['tp'])}"

    d0, c0 = small50(dense, range(N)), small50(causal, range(N))
    print(f"[bootstrap] {N} val images, rho={a.rho}")
    print(f"  point estimate: dense={d0:.4f}  causal-gated={c0:.4f}  Δ={c0 - d0:+.4f}")

    rng = np.random.default_rng(a.seed)
    deltas = np.empty(a.n)
    for bi in range(a.n):
        idx = rng.integers(0, N, N)               # paired resample
        deltas[bi] = small50(causal, idx) - small50(dense, idx)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"  paired bootstrap (n={a.n}): Δ mean={deltas.mean():+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  "
          f"P(Δ≤0)={(deltas <= 0).mean():.3f}")
    if lo > 0:
        print("  VERDICT: SIGNIFICANT — Δ>0 across the CI; the 'sparse beats dense' (denoising) claim holds.")
    elif hi < 0:
        print("  VERDICT: sparse is significantly WORSE than dense.")
    else:
        print("  VERDICT: NOT SIGNIFICANT — CI spans 0. Retract 'beats dense'; keep the PARITY claim.")


if __name__ == "__main__":
    main()
