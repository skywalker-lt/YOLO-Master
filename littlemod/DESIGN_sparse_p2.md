# Sparse-P2 branch — design note (read before building)

## What the trained model actually is (EsMoE-N-TPH, dumped from best.pt)

4-scale detect `[21,24,27,30]` at strides `[4,8,16,32]`. The stride-4 (P2) path in the head:

```
18: C3k2[256]  (N3, stride 8)
19: Upsample   -> stride 4
20: Concat[-1, 3]   (+ backbone P2, layer 3, stride 4)
21: C3k2[128]  (stride 4, ~160x160)   <-- THE expensive op, ~all of the +3.9 GFLOPs
22: Conv[128] s2  -> stride 8          <-- feeds the PAN
23: Concat[-1, 18]
24: C3k2[256]  (detect P3, stride 8)
...
31: Detect[21,24,27,30]
```

**Entanglement:** layer 21 feeds BOTH the P2 detect output AND layer 22, which flows back
up the PAN into the P3/P4/P5 detect features. So the P2 feature is not an isolated head.

## Why this matters for our result

The calibrated 89%/90% retention (`RESULTS.md`) masked **only the P2 detect outputs** — the P2
feature stayed computed densely, so the PAN was fully intact. That is the accuracy value of the P2
*detections*. A naive "compute layer 21 only in selected cells" also **starves the PAN** in unselected
cells — a different model we have NOT measured. The two must not be conflated.

## Reuse audit (ONE RULE)

- ES_MOE `use_sparse_inference`: routes top-k **experts** (image-level importance), each computed
  **densely** over the whole map. NOT a spatial-cell gather. Cannot be reused for spatial P2 routing.
- `routers.py` (EfficientSpatialRouter/AdaptiveRoutingLayer): pool to **image-level** expert logits.
  Not spatial-cell selection.
- No submanifold / sparse-conv / gather-conv / active-sites primitive anywhere in `nn/`.
- Reusable: `DensityRouter.select_topk` (static top-k, TRT-safe), ultralytics `Conv`/`C3k2`, torch
  `gather`/`index_select`/`scatter` primitives. The gather-conv wrapper itself is new (unavoidable).

## Two architectures

### Design A — decoupled parallel gated P2 head  (RECOMMENDED)
Make the stride-4 head parallel: fed from N3 (layer 18) upsampled, gated by the router, feeding **only**
the P2 detect. The PAN reverts to the clean 3-scale baseline (P3/P4/P5 no longer depend on P2).
- + Gating is LOCAL: compute the P2 head only in selected cells; clean FLOP accounting; static shapes.
- + Our 89% masking measurement (PAN intact) is the valid upper bound for exactly this design.
- + **Removes the co-adaptation pathology**: because P3/P4/P5 = baseline behaviour, a gated-off cell is
  covered by the baseline heads — the rho=0.1 sub-baseline dip cannot happen by construction.
- - New architecture -> from-scratch VisDrone training. P2-head value may differ from the entangled one
  (must confirm the dense decoupled head still reaches ~0.382 before adding sparsity).

### Design B — gate the existing entangled layer 21
Keep EsMoE-N-TPH; gather-compute layer 21 in selected cells, scatter-zero elsewhere; layer 22/PAN see
zero-filled P2.
- + Reuses the trained model (finetune, not from-scratch).
- - Zero-fill is OOD for the dense-trained layer 22; accuracy semantics differ from what we measured;
  co-adaptation dip persists. Messier FLOP accounting (PAN partially sparse).

## VERDICT (entanglement probe, oracle, `step2_val --gate-feature`)

PAN-starvation cost is ZERO-to-NEGATIVE — feature-gating **matches/beats** detect-only masking:

| rho | A detect-only small50 | B feature-gated small50 | A-B |
|---|---|---|---|
| 0.15 | 0.3355 (64%) | 0.3353 (63%) | +0.0002 |
| 0.20 | 0.3443 (90%) | 0.3482 (**102%**) | -0.0039 |
| 0.30 | 0.3467 (98%) | 0.3531 (**117%**) | -0.0064 |

At rho=0.3 gated-sparse P2 (0.3531) **exceeds dense P2 (0.3475)**. Mechanism: dense P2 injects
spurious small-object activations into the PAN in empty regions -> FPs at P3/P4/P5; zeroing P2 where
there's nothing removes that noise. Monotonic in rho -> real denoising, not noise.

**DECISION: Design B.** The entanglement is beneficial. Gate the existing trained EsMoE-N-TPH — no
decouple, no from-scratch training. The detector needs no retrain for accuracy (inference gating of
the dense-trained model already matches/beats dense). Remaining work:

1. **Causal router.** The router must gate layer 21 from features available BEFORE it — N3 (layer 18,
   stride 8) and N4 (layer 15, stride 16), NOT the post-P2 detect inputs (24/27) the current router
   used. Retrain the router (frozen detector, cheap — like step 1) on layers 15/18.
2. **Gather-conv sparse layer 21** for the realized FLOP win: gather selected cells (+halo for the 3x3
   receptive field), run C3k2 on them, scatter back (zeros elsewhere = exactly the probe's semantics).
   The probe is a slight UPPER bound on a no-halo gather; with halo it matches. Static top-k -> static
   shapes (TRT/edge-safe).
3. Validate: causal-router + feature-gating accuracy (probe extension), then gather-conv realized
   GFLOPs vs the 12.4 dense. Target ~9.3 GFLOPs for >=100% of the dense small-mAP.

## FLOP attribution (measured) — B3 targets the DETECT HEAD, not layer 21

Per-layer MAC count on the P2 model (total 12.33 GFLOPs, baseline 8.5, +3.9 from P2):

| component | GFLOPs | note |
|---|---|---|
| P2 neck path (layers 19–22) | 0.55 | layer 21 C3k2 = 0.43, layer 22 = 0.12 — TRIVIAL |
| **Detect cv2 @ P2/4 (160×160)** | **3.04** | box/DFL branch — **78% of the +3.9, the real cost** |
| Detect cv3 @ P2/4 | 0.15 | cls branch |

So sparsifying layer 21 saves ~nothing. **B3 must sparsify the Detect head's P2 branch (cv2/cv3).**

**This is clean and bit-exact:** the P2 detect outputs for unselected cells are masked (discarded) anyway,
so skipping their cv2/cv3 compute is exactly compute-then-mask. Keep layer 21 DENSE (cheap, 0.43G) so
its output is a full feature map → the detect gather has exact halos → bit-identical to B2. Separately
gate layer 21's output into the PAN (denoising, ~free) to keep B2's >dense accuracy.

**Honest FLOP ceiling (halo overhead caps it).** Detect cv2 RF≈2 → gather overhead (cell+4)²/cell²:
- stride-16 routing (4px cells): 4× overhead → ρ=0.2 net 0.8× → save ~0.6G (12.4→11.8, ~5%).
- stride-32 routing (8px cells): 2.25× → ρ=0.2 net 0.45× → save ~1.75G (12.4→10.6, ~14%) — IF accuracy
  holds at the coarser grid (needs re-validation; B2 used stride-16). Clustered small objects reduce
  real overhead below worst-case.

The earlier "9.3 GFLOPs / +9%" projection was WRONG — it assumed 3.9G of sparsifiable conv at ρ scaling.
Realistic win: **5–14% model FLOPs while matching/beating dense P2 accuracy.** Positive, but modest.

## B3 plan (revised)
1. `SparseP2Detect`: gather selected cells (+2px halo) from the dense layer-21 feature, run cv2/cv3
   only there, scatter (masked elsewhere). Bit-exactness test vs dense-then-mask BEFORE trusting FLOPs.
2. (optional) re-validate B2 accuracy at stride-32 routing to unlock the larger saving.
3. Benchmark realized GFLOPs + mAP. Static top-k → TRT-safe shapes.

## (Superseded) Recommended staged build (Design A)

1. **Config + DENSE decoupled P2 head.** Train on VisDrone. GATE: must reach ~0.382 small-region mAP
   like the entangled TPH. If it doesn't, the entanglement was load-bearing -> reconsider.
2. **Add router-gated gather-conv sparse P2** to that head. Train end-to-end (router gates P2; density
   loss + detect loss). Measure realized GFLOPs + mAP vs the dense head (step 1) and the 89% ceiling.
3. Report: mAP retained @ realized FLOPs. Target ~9.3 GFLOPs (+9% vs base) for ~89% of the gain.

Each step is a real training run; step 1 is a hard gate before step 2.
