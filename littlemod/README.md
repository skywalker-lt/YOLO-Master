# littlemod — GT-Density-Supervised Sparse P2 Routing

Research code for the plan in `/tmp/yolo-grandmaster-plan.md`: a tiny router, supervised by a
GT-derived density map, selects (expert-choice, fixed-capacity top-k) the few regions where the
expensive P2 (stride-4) branch is computed — turning P2's ~+45% FLOPs into ~+20% while keeping the
small-object gain. One router, three consumers (P2 gating, decode pre-filter, P3 modulation).

## ONE RULE — reuse, don't rebuild
| Need | Source (reused) |
|---|---|
| Router conv arch | `ultralytics.nn.modules.conv.Conv`, `DWConv` |
| Top-k routing pattern | `moe/routers.py::BaseRouter._process_logits`; `torch.topk` |
| **QFL** (`L_dens`) | `ultralytics.utils.loss.VarifocalLoss` (quality-focal, soft target) |
| GT cell coords / decode | `utils/tal.py::make_anchors`, `dist2bbox`, `TaskAlignedAssigner` |
| Sparse P2 head | `Detect` (`head.py`) + `Conv`/`C3k2` (`block.py`) |

Built new (grep-confirmed absent): GT Gaussian **density target**, **Dice** loss, **Recall@ρ** metric.
The repo's spatial routers pool to image level (the image-wise routing this project replaces), so we
reuse their primitives but keep the density map spatial.

## Modules (validated in isolation)
- `density.py` — `build_density_target` (Gaussian splat, small-object / size-weighted, `max` not `sum`)
  + `router_recall_at_rho`. Oracle (S=D) recall = 1.00 @ρ=10/20/30%; big objects excluded.
- `router.py` — `DensityRouter` (Conv→DWConv→Conv→sigmoid, 17k params) + static `select_topk` over cells.
- `loss.py` — `DensityLoss` (VarifocalLoss + Dice) + `anneal_beta`/`anneal_mix` (cold-start, β:1→0 over 30%).

## Validation order (plan §6 — falsify cheapest first)
1. **[modules ready]** Router + `L_dens` on a **frozen** detector → **Router Recall@ρ curve**. ≥~97%@20% ⇒ go.
2. Sparse P2 branch with **GT/oracle** routing → method upper bound.
3. Predicted routing + annealing → gap to oracle.
4. **Mechanism ablation** (density vs Gumbel/CEASC vs P3-confidence/QueryDet) — *decides the contribution*.
5. Consumers ②③, k-sweep, real Jetson/RK latency.

> Do not write the Introduction before step 4 (plan): if Gumbel matches density supervision in-framework,
> the core contribution collapses.

## Next
Step-1 training harness: attach `DensityRouter` to a frozen YOLO-Master-N neck P5 feature (stride 32,
20×20), supervise with `DensityLoss` on VisDrone, log Recall@ρ over epochs. (Needs VisDrone re-staged —
the dataset symlinks were wiped.) VisDrone / 300 ep / N-scale ⇒ <$12 a run.
