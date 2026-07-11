"""Step 2, calibrated: base/dense/sparse mAP via ultralytics' OFFICIAL validator (DetMetrics, rect=True,
its exact NMS), so base/dense match the reported figure and the retained-% is trustworthy.

Injects P2 masking into DetectionValidator.postprocess (before NMS): base = all P2 anchors zeroed,
sparse = P2 kept only in the GT-density (oracle) top-rho cells. base uses the standalone no-P2 model.

  python -m littlemod.step2_val --weights EsMoE-N-TPH-640/VisDrone_EsMoE-N-TPH/weights/best.pt \
      --baseline-weights runs/baseline/EsMoE-N_VisDrone/weights/best.pt --grid 40 --rho 0.15
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import DEFAULT_CFG, LOGGER, SETTINGS
from ultralytics.utils.metrics import ap_per_class, box_iou

from littlemod.density import build_density_target
from littlemod.router import DensityRouter, MultiLevelDensityRouter


class MaskedValidator(DetectionValidator):
    """DetectionValidator that masks P2 (stride-4) anchors before NMS. Set attrs after construction:
    mask_mode in {dense, base, sparse, predicted}, routing_stride, rho, small_thresh, stride_p2.
    For `predicted`, also set router/router_multi/router_level and attach a Detect forward_pre_hook
    that stashes the neck features into self._neck (run() does this)."""

    mask_mode = "dense"
    routing_stride = 16
    rho = 0.2
    small_thresh = 64.0
    stride_p2 = 4
    router = None
    router_multi = False
    router_level = 2
    _neck = None
    gate_feature = False        # also zero the P2 FEATURE (its source layer) in unselected cells
    p2_src_layer = None         # index of the layer feeding the P2 detect (Detect.f[0]); set in run()
    causal_router = None        # trained CAUSAL router (gates the P2 feature from pre-P2 layers)
    causal_multi = False
    causal_level = 2
    _causal_feats = None        # {layer_idx: output} captured before the P2 layer
    _pred_keep = None           # per-image top-k cell indices, set by the gate hook, consumed by _apply_mask
    _pred_grid = None

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        self._batch = batch                      # stash GT for the oracle in postprocess
        return batch

    def postprocess(self, preds):
        if self.mask_mode != "dense":
            y = preds[0] if isinstance(preds, (list, tuple)) else preds
            y = self._apply_mask(y.clone())
            preds = (y, *preds[1:]) if isinstance(preds, (list, tuple)) else y
        return super().postprocess(preds)

    def _router_score(self):
        """Trained router on the captured neck features -> per-cell density logits [B,1,gh,gw].
        Picks P3(stride8)/P4(stride16) BY STRIDE so it works on the 4-scale P2 model, whose Detect
        input is [P2,P3,P4,P5] (index-based selection would grab P2)."""
        H = int(self._batch["img"].shape[-2])
        by_stride = {round(H / int(f.shape[-2])): f for f in self._neck}
        with torch.no_grad():
            if self.router_multi:
                return self.router(by_stride[8], by_stride[16])
            return self.router(by_stride[{0: 8, 1: 16, 2: 32}[self.router_level]])

    def _apply_mask(self, y):
        H, W = int(self._batch["img"].shape[-2]), int(self._batch["img"].shape[-1])
        ps = self.stride_p2
        ph, pw = H // ps, W // ps                       # P2 feature grid (rect)
        n_p2 = ph * pw
        if self.mask_mode == "base":
            y[:, 4:, :n_p2] = 0.0
            return y

        # If the feature-gating hook already chose cells this forward, reuse them EXACTLY so the P2
        # detect and the gated P2 feature agree on the same cells.
        if self._pred_keep is not None:
            gh, gw = self._pred_grid
            idx = torch.arange(n_p2, device=y.device)
            gy, gx = idx // pw, idx % pw
            p2cell = (gy * gh // ph).clamp(max=gh - 1) * gw + (gx * gw // pw).clamp(max=gw - 1)
            for b in range(y.shape[0]):
                y[b, 4:, :n_p2][:, ~torch.isin(p2cell, self._pred_keep[b])] = 0.0
            self._pred_keep = None                       # consume (avoid stale reuse next batch)
            return y

        # per-image top-k cells on a (gh, gw) routing grid; source = router (predicted) or GT density
        s = None
        if self.mask_mode == "predicted":
            s = self._router_score()                    # [B,1,gh,gw] on the P4-stride grid
            gh, gw = int(s.shape[-2]), int(s.shape[-1])
        else:                                            # oracle sparse
            rr = max(1, self.routing_stride // ps)
            gh, gw = ph // rr, pw // rr
        idx = torch.arange(n_p2, device=y.device)
        gy, gx = idx // pw, idx % pw
        p2cell = (gy * gh // ph).clamp(max=gh - 1) * gw + (gx * gw // pw).clamp(max=gw - 1)  # P2 anchor -> cell
        k = max(1, min(round(self.rho * gh * gw), gh * gw))
        for b in range(y.shape[0]):
            if s is not None:
                keepcell = torch.topk(s[b].reshape(-1), k).indices
            else:
                m = self._batch["batch_idx"] == b
                if not m.any():
                    y[b, 4:, :n_p2] = 0.0
                    continue
                d = build_density_target(self._batch["bboxes"][m], torch.zeros(int(m.sum()), device=y.device),
                                         1, (gh, gw), W, self.routing_stride, self.small_thresh)
                keepcell = torch.topk(d.reshape(-1), k).indices
            keep = torch.isin(p2cell, keepcell)
            y[b, 4:, :n_p2][:, ~keep] = 0.0
        return y

    def _gate_p2_feature(self, module, inp, out):
        """forward_hook on the P2-source layer (Detect.f[0]): zero its output in cells NOT selected, so
        the sparsity propagates into the PAN (its stride-2 consumer -> P3/P4/P5), not only the P2 detect.
        Selection = the trained CAUSAL router (run here on features captured before this layer) if set,
        else the GT-density ORACLE. The per-image top-k cells are stashed so postprocess masks the P2
        detect with the SAME cells (feature-gating and detect-masking stay consistent)."""
        if getattr(self, "_batch", None) is None:
            return out                                   # warmup forward (no batch yet) — don't gate
        B, C, ph, pw = out.shape
        ps = self.stride_p2
        W = int(self._batch["img"].shape[-1])
        s = None
        if self.causal_router is not None:               # predicted routing from pre-P2 features
            fs = sorted(self._causal_feats.values(), key=lambda f: -f.shape[-2])   # finer, coarser
            with torch.no_grad():
                s = self.causal_router(fs[0], fs[1]) if self.causal_multi else self.causal_router(fs[0])
            gh, gw = int(s.shape[-2]), int(s.shape[-1])
        else:                                            # oracle (GT density)
            rr = max(1, self.routing_stride // ps)
            gh, gw = ph // rr, pw // rr
        idx = torch.arange(ph * pw, device=out.device)
        gy, gx = idx // pw, idx % pw
        cell2d = ((gy * gh // ph).clamp(max=gh - 1) * gw + (gx * gw // pw).clamp(max=gw - 1)).view(ph, pw)
        k = max(1, min(round(self.rho * gh * gw), gh * gw))
        out = out.clone()
        keeps = []
        for b in range(B):
            if s is not None:
                keepcell = torch.topk(s[b].reshape(-1), k).indices
            else:
                m = self._batch["batch_idx"] == b
                if not m.any():
                    out[b] = 0.0
                    keeps.append(torch.empty(0, dtype=torch.long, device=out.device))
                    continue
                d = build_density_target(self._batch["bboxes"][m], torch.zeros(int(m.sum()), device=out.device),
                                         1, (gh, gw), W, self.routing_stride, self.small_thresh)
                keepcell = torch.topk(d.reshape(-1), k).indices
            out[b, :, ~torch.isin(cell2d, keepcell)] = 0.0
            keeps.append(keepcell)
        self._pred_keep, self._pred_grid = keeps, (gh, gw)   # consumed by _apply_mask (same cells)
        return out

    # --- calibrated AP on SMALL GTs (<small_thresh px, letterboxed frame) -----------------------
    # Reuses the exact matcher the overall metric uses (box_iou + match_predictions + ap_per_class),
    # restricted to small targets: (dense - base) here is P2's small-object gain — where it lives.
    # NB: targets restricted, predictions kept (no COCO ignore-on-large), so absolute AP_small runs a
    # touch below pycocotools; base/dense/sparse are evaluated identically, so the DELTAS are exact.
    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        if not hasattr(self, "_small"):
            self._small = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            gtb, gtc = pbatch["bboxes"], pbatch["cls"]
            if gtb.shape[0]:
                side = torch.maximum(gtb[:, 2] - gtb[:, 0], gtb[:, 3] - gtb[:, 1])
                sm = side < self.small_thresh
                gtb, gtc = gtb[sm], gtc[sm]
            npd = predn["cls"].shape[0]
            if gtb.shape[0] and npd:
                tp = self.match_predictions(predn["cls"], gtc, box_iou(gtb, predn["bboxes"])).cpu().numpy()
            else:
                tp = np.zeros((npd, self.niou), dtype=bool)
            self._small["tp"].append(tp)
            self._small["conf"].append(np.zeros(0) if not npd else predn["conf"].cpu().numpy())
            self._small["pred_cls"].append(np.zeros(0) if not npd else predn["cls"].cpu().numpy())
            self._small["target_cls"].append(gtc.cpu().numpy())

    def get_stats(self):
        stats = super().get_stats()
        S = getattr(self, "_small", None)
        ap50 = ap = 0.0
        if S and sum(len(t) for t in S["target_cls"]):
            tp = np.concatenate(S["tp"]) if S["tp"] else np.zeros((0, self.niou), bool)
            r = ap_per_class(tp, np.concatenate(S["conf"]), np.concatenate(S["pred_cls"]),
                             np.concatenate(S["target_cls"]), plot=False)
            a = np.asarray(r[5])                                   # [n_cls, n_iou]; index 5 = ap
            ap50, ap = float(a[:, 0].mean()), float(a.mean())
        stats["metrics/mAP50(S)"], stats["metrics/mAP50-95(S)"] = ap50, ap
        return stats


def _fit_state_dict(module, sd):
    """Load a router state_dict tolerant of a param-less Dropout2d that a later router.py inserted
    before the final conv (shifts the output Conv2d's index, e.g. net.9 -> net.10). Remap any leftover
    key to the same-shape target key; assert every target key is filled so the OUTPUT conv is never
    left randomly initialized (a silent strict=False would do exactly that)."""
    sd, tgt = dict(sd), module.state_dict()
    missing = [k for k in tgt if k not in sd]
    extra = [k for k in sd if k not in tgt]
    for mk in missing:
        for ek in list(extra):
            if sd[ek].shape == tgt[mk].shape:
                sd[mk] = sd.pop(ek)
                extra.remove(ek)
                break
    still = [k for k in tgt if k not in sd]
    assert not still, f"router load: could not map keys {still} (extra={extra})"
    return sd


def load_router(ckpt_path, device):
    """Rebuild the trained DensityRouter/MultiLevelDensityRouter from a step-1 checkpoint. Input
    channels aren't stored, so infer them from the state_dict conv shapes."""
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["router"]
    c, layers = int(ck.get("router_c", 64)), int(ck.get("router_layers", 3))
    multi, level = bool(ck.get("multi_level", False)), int(ck.get("level", 2))
    if multi:
        c_p3_out = sd["down.0.weight"].shape[0]
        router = MultiLevelDensityRouter(sd["down.0.weight"].shape[1], sd["net.0.weight"].shape[1] - c_p3_out,
                                         c=c, layers=layers, c_p3_out=c_p3_out)
    else:
        router = DensityRouter(sd["net.0.weight"].shape[1], c=c, layers=layers)
    router.load_state_dict(_fit_state_dict(router, sd))
    return router.eval().to(device), multi, level, ck.get("recall@0.2"), ck.get("feat_layers")


def run(weights, mode, routing_stride, rho, small_thresh, data, imgsz, batch, workers, router_ckpt=None,
        gate_feature=False, causal_router_ckpt=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ym = YOLO(weights)
    try:
        from ultralytics.nn.modules.moe.modules import ES_MOE
        for mod in ym.model.modules():
            if isinstance(mod, ES_MOE):
                mod.use_sparse_inference = False   # dense MoE eval (matches --no-sparse-eval)
    except Exception:
        pass
    args = get_cfg(DEFAULT_CFG)
    args.data, args.imgsz, args.batch, args.workers = data, imgsz, batch, workers
    args.conf, args.iou, args.rect, args.split = 0.001, 0.7, True, "val"
    args.plots, args.verbose, args.save_json = False, False, False
    v = MaskedValidator(args=args)
    v.mask_mode, v.routing_stride, v.rho, v.small_thresh = mode, routing_stride, rho, small_thresh
    v.stride_p2 = int(ym.model.model[-1].stride[0])
    if mode == "predicted":
        v.router, v.router_multi, v.router_level, r02, _ = load_router(router_ckpt, dev)
        # capture the Detect head's input ([P2,]P3,P4,P5) each forward for the router
        ym.model.model[-1].register_forward_pre_hook(lambda mod, a: setattr(v, "_neck", a[0]))
    if causal_router_ckpt is not None:
        # B2: predicted CAUSAL feature-gating — router runs mid-forward on pre-P2 features and gates
        # both the P2 feature (layer 21) and the P2 detect. Implies gate_feature.
        v.causal_router, v.causal_multi, v.causal_level, r02, feat_layers = load_router(causal_router_ckpt, dev)
        v._causal_feats = {}
        for i in feat_layers:                            # capture pre-P2 features (computed before layer 21)
            ym.model.model[i].register_forward_hook(lambda mod, inp, out, i=i: v._causal_feats.__setitem__(i, out))
        gate_feature = True
        LOGGER.info(f"[step2-cal] CAUSAL predicted feature-gating: router on layers {feat_layers} (R@0.2={r02:.3f})")
    if gate_feature:
        v.gate_feature = True
        p2_src = int(ym.model.model[-1].f[0])            # layer feeding the stride-4 (P2) detect
        v.p2_src_layer = p2_src
        ym.model.model[p2_src].register_forward_hook(v._gate_p2_feature)
        LOGGER.info(f"[step2-cal] P2-feature gating ON at layer {p2_src}")
    m = v(model=ym.model)
    return (float(m["metrics/mAP50(B)"]), float(m["metrics/mAP50-95(B)"]),
            float(m["metrics/mAP50(S)"]), float(m["metrics/mAP50-95(S)"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="4-scale P2 model")
    ap.add_argument("--baseline-weights",
                    default="scripts/reproduce/results/result-esmoen-visdrone/weights/best.pt",
                    help="figure-exact EsMoE-N (0.3504); NOT runs/baseline/EsMoE-N (=sparse variant, 0.325)")
    ap.add_argument("--data", default="VisDrone.yaml")
    ap.add_argument("--datasets-dir", default="/data/datasets")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--routing-stride", type=int, default=16, help="16 (~40x40 grid) or 32 (~20x20)")
    ap.add_argument("--rho", type=float, nargs="+", default=[0.15], help="one or more capacity fractions to sweep")
    ap.add_argument("--router", default=None, help="step-1 router ckpt; adds a PREDICTED-routing sweep")
    ap.add_argument("--gate-feature", action="store_true",
                    help="entanglement probe: also zero the P2 FEATURE (Design B, PAN starved) and report vs detect-only")
    ap.add_argument("--causal-router", default=None,
                    help="B2: causal router ckpt (feat_layers set) — predicted feature-gating vs dense")
    ap.add_argument("--small-thresh", type=float, default=64.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    SETTINGS["datasets_dir"] = a.datasets_dir

    LOGGER.info("[step2-cal] official-validator eval (DetMetrics, rect=True)")
    # base + dense are rho-independent -> compute once, sweep sparse only.
    b50, b, bs50, bs = run(a.baseline_weights, "dense", a.routing_stride, a.rho[0], a.small_thresh, a.data, a.imgsz, a.batch, a.workers)
    d50, d, ds50, ds = run(a.weights, "dense", a.routing_stride, a.rho[0], a.small_thresh, a.data, a.imgsz, a.batch, a.workers)
    st = int(a.small_thresh)
    LOGGER.info(f"  baseline (no P2): full mAP50={b50:.4f} mAP50-95={b:.4f} | small<{st}px mAP50={bs50:.4f} mAP50-95={bs:.4f}")
    LOGGER.info(f"  dense P2        : full mAP50={d50:.4f} mAP50-95={d:.4f} | small<{st}px mAP50={ds50:.4f} mAP50-95={ds:.4f}")
    gf, gs5, gs = d50 - b50, ds50 - bs50, ds - bs
    LOGGER.info(f"  dense P2 gain over baseline: full mAP50={gf:+.4f} | small mAP50={gs5:+.4f} | small mAP50-95={gs:+.4f}")

    def ret(sp, ba, ga):
        return f"{(sp - ba) / ga * 100:.0f}%" if ga > 0 else "n/a"

    # oracle (GT-density) routing, and — if a router ckpt is given — predicted routing.
    sweeps = [("oracle  ", "sparse", None)]
    if a.router:
        sweeps.append(("router  ", "predicted", a.router))
    for tag, mode, rck in sweeps:
        LOGGER.info(f"  --- {tag.strip()} routing ({mode}) ---")
        for rho in a.rho:
            s50, s, ss50, ss = run(a.weights, mode, a.routing_stride, rho, a.small_thresh, a.data, a.imgsz, a.batch, a.workers, rck)
            LOGGER.info(f"  {tag} rho={rho:<4}: full mAP50={s50:.4f} (retains {ret(s50, b50, gf)}) | "
                        f"small mAP50={ss50:.4f} (retains {ret(ss50, bs50, gs5)}) | "
                        f"small mAP50-95={ss:.4f} (retains {ret(ss, bs, gs)})")

    if a.gate_feature:
        LOGGER.info("  --- ENTANGLEMENT PROBE: detect-only (Design A) vs P2-FEATURE gated (Design B, PAN starved), oracle ---")
        for rho in a.rho:
            _, _, aA, aA95 = run(a.weights, "sparse", a.routing_stride, rho, a.small_thresh, a.data, a.imgsz, a.batch, a.workers)
            _, _, bB, bB95 = run(a.weights, "sparse", a.routing_stride, rho, a.small_thresh, a.data, a.imgsz, a.batch, a.workers, gate_feature=True)
            LOGGER.info(f"  rho={rho:<4}: A detect-only small mAP50={aA:.4f} (ret {ret(aA, bs50, gs5)}) | "
                        f"B feature-gated small mAP50={bB:.4f} (ret {ret(bB, bs50, gs5)}) | "
                        f"PAN cost (A-B)={aA - bB:+.4f} on small50, {aA95 - bB95:+.4f} on small50-95")

    if a.causal_router:
        LOGGER.info("  --- B2: CAUSAL predicted feature-gating (router gates P2 feature+detect) vs dense ---")
        LOGGER.info(f"        dense P2 small mAP50={ds50:.4f} mAP50-95={ds:.4f} (target to match/beat)")
        for rho in a.rho:
            _, _, c50, c95 = run(a.weights, "sparse", a.routing_stride, rho, a.small_thresh, a.data, a.imgsz,
                                 a.batch, a.workers, causal_router_ckpt=a.causal_router)
            LOGGER.info(f"  rho={rho:<4}: causal-gated small mAP50={c50:.4f} (retains {ret(c50, bs50, gs5)}, "
                        f"{c50 - ds50:+.4f} vs dense) | small mAP50-95={c95:.4f} (retains {ret(c95, bs, gs)}, "
                        f"{c95 - ds:+.4f} vs dense)")


if __name__ == "__main__":
    main()
