#!/usr/bin/env python3
"""Evaluate a trained detector on AI-TOD(-v2) with the official AI-TOD COCO metrics.

Reports the exact size-binned metrics the AI-TOD benchmark and YOLOv12-X report, so
results are directly comparable:

    GFLOPs  params  AP  AP50  AP75  AP_vt  AP_t  AP_s  AP_m

Size bins follow the AI-TOD evaluator (aitodpycocotools) — object scale = sqrt(area):
    very tiny (vt) [0,   8px]   -> area [0,      64]
    tiny      (t)  [8,  16px]   -> area [64,    256]
    small     (s)  [16, 32px]   -> area [256,  1024]
    medium    (m)  [32, inf]    -> area [1024,  inf]
AP/AP50/AP75 are over 'all'. maxDets=1500 (AI-TOD dense-scene convention, vs COCO's 100).

Requires: ultralytics + pycocotools. Install `faster-coco-eval` too — it's auto-detected
and runs the C++ COCOeval (~50x faster, identical numbers), which matters a lot at conf=0.001
where there are ~1M detections. `pip install faster-coco-eval pycocotools`.

Usage:
    python scripts/reproduce/eval_aitod.py --weights runs/.../best.pt --split test
    python scripts/reproduce/eval_aitod.py --weights best.pt --split val --imgsz 800 --out preds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# AI-TOD area ranges (px^2) + labels, matching aitodpycocotools. maxDets=1500.
AITOD_AREARNG = [[0, 1e10], [0, 8 ** 2], [8 ** 2, 16 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 1e10]]
AITOD_LBL = ["all", "vt", "t", "s", "m"]


def _ap(coco_eval, area_idx: int, iou_thr: float | None = None, max_det_idx: int = -1) -> float:
    """Mean precision from cocoEval.eval['precision'] [T,R,K,A,M] for one area range."""
    p = coco_eval.eval["precision"]
    if iou_thr is not None:
        t = np.where(np.isclose(coco_eval.params.iouThrs, iou_thr))[0]
        p = p[t]
    p = p[:, :, :, area_idx, max_det_idx]
    p = p[p > -1]
    return float(np.mean(p)) if p.size else float("nan")


def _measure_gflops(model, imgsz: int, img_files=None, n: int = 16, seed: int = 0) -> float:
    """Deterministic, sparse-MoE-honest GFLOPs at FULL resolution.

    ultralytics' get_flops profiles a `torch.empty` (uninitialised memory) 32x32
    input and scales by (imgsz/32)^2. For input-dependent MoE (top-k routing +
    conditional compute) that is both NON-reproducible (garbage values route to a
    different mix of experts each call -> FLOPs swing by whole GF once x625'd) and
    inaccurate (routing at 32x32 doesn't represent 800x800). Here we profile at full
    resolution on real images (mean over n; reproducible since the images are fixed
    and the model is in eval) -> the honest expected sparse cost. Falls back to a
    fixed-seed input when images are unavailable.
    """
    import torch
    try:
        import thop
    except ImportError:
        return float("nan")
    import contextlib
    import io
    from copy import deepcopy

    m = deepcopy(model).eval()
    dev = next(m.parameters()).device
    x = None
    if img_files:
        import cv2
        xs = []
        for p in img_files[:n]:
            im = cv2.imread(str(p))
            if im is None:
                continue
            im = cv2.resize(im, (imgsz, imgsz))[:, :, ::-1]  # BGR->RGB
            xs.append(torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0)
        if xs:
            x = torch.stack(xs).to(dev)
    if x is None:  # fallback: fixed-seed deterministic input
        g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(1, 3, imgsz, imgsz, device=dev, generator=g)
    with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
        flops = thop.profile(m, inputs=[x], verbose=False)[0] / 1e9 * 2
    return flops / x.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="AI-TOD COCO evaluation (size-binned APs).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="Trained .pt checkpoint.")
    ap.add_argument("--data", default="ultralytics/cfg/datasets/AI-TOD-v2.yaml")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--gt-json", default=None, help="COCO GT json (default: <path>/aitodv2_anns/aitodv2_<split>.json).")
    ap.add_argument("--imgsz", type=int, default=800, help="Inference size (AI-TOD native = 800).")
    ap.add_argument("--conf", type=float, default=0.001, help="Low conf keeps the full PR curve (standard for AP).")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU.")
    ap.add_argument("--max-det", type=int, default=1500, help="Max detections/image at inference (AI-TOD dense).")
    ap.add_argument("--max-dets-eval", type=int, default=1500, help="COCOeval maxDets (AI-TOD convention).")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--half", action="store_true", help="FP16 inference.")
    ap.add_argument("--flops-imgsz", type=int, default=None, help="Imgsz for the GFLOPs figure (default = --imgsz).")
    ap.add_argument("--flops-imgs", type=int, default=8, help="Real images to average GFLOPs over (sparse-MoE FLOPs are input-dependent; deterministic).")
    ap.add_argument("--out", default=None, help="Also dump the COCO predictions json here.")
    a = ap.parse_args()

    # raise the open-file-descriptor soft limit (dense image dirs + dataloader workers)
    try:
        import resource
        _s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(_h, 65536), _h))
    except Exception:  # noqa: BLE001
        pass

    import yaml
    from ultralytics import YOLO
    from ultralytics.utils.checks import check_file
    # C++ COCOeval (~50x faster, identical numbers) if available, else pure-Python pycocotools.
    try:
        from faster_coco_eval import COCO, COCOeval_faster as COCOeval
        eval_backend = "faster-coco-eval (C++)"
    except ImportError:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        eval_backend = "pycocotools (pure-Python; `pip install faster-coco-eval` for ~50x speedup)"

    # ---- resolve dataset paths ----
    data_yaml = check_file(a.data)
    d = yaml.safe_load(open(data_yaml))
    root = Path(d["path"])
    img_dir = root / d[a.split]
    gt_json = Path(a.gt_json) if a.gt_json else root / "aitodv2_anns" / f"aitodv2_{a.split}.json"
    if not gt_json.exists():
        raise FileNotFoundError(f"GT json not found: {gt_json}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"image dir not found: {img_dir}")

    # ---- GT + id maps ----
    coco_gt = COCO(str(gt_json))
    stem2id = {Path(im["file_name"]).stem: im["id"] for im in coco_gt.dataset["images"]}
    cats = sorted(coco_gt.dataset["categories"], key=lambda c: c["id"])
    yolo2cat = {i: cats[i]["id"] for i in range(len(cats))}  # yolo class idx -> coco cat id (identity for AI-TOD)
    print(f"[eval] GT {gt_json.name}: {len(coco_gt.dataset['images'])} imgs, {len(cats)} classes, "
          f"{len(coco_gt.dataset['annotations'])} anns")

    # ---- model ----
    model = YOLO(a.weights)
    n_params = sum(p.numel() for p in model.model.parameters())
    flops_sz = a.flops_imgsz or a.imgsz

    # ---- inference -> COCO detections ----
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

    # deterministic, sparse-MoE-honest GFLOPs (mean over real images @ full res)
    gflops = _measure_gflops(model.model, flops_sz, img_files, n=a.flops_imgs)
    print(f"[eval] GFLOPs={gflops:.2f} @ {flops_sz}px "
          f"(mean over {min(a.flops_imgs, len(img_files))} real imgs, deterministic full-res)")
    print(f"[eval] running inference on {len(img_files)} images @ imgsz={a.imgsz} conf={a.conf} "
          f"iou={a.iou} max_det={a.max_det} ...")
    dets, missing = [], 0
    # Pass the directory (streamed one file at a time), NOT a list of paths — a big path list
    # makes ultralytics' autocast_list open every file at once -> "Too many open files".
    for r in model.predict(source=str(img_dir), imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                           max_det=a.max_det, device=a.device, half=a.half, batch=a.batch,
                           stream=True, verbose=False, save=False):
        img_id = stem2id.get(Path(r.path).stem)
        if img_id is None:
            missing += 1
            continue
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
            dets.append({"image_id": int(img_id),
                         "category_id": int(yolo2cat.get(int(c), int(c))),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "score": float(s)})
    if missing:
        print(f"[eval] WARNING: {missing} predicted images had no matching GT image_id (stem mismatch).")
    print(f"[eval] {len(dets)} detections over {len(img_files)} images")
    if a.out:
        json.dump(dets, open(a.out, "w"))
        print(f"[eval] predictions -> {a.out}")
    if not dets:
        raise RuntimeError("No detections produced; cannot evaluate.")

    # ---- COCOeval with AI-TOD area ranges + maxDets ----
    print(f"[eval] scoring backend: {eval_backend}")
    coco_dt = coco_gt.loadRes(dets)
    ce = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ce.params.maxDets = [a.max_dets_eval]
    ce.params.areaRng = AITOD_AREARNG
    ce.params.areaRngLbl = AITOD_LBL
    ce.evaluate()
    ce.accumulate()

    m = {
        "AP":    _ap(ce, 0),
        "AP50":  _ap(ce, 0, iou_thr=0.5),
        "AP75":  _ap(ce, 0, iou_thr=0.75),
        "AP_vt": _ap(ce, 1),
        "AP_t":  _ap(ce, 2),
        "AP_s":  _ap(ce, 3),
        "AP_m":  _ap(ce, 4),
    }
    print("\n" + "=" * 72)
    print(f"AI-TOD eval  |  weights={Path(a.weights).name}  split={a.split}  maxDets={a.max_dets_eval}")
    print(f"  params={n_params/1e6:.3f}M   GFLOPs={gflops:.1f} @ {flops_sz}px")
    print("  " + "  ".join(f"{k}={v*100:5.1f}" for k, v in m.items()) + "   (x100, %)")
    print("=" * 72)
    # machine-readable line
    print("RESULT_JSON " + json.dumps({"params_M": round(n_params/1e6, 3), "gflops": round(gflops, 2),
                                       "flops_imgsz": flops_sz, **{k: round(v*100, 2) for k, v in m.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
