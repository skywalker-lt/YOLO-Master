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

Requires: pycocotools (`pip install pycocotools`), ultralytics.

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
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

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

    # ---- model + GFLOPs ----
    model = YOLO(a.weights)
    n_params = sum(p.numel() for p in model.model.parameters())
    flops_sz = a.flops_imgsz or a.imgsz
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = get_flops(model.model, imgsz=flops_sz)
    except Exception as e:  # noqa: BLE001
        gflops = float("nan")
        print(f"[eval] GFLOPs unavailable: {e}")

    # ---- inference -> COCO detections ----
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
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
