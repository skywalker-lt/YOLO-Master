#!/usr/bin/env python3
"""Score yolomaster_edge --save-txt dumps ('class conf x1 y1 x2 y2' in pixel xyxy, one file per image) with pycocotools."""
import argparse, glob, json, os, sys
COCO80_TO_91 = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
ap = argparse.ArgumentParser(); ap.add_argument("preds"); ap.add_argument("--ann", default="/root/coco/annotations/instances_val2017.json"); ap.add_argument("--out", default=None)
a = ap.parse_args()
res = []; n_img = 0
for f in sorted(glob.glob(os.path.join(a.preds, "*.txt"))):
    stem = os.path.splitext(os.path.basename(f))[0]
    try: img_id = int(stem)
    except ValueError: continue
    n_img += 1
    for line in open(f):
        p = line.split()
        if len(p) < 6: continue
        c, s, x1, y1, x2, y2 = int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
        res.append({"image_id": img_id, "category_id": COCO80_TO_91[c], "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)], "score": round(s, 5)})
out = a.out or (a.preds.rstrip("/") + "_coco.json")
json.dump(res, open(out, "w"))
print(f"images_with_dumps={n_img} detections={len(res)} -> {out}")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt = COCO(a.ann); dt = gt.loadRes(out) if res else None
if dt is None: print("no detections"); sys.exit(1)
ev = COCOeval(gt, dt, "bbox"); ev.params.imgIds = sorted(gt.getImgIds()); ev.evaluate(); ev.accumulate(); ev.summarize()
print(f"PYCOCO {os.path.basename(a.preds.rstrip('/'))} mAP50-95={ev.stats[0]:.4f} mAP50={ev.stats[1]:.4f} AP_S={ev.stats[3]:.4f} AP_M={ev.stats[4]:.4f} AP_L={ev.stats[5]:.4f}")
