"""Score the three 80-epoch arms like for like: PyTorch validation on full val2017 (multi-label, conf 0.001, IoU 0.7,
max_det 300) with pycocotools on the saved predictions, for last.pt and best.pt of each run. Writes a summary table.
usage: python final_eval80.py [--project runs80] [--runs dense80,top180,pair680] [--device 0]"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="runs80")
    ap.add_argument("--runs", default="dense80,top180,pair680")
    ap.add_argument("--device", default="0")
    ap.add_argument("--data", default="/data/datasets/coco/coco.yaml")
    a = ap.parse_args()
    out_dir = Path("/data") / a.project / "final_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in a.runs.split(","):
        for w in ("last", "best"):
            pt = Path("/data") / a.project / run / "weights" / f"{w}.pt"
            if not pt.exists():
                print("missing", pt, flush=True)
                continue
            name = f"{run}_{w}"
            m = YOLO(str(pt))
            r = m.val(
                data=a.data, imgsz=640, batch=32, conf=0.001, iou=0.7, max_det=300, device=a.device, save_json=True,
                plots=False, project=str(out_dir), name=name, exist_ok=True, verbose=False,
            )
            row = {"run": name, "map50_95": float(r.box.map), "map50": float(r.box.map50), "pycoco": None}
            pred = out_dir / name / "predictions.json"
            if pred.exists():  # the validator prints the pycocotools numbers; recompute them here so they are on file
                try:
                    from pycocotools.coco import COCO
                    from pycocotools.cocoeval import COCOeval

                    gt = COCO("/data/datasets/coco/annotations/instances_val2017.json")
                    ev = COCOeval(gt, gt.loadRes(str(pred)), "bbox")
                    ev.params.imgIds = sorted(gt.getImgIds())
                    ev.evaluate()
                    ev.accumulate()
                    ev.summarize()
                    row["pycoco"] = float(ev.stats[0])
                    row["pycoco_s_m_l"] = [float(ev.stats[3]), float(ev.stats[4]), float(ev.stats[5])]
                except Exception as exc:  # noqa: BLE001
                    row["pycoco_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print("ROW", json.dumps(row), flush=True)
            (out_dir / "summary.json").write_text(json.dumps(rows, indent=1))
    print("FINAL EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
