"""Pre-build the Ultralytics disk cache (.npy next to each image) for COCO train2017 and val2017 on the pod's local disk."""

import time

from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

if __name__ == "__main__":
    yaml = "/data/datasets/coco/coco.yaml"
    data = check_det_dataset(yaml)
    cfg = get_cfg(overrides={"task": "detect", "data": yaml, "imgsz": 640, "cache": "disk", "workers": 32})
    for split, mode in (("train", "train"), ("val", "val")):
        t0 = time.time()
        ds = build_yolo_dataset(cfg, data[split], batch=64, data=data, mode=mode, stride=32)
        print(f"CACHE {split}: {len(ds)} images, cache={ds.cache}, {time.time() - t0:.0f} s", flush=True)
