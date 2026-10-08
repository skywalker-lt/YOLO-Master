# DS-YOLO 80-epoch test (H200 launcher)

Three arms upcycled from `yolo26m-objv1-150.pt` with YOLO26-M's own stage-2 recipe as this fork expresses it
(`recipe_yolo26m_stage2.yaml`, read from the public `yolo26m.pt` train_args; batch 64 with the weight decay doubled so the
effective decay equals the official batch-128 run's): the dense YOLO26-M control, a conditional top-1 weight bank (four
branches) and a hard top-2 weight bank (six pair branches), both with a fixed balanced k-means router.

Pod layout: code in `/data/YOLO-Master` (this branch), weights in `/data/weights`, dataset in `/data/datasets/coco` with
`cache=disk` on the pod's local disk, runs in `/data/runs80`; a backup of `weights/last.pt`, `weights/best.pt`, `results.csv`
and `args.yaml` is written to `/training_data/runs80/<run>/` every 10 epochs and at the end.

```bash
python scripts/ds_yolo/build_cache.py                                     # disk cache for train2017 and val2017
python scripts/ds_yolo/upcycle_fixed.py                                   # the two routed checkpoints into /data/weights
bash scripts/ds_yolo/launch80.sh                                          # launches or resumes the three arms
python scripts/ds_yolo/train_80ep.py --model yolo26m.yaml --name dense80 --pretrained /data/weights/yolo26m-objv1-150.pt
```

`train_80ep.py` resumes from the local `last.pt`, else restores the run from `/training_data/runs80/<run>/` and resumes,
else starts fresh. Re-running `launch80.sh` after a crash or reboot is safe.
