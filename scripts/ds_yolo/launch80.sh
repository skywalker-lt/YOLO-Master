#!/usr/bin/env bash
# Launch (or resume) the three 80-epoch arms on the H200. Safe to re-run after a crash or reboot: each run resumes.
# Each process is capped at 31% of the GPU (three runs share one 140 GB H200; uncapped, top180 died of OOM at epoch 2).
source /root/anaconda3/etc/profile.d/conda.sh; conda activate yolo_master
cd /data/YOLO-Master; export PYTHONPATH=/data/YOLO-Master; mkdir -p /data/runs80 /training_data/runs80 /training_data/logs
for spec in "dense80 yolo26m.yaml --pretrained /data/weights/yolo26m-objv1-150.pt" "top180 /data/weights/yolo26m-objv1-wb-top1-fixed.pt" "pair680 /data/weights/yolo26m-objv1-wb-pair6-fixed.pt"; do
  set -- $spec; n=$1; shift
  if pgrep -f "train_80ep.py.*--name $n\b" > /dev/null; then echo "$n already running"; continue; fi
  setsid nohup python scripts/ds_yolo/train_80ep.py --model "$@" --name $n --mem-fraction 0.31 >> /data/runs80/$n.out 2>&1 < /dev/null & disown
  echo "launched $n"; sleep 20
done

pgrep -f "scripts/ds_yolo/monitor80.sh" > /dev/null || { setsid nohup bash scripts/ds_yolo/monitor80.sh > /dev/null 2>&1 < /dev/null & disown; echo "monitor started: tail -F /data/runs80/train.log"; }
