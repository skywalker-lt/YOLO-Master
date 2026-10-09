#!/usr/bin/env bash
# Wait for the three 80-epoch trainers to exit, then score all arms (final_eval80.py); log to /data/runs80/final_eval.log.
while pgrep -f "^python scripts/ds_yolo/train_80ep.py" > /dev/null; do sleep 120; done
source /root/anaconda3/etc/profile.d/conda.sh; conda activate yolo_master
cd /data/YOLO-Master; export PYTHONPATH=/data/YOLO-Master
python scripts/ds_yolo/final_eval80.py > /data/runs80/final_eval.log 2>&1
mkdir -p /training_data/runs80/final_eval && cp /data/runs80/final_eval/summary.json /data/runs80/final_eval.log /training_data/runs80/final_eval/
