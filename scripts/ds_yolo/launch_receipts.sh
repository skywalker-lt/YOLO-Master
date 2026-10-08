#!/usr/bin/env bash
# Seminar-4 receipts on the second H200 (one GPU, two concurrent runs; safe to re-run: each run resumes).
#   splice10: dense splice H = public YOLO26-M stem (frozen, layers 0-5) + public YOLO26-L tail, 10 epochs, stage-2 recipe.
#             Gate G1: forced-L val (CLI protocol) >= 0.5337.
#   twoscale20: public YOLO26-M fine-tuned 20 epochs with every batch at 512 or 768. Gate G2: 512/768 dumps mixed by the
#             thumbnail count router at 768 share 0.43 >= 0.5291.
source /root/anaconda3/etc/profile.d/conda.sh; conda activate yolo_master
cd /data/YOLO-Master; export PYTHONPATH=/data/YOLO-Master; P=runs_receipts; mkdir -p /data/$P /training_data/$P /training_data/logs
[ -f /data/weights/yolo26l-mstem-init.pt ] || python scripts/ds_yolo/splice_build.py /data/weights/yolo26m.pt /data/weights/yolo26l.pt /data/weights/yolo26l-mstem-init.pt
launch() { n=$1; shift
  if pgrep -f "^python scripts/ds_yolo/train_80ep.py .*--name $n\b" > /dev/null; then echo "$n already running"; return; fi
  setsid nohup python scripts/ds_yolo/train_80ep.py --project $P --name $n --mem-fraction 0.48 "$@" >> /data/$P/$n.out 2>&1 < /dev/null & disown
  echo "launched $n"; sleep 20; }
launch splice10 --model /data/weights/yolo26l-mstem-init.pt --epochs 10 --freeze-stem 6 --extra close_mosaic=3
launch twoscale20 --model /data/weights/yolo26m.pt --epochs 20 --scales 512,768 --extra close_mosaic=5
pgrep -f "^bash scripts/ds_yolo/monitor80.sh" > /dev/null || { PROJECT=$P RUNS="splice10 twoscale20" setsid nohup bash scripts/ds_yolo/monitor80.sh > /dev/null 2>&1 < /dev/null & disown; echo "monitor started: tail -F /data/$P/train.log"; }
