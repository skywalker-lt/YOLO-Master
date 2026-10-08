#!/usr/bin/env bash
# Seminar-4 receipt evaluation (pod 2). The seminar's protocol: unchanged yolo-master-edge 1.2.0 trt10-cuda12 bundle,
# TensorRT fp16, batch 1, conf 0.001, IoU 0.7, multi-label, full val2017 dumps scored with pycocotools. AP is
# device-independent (the T4 and L4 engines agree to 0.0004), so the H200 engines stand in for the T4 dumps.
#   bash receipt_eval.sh [last|best]   -> /data/runs_receipts/eval/eval.log, COCO-format dumps copied to
#   /training_data/runs_receipts/eval/ for the gate-G2 mixing (scripts/ds_yolo/receipt_mix.py, CPU, anywhere).
source /root/anaconda3/etc/profile.d/conda.sh; conda activate yolo_master
cd /data/YOLO-Master; export PYTHONPATH=/data/YOLO-Master
BIN=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0/yolomaster_edge
IMG=/data/datasets/coco/images/val2017; ANN=/data/datasets/coco/annotations/instances_val2017.json
W=${1:-last}; E=/data/runs_receipts/eval; S=/training_data/runs_receipts/eval; mkdir -p $E $S; L=$E/eval.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L; }
export_onnx() {  # <pt> <size> <name>: static ONNX + the sidecar the CLI reads (same exporter settings as the Phase-2 graphs)
  python - "$1" "$2" "$E/$3" <<'PY'
import shutil, sys, yaml
from ultralytics import YOLO
pt, sz, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = YOLO(pt)
f = m.export(format="onnx", imgsz=sz, batch=1, dynamic=False, half=False, simplify=True, device="cpu", verbose=False)
shutil.move(f, out + ".onnx")
yaml.safe_dump({"task": "detect", "imgsz": [sz, sz], "end2end": True, "names": {int(k): v for k, v in m.names.items()}},
               open(out + ".metadata.yaml", "w"), sort_keys=False)
print("exported", out + ".onnx")
PY
}
run() {  # <name> <pt> <size>
  n=$1; pt=$2; sz=$3
  [ -f $E/$n.onnx ] || export_onnx $pt $sz $n >> $E/export_$n.log 2>&1 || { log "EXPORT FAILED $n"; return; }
  rm -rf $E/dumpml_$n
  timeout 3600 $BIN -m $E/$n.onnx -s $IMG -b trt --precision fp16 --conf 0.001 --iou 0.7 --multi-label --save-txt $E/dumpml_$n --no-save --quiet 2>&1 | grep -E "^\[summary\]|\[trt\] built" | sed "s/^/  $n: /" | tee -a $L
  python scripts/ds_yolo/txt_to_coco_eval.py $E/dumpml_$n --ann $ANN 2>&1 | grep -E "images_with|PYCOCO" | sed "s/^/  $n: /" | tee -a $L
  cp $E/dumpml_${n}_coco.json $S/ 2>/dev/null
}
log "eval start ($W.pt) on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
run yolo26m_640 /data/weights/yolo26m.pt 640          # protocol check: the seminar's public rows are 0.5261 (640)
run yolo26l_640 /data/weights/yolo26l.pt 640          # 0.5417: the splice gate's reference
run splice10_${W}_640 /data/runs_receipts/splice10/weights/$W.pt 640            # gate G1 >= 0.5337
run twoscale20_${W}_512 /data/runs_receipts/twoscale20/weights/$W.pt 512        # gate G2 inputs (mixed by receipt_mix.py)
run twoscale20_${W}_768 /data/runs_receipts/twoscale20/weights/$W.pt 768
cp $L $S/; log "eval done"
