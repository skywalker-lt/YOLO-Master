#!/usr/bin/env bash
# Dump a list of models under the seminar protocol (CLI dumpml on the pod's GPU, pycocotools) and copy the COCO json files
# to the shared volume. Spec file lines: <name> <checkpoint.pt> <size> <o2o|o2m>   (o2m = one-to-many head + the CLI's NMS)
#   bash dump_batch.sh <spec file> <out dir name>      -> /data/runs_receipts/<out>/, /training_data/runs_receipts/<out>/
source /root/anaconda3/etc/profile.d/conda.sh; conda activate yolo_master
cd /data/YOLO-Master; export PYTHONPATH=/data/YOLO-Master
BIN=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0/yolomaster_edge
IMG=/data/datasets/coco/images/val2017; ANN=/data/datasets/coco/annotations/instances_val2017.json
SPEC=$1; E=/data/runs_receipts/$2; S=/training_data/runs_receipts/$2; mkdir -p $E $S; L=$E/dump.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L; }
export_onnx() {  # <pt> <size> <out path without extension> <o2o|o2m>
  python - "$1" "$2" "$3" "$4" <<'PY'
import shutil, sys, yaml
from ultralytics import YOLO
pt, sz, out, head = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
m = YOLO(pt)
e2e = head == "o2o"
m.model.model[-1].end2end = e2e
f = m.export(format="onnx", imgsz=sz, batch=1, dynamic=False, half=False, simplify=True, device="cpu", verbose=False)
shutil.move(f, out + ".onnx")
yaml.safe_dump({"task": "detect", "imgsz": [sz, sz], "end2end": e2e, "names": {int(k): v for k, v in m.names.items()}},
               open(out + ".metadata.yaml", "w"), sort_keys=False)
print("exported", out + ".onnx", "end2end", e2e)
PY
}
while read -r n pt sz head; do
  [ -z "$n" ] && continue; case "$n" in \#*) continue;; esac
  [ -f $E/$n.onnx ] || export_onnx $pt $sz $E/$n $head >> $E/export_$n.log 2>&1 || { log "EXPORT FAILED $n"; continue; }
  rm -rf $E/dumpml_$n
  timeout 3600 $BIN -m $E/$n.onnx -s $IMG -b trt --precision fp16 --conf 0.001 --iou 0.7 --multi-label --save-txt $E/dumpml_$n --no-save --quiet 2>&1 | grep -E "^\[summary\]|\[trt\] built" | sed "s/^/  $n: /" | tee -a $L
  python scripts/ds_yolo/txt_to_coco_eval.py $E/dumpml_$n --ann $ANN 2>&1 | grep -E "images_with|PYCOCO" | sed "s/^/  $n: /" | tee -a $L
  cp $E/dumpml_${n}_coco.json $S/ 2>/dev/null
done < $SPEC
cp $L $S/; log "batch done"
