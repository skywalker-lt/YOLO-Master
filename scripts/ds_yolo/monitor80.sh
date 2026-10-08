#!/usr/bin/env bash
# Clean, tail-friendly log of the three runs: one progress line per run every minute, one line per completed epoch.
# Writes /data/$PROJECT/train.log (pod) and mirrors it to /training_data/logs/$PROJECT.log (shared volume); PROJECT and RUNS from the environment.
#   tail -F /data/runs80/train.log          # on the pod
#   tail -F /training_data/logs/runs80.log  # from any pod with the shared volume
PROJECT=${PROJECT:-runs80}; LOCAL=/data/$PROJECT/train.log; SHARED=/training_data/logs/$PROJECT.log; mkdir -p /data/$PROJECT /training_data/logs
RUNS=${RUNS:-"dense80 top180 pair680"}; declare -A DONE LASTERR DEAD
log() { echo "$(date -u +%FT%TZ) $*" >> $LOCAL; }
log "monitor started (runs: $RUNS)"
while true; do
  for n in $RUNS; do
    out=/data/$PROJECT/$n.out; [ -f $out ] || continue
    line=$(tail -c 4000 $out | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -a -E "^ +[0-9]+/[0-9]+ " | tail -1)
    if [ -n "$line" ]; then
      ep=$(echo "$line" | awk '{print $1}'); box=$(echo "$line" | awk '{print $3}'); cls=$(echo "$line" | awk '{print $4}')
      prog=$(echo "$line" | grep -oE "[0-9]+/[0-9]+ [0-9.]+(it/s|s/it) [0-9:]+<[0-9:]+" | tail -1)
      log "$n epoch $ep  box $box cls $cls  $prog"
    fi
    err=$(tail -c 20000 $out | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -a -E "Traceback|Error|BACKUP" | tail -1)
    [ -n "$err" ] && [ "${LASTERR[$n]:-}" != "$err" ] && { log "$n NOTE: $err"; LASTERR[$n]="$err"; }
    csv=/data/$PROJECT/$n/results.csv
    if [ -f $csv ]; then
      rows=$(( $(wc -l < $csv) - 1 ))
      if [ "$rows" != "${DONE[$n]:-0}" ]; then
        awk -F, -v n="$n" -v from="${DONE[$n]:-0}" 'NR==1{for(i=1;i<=NF;i++){k=$i; gsub(/ /,"",k); if(k=="epoch")e=i; if(k=="metrics/mAP50-95(B)")a=i; if(k=="metrics/mAP50(B)")b=i; if(k=="time")t=i}} NR>1+from{printf "%s EPOCH DONE %s  mAP50-95 %.4f  mAP50 %.4f  elapsed %.0f s\n", n, $e, $a, $b, $t}' $csv | while read -r l; do log "$l"; done
        DONE[$n]=$rows
      fi
    fi
    pgrep -f "^python scripts/ds_yolo/train_80ep.py .*--name $n\b" > /dev/null || { [ "${DEAD[$n]:-}" = 1 ] || { log "$n PROCESS NOT RUNNING"; DEAD[$n]=1; }; }
  done
  cp $LOCAL $SHARED 2>/dev/null
  sleep 60
done
