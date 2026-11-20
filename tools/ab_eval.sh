#!/usr/bin/env bash
# Evaluates the arms of an A/B: every pair of arms head to head, every arm against a reference model,
# and every arm plus the reference against retail Crazy. 400 games per match (200 seeds, seats swapped),
# run JOBS at a time. Each match file is resumable: finished games are skipped on a rerun.
#
# Usage: ab_eval.sh OUT_DIR REF_NAME=CKPT ARM_NAME=CKPT [ARM_NAME=CKPT ...]
# Env: ENGINE_DIR (folder with the polyai_engine bindings, put first on PYTHONPATH), AI_DIR (the ai checkout,
#      default /kvm/polytopia/ai), DEVICE (default cpu, as the autoloop gate; cuda is about 1.8x faster on poly4090
#      and plays some games differently, so keep one device within a comparison), JOBS (default 6),
#      NICE (default 15), H2H_SEED (default 3000000), CRAZY_SEED (default 1000000, the autoloop guard's seeds).
# Results: OUT_DIR/<a>_v_<b>.jsonl (a in seat `swap`), OUT_DIR/<arm>_crazy.jsonl; OUT_DIR/done.txt logs each exit.
set -u
OUT=$1; shift
REF=$1; shift
ARMS=("$@")
export OUT PY=/kvm/polytopia/.venv/bin/python NICE=${NICE:-15} H2H_SEED=${H2H_SEED:-3000000} \
  CRAZY_SEED=${CRAZY_SEED:-1000000} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  AI_DIR=${AI_DIR:-/kvm/polytopia/ai} DEVICE=${DEVICE:-cpu}
export PYTHONPATH=${ENGINE_DIR:+$ENGINE_DIR:}$AI_DIR/python
if [ -d /kvm/tmp/tmp ]; then export TMPDIR=/kvm/tmp/tmp; fi
mkdir -p "$OUT"

job() {
  local name=$1 a=$2 b=$3
  cd "$AI_DIR/python"
  if [ "$b" = crazy ]; then
    nice -n "$NICE" $PY -m polyai.arena --model "$a" --opponent retail:crazy --seeds 200 --seed-start "$CRAZY_SEED" \
      --mode might --threads 8 --device "$DEVICE" --out "$OUT/$name.jsonl" >> "$OUT/$name.log" 2>&1
  else
    nice -n "$NICE" $PY -m polyai.h2h --a "$a" --b "$b" --seeds 200 --seed-start "$H2H_SEED" \
      --mode might --threads 8 --device "$DEVICE" --out "$OUT/$name.jsonl" >> "$OUT/$name.log" 2>&1
  fi
  echo "$(date -u +%FT%TZ) $name exit $?" >> "$OUT/done.txt"
}
export -f job

{
  n=${#ARMS[@]}
  for ((i = 0; i < n; i++)); do
    for ((j = 0; j < i; j++)); do
      echo "${ARMS[i]%%=*}_v_${ARMS[j]%%=*} ${ARMS[i]#*=} ${ARMS[j]#*=}"
    done
  done
  for arm in "${ARMS[@]}"; do echo "${arm%%=*}_v_${REF%%=*} ${arm#*=} ${REF#*=}"; done
  for arm in "${ARMS[@]}" "$REF"; do echo "${arm%%=*}_crazy ${arm#*=} crazy"; done
} | xargs -P "${JOBS:-6}" -L 1 bash -c 'job "$@"' _
echo "$(date -u +%FT%TZ) ALL DONE" >> "$OUT/done.txt"
