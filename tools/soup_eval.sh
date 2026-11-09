#!/bin/bash
# Evaluate weight-averaged models against the champion, with the autoloop's protocol:
#   gate:  400 games head to head vs the champion (200 seeds, seats swapped), 150-turn cap, one shared fresh seed block
#   guard: 200 games vs retail Crazy on the loop's fixed seeds (1,000,000 onwards)
#   water: 200 games vs retail Crazy on Water World only (not in the main suite), for every model including the champion
# Runs at most JOBS matches at once, 8 threads each, under nice. Matches skip games already in their output files,
# so the script can be rerun after an interruption.
# Usage: tools/soup_eval.sh CHAMPION_CKPT OUT_DIR GATE_SEED_START MODEL_CKPT...
set -u
CHAMP=$1 OUT=$2 GSEED=$3; shift 3
JOBS=${JOBS:-4}
PY=/kvm/polytopia/.venv/bin/python
mkdir -p "$OUT"
export TMPDIR=/kvm/tmp/tmp PYTHONPATH=/kvm/polytopia/ai/python OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /kvm/polytopia/ai/python || exit 1
name() { basename "$(dirname "$1")"; }
{
  for m in "$@"; do
    n=$(name "$m")
    echo "-m polyai.h2h --a $m --b $CHAMP --seeds 200 --seed-start $GSEED --mode might --turn-limit 150 --out $OUT/${n}_gate.jsonl"
    echo "-m polyai.arena --model $m --opponent retail:crazy --seeds 100 --seed-start 1000000 --mode might --turn-limit 150 --out $OUT/${n}_crazy.jsonl"
  done
  for m in "$@" "$CHAMP"; do
    n=$(name "$m")
    echo "-m polyai.arena --model $m --opponent retail:crazy --seeds 100 --seed-start 1100000 --mode might --turn-limit 150 --presets waterworld --out $OUT/${n}_water.jsonl"
  done
} | xargs -P "$JOBS" -I{} sh -c "nice -n 10 ionice -c3 $PY {} --threads 8 --device cpu >> $OUT/run.log 2>&1"
echo "$(date -Is) done" >> "$OUT/run.log"
