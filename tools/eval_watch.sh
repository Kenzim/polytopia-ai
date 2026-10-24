#!/usr/bin/env bash
# Put a run's checkpoints through the 1v1 suite as they appear: head to head against a reference checkpoint and
# against retail Hard and Crazy, three processes of 8 threads each, all under nice.
#   tools/eval_watch.sh RUN_DIR REF_CKPT OUT_DIR [EVERY] [FIRST] [SEEDS]
set -u
run=$1 ref=$2 out=$3 every=${4:-40} next=${5:-240} seeds=${6:-200}
py=/kvm/polytopia/.venv/bin/python
export TMPDIR=/kvm/tmp/tmp
cd "$(dirname "$0")/../python"
mkdir -p "$out"
while :; do
  ck=$(printf '%s/ckpt_u%05d.pt' "$run" "$next")
  if [ ! -f "$ck" ]; then
    [ -f "$run/ckpt_final.pt" ] && [ "$next" -gt "$(ls "$run"/ckpt_u*.pt | sed 's/.*ckpt_u0*//; s/\.pt//' | sort -n | tail -1)" ] && break
    sleep 120; continue
  fi
  sleep 5  # let the writer finish
  tag=$(printf 'u%05d' "$next")
  echo "$(date -Is) eval $tag"
  nice -n 15 $py -m polyai.h2h --a "$ck" --b "$ref" --seeds "$seeds" --device cpu --threads 8 \
    --out "$out/h2h_$tag.jsonl" > "$out/h2h_$tag.log" 2>&1 &
  nice -n 15 $py -m polyai.arena --model "$ck" --opponent retail:hard --seeds "$seeds" --device cpu --threads 8 \
    --out "$out/retail_hard_$tag.jsonl" > "$out/retail_hard_$tag.log" 2>&1 &
  nice -n 15 $py -m polyai.arena --model "$ck" --opponent retail:crazy --seeds "$seeds" --device cpu --threads 8 \
    --out "$out/retail_crazy_$tag.jsonl" > "$out/retail_crazy_$tag.log" 2>&1 &
  wait
  grep -h ' vs ' "$out"/h2h_$tag.log "$out"/retail_*_$tag.log | grep 'W-L-D' | sed "s/^/$tag /"
  next=$((next + every))
done
echo "$(date -Is) done"
