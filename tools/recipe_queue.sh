#!/usr/bin/env bash
# Recipe A/Bs (gameplan v3: recipe-ema, recipe-entropy, recipe-anchor, recipe-batch).
# Waits for the runs named in WAIT_FOR to exit, then trains the recipe arms two at a
# time on the might1 config (seed 0, 300 updates, init rl3 u480), each with --ema so
# every run also gives an EMA-vs-live pair. Then evaluates each arm against retail
# Hard and Crazy and head to head against might1 u300 (the control), 400 games each.
set -u
PY=/kvm/polytopia/.venv/bin/python
DATA=/kvm/polytopia-ai-data
RUNS=$DATA/runs
OUT=$DATA/arena/recipe
WAIT_FOR=${WAIT_FOR:-"ab_plain_s1 ab_feat_s1"}
CONTROL=$RUNS/might1_b6c96/ckpt_u00300.pt
export TMPDIR=/kvm/tmp/tmp
mkdir -p "$OUT"
cd /kvm/polytopia/ai/python

BASE="--init $RUNS/rl3_b6c96/ckpt_u00480.pt --ref $RUNS/bc3_b6c96_v11/ckpt_final.pt --runs-dir $RUNS
 --updates 300 --games 384 --rollout-steps 32768 --n-players 2:55,3:25,4:20 --turn-limit 150 --mode might
 --pool-weights retail:easy=0.05,retail:normal=0.15,retail:hard=0.15,retail:crazy=0.15,bc=0.15,self=0.35
 --lr 3e-05 --ent-coef 0.0005 --kl-coef 0.02 --kl-coef-end 0.02 --ckpt-every 20 --seed 0
 --threads 12 --torch-threads 4 --need-gb 1.5 --ema 0.999"

declare -A ARM
ARM[rec_ema]=""
ARM[rec_ent0]="--ent-schedule --ent-coef-end 0 --ent-decay-updates 300"
ARM[rec_anchor]="--anchor-every 20"
ARM[rec_advf]="--adv-filter 0.5"

wait_names() { for n in "$@"; do while pgrep -f -- "--name $n " >/dev/null; do sleep 60; done; done; }

train() {
  local n=$1
  [ -f "$RUNS/$n/ckpt_final.pt" ] && return 0
  nice -n 10 $PY -m polyai.rl --name "$n" $BASE ${ARM[$n]} > "$RUNS/$n.log" 2>&1
}

evaluate() {
  local n=$1 ck
  for ck in ckpt_final ckpt_ema_final; do
    local p=$RUNS/$n/$ck.pt tag=${n}_${ck#ckpt_}
    [ -f "$p" ] || continue
    nice -n 10 $PY -m polyai.arena --model "$p" --opponent retail:hard,retail:crazy --seeds 200 \
      --mode might --threads 6 --device cpu --out "$OUT/$tag.jsonl" > "$OUT/$tag.log" 2>&1 &
    nice -n 10 $PY -m polyai.h2h --a "$p" --b "$CONTROL" --seeds 200 --mode might --threads 6 \
      --device cpu --out "$OUT/h2h_${tag}_vs_might1u300.jsonl" > "$OUT/h2h_${tag}.log" 2>&1 &
    wait
  done
}

wait_names $WAIT_FOR
echo "$(date -Is) start pair 1"
train rec_ema & train rec_ent0 & wait
evaluate rec_ema & evaluate rec_ent0 &
echo "$(date -Is) start pair 2"
train rec_anchor & train rec_advf & wait
wait
evaluate rec_anchor & evaluate rec_advf & wait
echo "$(date -Is) done"
