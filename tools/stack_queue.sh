#!/usr/bin/env bash
# Second seeds and stacking (gameplan v3: action-features, belief-aux).
# 1. Waits for the seed-1 feature A/B (ab_plain_s1, ab_feat_s1), then evaluates it:
#    both vs retail Hard and Crazy, and features vs plain head to head.
# 2. Trains two more arms on the might1 config, two at a time on the 3060:
#    belief_s1     belief loss, seed 1 (control: ab_plain_s1)
#    featbelief_s0 features + belief loss, seed 0 (compare with featrl and beliefrl)
# 3. Evaluates them, 400 games per match, while recipe_queue.sh trains the recipe arms.
set -u
PY=/kvm/polytopia/.venv/bin/python
DATA=/kvm/polytopia-ai-data
RUNS=$DATA/runs
OUT=$DATA/arena/stack
export TMPDIR=/kvm/tmp/tmp
mkdir -p "$OUT"
cd /kvm/polytopia/ai/python

BASE="--init $RUNS/rl3_b6c96/ckpt_u00480.pt --ref $RUNS/bc3_b6c96_v11/ckpt_final.pt --runs-dir $RUNS
 --updates 300 --games 384 --rollout-steps 32768 --n-players 2:55,3:25,4:20 --turn-limit 150 --mode might
 --pool-weights retail:easy=0.05,retail:normal=0.15,retail:hard=0.15,retail:crazy=0.15,bc=0.15,self=0.35
 --lr 3e-05 --ent-coef 0.0005 --kl-coef 0.02 --kl-coef-end 0.02 --ckpt-every 20
 --threads 12 --torch-threads 4 --need-gb 1.5"

declare -A ARM
ARM[belief_s1]="--seed 1 --belief-weight 0.5 --belief-warmup 5"
ARM[featbelief_s0]="--seed 0 --features --belief-weight 0.5 --belief-warmup 5"

wait_names() { for n in "$@"; do while pgrep -f -- "--name $n " >/dev/null; do sleep 60; done; done; }

train() {
  local n=$1
  [ -f "$RUNS/$n/ckpt_final.pt" ] && return 0
  nice -n 10 $PY -m polyai.rl --name "$n" $BASE ${ARM[$n]} > "$RUNS/$n.log" 2>&1
}

retail() {  # tag ckpt
  nice -n 10 $PY -m polyai.arena --model "$2" --opponent retail:hard,retail:crazy --seeds 200 \
    --mode might --threads 6 --device cpu --out "$OUT/$1_vs_retail.jsonl" > "$OUT/$1_vs_retail.log" 2>&1
}

h2h() {  # tag a b
  nice -n 10 $PY -m polyai.h2h --a "$2" --b "$3" --seeds 200 --mode might --threads 6 \
    --device cpu --out "$OUT/h2h_$1.jsonl" > "$OUT/h2h_$1.log" 2>&1
}

F=ckpt_final.pt
wait_names ab_plain_s1 ab_feat_s1
echo "$(date -Is) seed-1 A/B finished; evaluating and starting belief_s1, featbelief_s0"
( retail ab_plain_s1 $RUNS/ab_plain_s1/$F & retail ab_feat_s1 $RUNS/ab_feat_s1/$F &
  h2h feat_s1_vs_plain_s1 $RUNS/ab_feat_s1/$F $RUNS/ab_plain_s1/$F & wait ) &
train belief_s1 & train featbelief_s0 & wait
echo "$(date -Is) training done; starting recipe_queue.sh on the GPU, evaluating here"
WAIT_FOR=none bash /kvm/polytopia/ai/tools/recipe_queue.sh > "$DATA/arena/recipe/queue.log" 2>&1 &
retail belief_s1 $RUNS/belief_s1/$F &
retail featbelief_s0 $RUNS/featbelief_s0/$F &
h2h belief_s1_vs_plain_s1 $RUNS/belief_s1/$F $RUNS/ab_plain_s1/$F &
h2h featbelief_vs_belief $RUNS/featbelief_s0/$F $RUNS/beliefrl_b6c96/$F &
h2h featbelief_vs_feat $RUNS/featbelief_s0/$F $RUNS/featrl_b6c96/$F &
wait
echo "$(date -Is) done"
