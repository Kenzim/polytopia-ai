#!/bin/bash
# Sixth A/B on /kvm/ab2 (same build as abk, plus arena --n-players from ai 7489e52), control abk_1v1
# (loop pool, two-player games only, seed 900).
# Why: two-player-only training (abk_1v1 53.8% and 53.4% against the two controls, and the exploiter) is the one
# change besides engine features that beat the control. Before the loop drops its 3- and 4-player games, measure
# what that costs in free-for-all games, which the website can seat. The trainer's learning rate (3e-5) has never
# been varied, so the two new arms test it on the two-player recipe.
# abl_lr1e4: abk_1v1 with --lr 1e-4.
# abl_lr1e5: abk_1v1 with --lr 1e-5.
# eval-mp: each model against retail Crazy in 3-player (600 games) and 4-player (800 games) Might free-for-alls,
#          200 seeds from 1,000,000, the model once in every seat.
D=/kvm/polytopia-ai-data/ab; L=/kvm/polytopia-ai-data/runs; M=$D/eval-mp
/kvm/ab2/run.sh abl_lr1e4 --features --n-players 2:100 --lr 1e-4 &
/kvm/ab2/run.sh abl_lr1e5 --features --n-players 2:100 --lr 1e-5 &

mkdir -p $M
export M PYTHONPATH=/kvm/ab2/ext:/kvm/ab2/ai/python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mpjob() {
  local name=$1 ckpt=$2 n=$3
  cd /kvm/ab2/ai/python
  nice -n 15 /kvm/polytopia/.venv/bin/python -m polyai.arena --model "$ckpt" --opponent retail:crazy --seeds 200 \
    --seed-start 1000000 --n-players "$n" --mode might --threads 8 --device cuda \
    --out "$M/${name}_crazy$n.jsonl" >> "$M/${name}_crazy$n.log" 2>&1
  echo "$(date -u +%FT%TZ) ${name}_crazy$n exit $?" >> "$M/done.txt"
}
export -f mpjob
mp() {
  for n in 4 3; do for a in "$@"; do echo "${a%%=*} ${a#*=} $n"; done; done | xargs -P 3 -L 1 bash -c 'mpjob "$@"' _
}
mp r77=$L/auto_r0077/ckpt_final.pt ctrl=$D/runs/abg_ctrl/ckpt_final.pt ctrl2=$D/runs/abk_ctrl2/ckpt_final.pt \
  1v1=$D/runs/abk_1v1/ckpt_final.pt expl2=$D/runs/abk_expl2/ckpt_final.pt
echo "$(date -u +%FT%TZ) MP DONE" >> $M/done.txt

wait
echo "$(date -u +%FT%TZ) abl arms done"
mkdir -p $D/eval-abl
for f in 1v1_v_ctrl ctrl_v_r77 1v1_v_r77 ctrl_crazy 1v1_crazy r77_crazy; do cp -n $D/eval-abk/$f.jsonl $D/eval-abl/; done
ENGINE_DIR=/kvm/ab2/ext AI_DIR=/kvm/ab2/ai DEVICE=cuda JOBS=3 /kvm/ab2/ai/tools/ab_eval.sh $D/eval-abl \
  r77=$L/auto_r0077/ckpt_final.pt ctrl=$D/runs/abg_ctrl/ckpt_final.pt 1v1=$D/runs/abk_1v1/ckpt_final.pt \
  lr1e4=$D/runs/abl_lr1e4/ckpt_final.pt lr1e5=$D/runs/abl_lr1e5/ckpt_final.pt
echo "$(date -u +%FT%TZ) eval-abl done"
mp lr1e4=$D/runs/abl_lr1e4/ckpt_final.pt lr1e5=$D/runs/abl_lr1e5/ckpt_final.pt
echo "$(date -u +%FT%TZ) MP2 DONE" >> $M/done.txt
