#!/bin/bash
# Eighth A/B on /kvm/ab2 (same build as abm), recipe abk_1v1 (loop pool, two-player games only, features, seed 900).
# abm found that 65,536 decisions an update (abm_b64k) beats the same games and gradient steps at 32,768 (abm_u600)
# 54.9%. With 384 parallel games, doubling the rollout also doubles how far each game runs between updates (about 85
# to 170 decisions a game), so less of the advantage comes from the value's bootstrap. These arms split the two, each
# with abm_b64k's games and gradient steps (19.7M decisions, about 77,000 minibatches of 256):
# abn_b128k: --rollout-steps 131072, 150 updates (a bigger batch again; segments of about 340 decisions).
# abn_g192:  --games 192, 600 updates at 32,768 (abm_b64k's segment length, abm_u600's batch size).
# Reading: g192 level with b64k means segment length matters; level with u600 means batch size does.
D=/kvm/polytopia-ai-data/ab; L=/kvm/polytopia-ai-data/runs; M=$D/eval-mp
/kvm/ab2/run.sh abn_b128k --features --n-players 2:100 --rollout-steps 131072 --updates 150 &
/kvm/ab2/run.sh abn_g192 --features --n-players 2:100 --games 192 --updates 600 &

export M PYTHONPATH=/kvm/ab2/ext:/kvm/ab2/ai/python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mpjob() {
  local name=$1 ckpt=$2 n=$3
  cd /kvm/ab2/ai/python
  nice -n 15 /kvm/polytopia/.venv/bin/python -m polyai.arena --model "$ckpt" --opponent retail:crazy --seeds 200 \
    --seed-start 1000000 --n-players "$n" --mode might --threads 8 --device cuda \
    --out "$M/${name}_crazy$n.jsonl" >> "$M/${name}_crazy$n.log" 2>&1
  echo "$(date -u +%FT%TZ) ${name}_crazy$n exit $?" >> "$M/done.txt"
}
mpjob b64k $D/runs/abm_b64k/ckpt_final.pt 4
mpjob b64k $D/runs/abm_b64k/ckpt_final.pt 3

wait
echo "$(date -u +%FT%TZ) abn arms done"
mkdir -p $D/eval-abn
for f in u600_v_1v1 b64k_v_1v1 b64k_v_u600 1v1_v_r77 u600_v_r77 b64k_v_r77 1v1_crazy u600_crazy b64k_crazy r77_crazy; do
  cp -n $D/eval-abm/$f.jsonl $D/eval-abn/
done
ENGINE_DIR=/kvm/ab2/ext AI_DIR=/kvm/ab2/ai DEVICE=cuda JOBS=3 /kvm/ab2/ai/tools/ab_eval.sh $D/eval-abn \
  r77=$L/auto_r0077/ckpt_final.pt 1v1=$D/runs/abk_1v1/ckpt_final.pt u600=$D/runs/abm_u600/ckpt_final.pt \
  b64k=$D/runs/abm_b64k/ckpt_final.pt b128k=$D/runs/abn_b128k/ckpt_final.pt g192=$D/runs/abn_g192/ckpt_final.pt
echo "$(date -u +%FT%TZ) eval-abn done"
